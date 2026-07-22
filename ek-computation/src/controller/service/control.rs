use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use ek_db::{safetensor::ExpertKey, weight_srv::client::WeightSrvClient};
use tokio::task::JoinSet;

use crate::{
    controller::{poller::request_immediate_poll, scheduler::max_experts},
    proto::ek::control::v1::{self},
    state::{
        io::StateReaderImpl,
        models::{NewExpert, NewInstance, Node},
        writer::StateWriterImpl,
    },
};
pub struct PlanServiceImpl {}

impl Default for PlanServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanServiceImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl v1::plan_service_server::PlanService for PlanServiceImpl {
    async fn rebalance(
        &self,
        _request: tonic::Request<v1::RebalanceReq>,
    ) -> Result<tonic::Response<v1::RebalanceResp>, tonic::Status> {
        execute_rebalance().await?;
        let resp = v1::RebalanceResp {};
        Ok(tonic::Response::new(resp))
    }

    async fn duplicate(
        &self,
        request: tonic::Request<v1::DuplicateReq>,
    ) -> Result<tonic::Response<v1::DuplicateResp>, tonic::Status> {
        let req = request.into_inner();
        execute_duplicate_schedule(req.hostnames).await?;
        let resp = v1::DuplicateResp {};
        Ok(tonic::Response::new(resp))
    }

    async fn manual(
        &self,
        request: tonic::Request<v1::ManualReq>,
    ) -> Result<tonic::Response<v1::ManualResp>, tonic::Status> {
        let req = request.into_inner();
        execute_manual_schedule(req.hostnames, req.layers).await?;
        let resp = v1::ManualResp {};
        Ok(tonic::Response::new(resp))
    }
}

async fn execute_rebalance() -> EKResult<()> {
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings
        .weight
        .server
        .as_ref()
        .ok_or_else(|| EKError::RuntimeError("rebalance requires a weight server".to_owned()))?
        .addr
        .clone();
    log::info!(
        "Running rebalance for model: {model_name}, instance: {instance_name}, weight server: {ws_addr}"
    );
    let cli = WeightSrvClient::new(ws_addr);
    let vital = cli.load_meta_vital(&model_name).await?;
    log::info!("model info : {:?}", &vital);

    let reader = StateReaderImpl::new();
    let model = reader
        .model_by_name(&model_name)
        .await?
        .ok_or(EKError::NotFound("model not found".to_string()))?;

    let writer = StateWriterImpl::new();
    let instance_obj = writer
        .instance_upsert(NewInstance {
            model_id: model.id,
            name: instance_name,
        })
        .await?;
    if instance_obj.model_id != model.id {
        return Err(EKError::RuntimeError(format!(
            "instance '{}' belongs to model {}, not configured model {}",
            instance_obj.name, instance_obj.model_id, model.id
        )));
    }

    let mut experts = vec![];
    for layer in vital.moe_layers.0..vital.moe_layers.1 {
        for expert in 0..vital.routed_experts {
            experts.push(ExpertKey::new(model_name.clone(), layer, expert));
        }
    }
    log::info!("total experts to schedule {}", experts.len());

    let active_nodes = reader.active_nodes().await?;
    let node_ids = plan_rebalance_nodes(instance_obj.id, &active_nodes, experts.len())?;
    let assignments = experts
        .into_iter()
        .zip(node_ids)
        .map(|(expert, node_id)| NewExpert {
            instance_id: instance_obj.id,
            node_id,
            expert_id: expert.as_object_key(),
            replica: 1,
            state: serde_json::json!({"status": "pending"}),
        })
        .collect::<Vec<_>>();
    let inserted = writer
        .replace_experts_for_instance(instance_obj.id, assignments)
        .await?;
    request_immediate_poll();
    log::info!("rebalance committed {inserted} expert assignments");

    Ok(())
}

fn plan_rebalance_nodes(
    instance_id: i32,
    active_nodes: &[Node],
    required_experts: usize,
) -> EKResult<Vec<i32>> {
    if instance_id <= 0 {
        return Err(EKError::RuntimeError(
            "instance ID must be positive".to_owned(),
        ));
    }
    if required_experts == 0 {
        return Err(EKError::RuntimeError(
            "model has no routed experts to rebalance".to_owned(),
        ));
    }
    let instance_id = u64::try_from(instance_id)
        .map_err(|_| EKError::RuntimeError("instance ID must be positive".to_owned()))?;
    let mut workers = active_nodes
        .iter()
        .filter(|node| {
            node.config.get("instance_id").and_then(|id| id.as_u64()) == Some(instance_id)
                && max_experts(node) > 0
        })
        .collect::<Vec<_>>();
    workers.sort_by(|left, right| {
        left.hostname
            .cmp(&right.hostname)
            .then_with(|| left.id.cmp(&right.id))
    });
    if workers.is_empty() {
        return Err(EKError::NotFound(
            "no active Worker reports capacity for the configured instance".to_owned(),
        ));
    }

    let capacities = workers
        .iter()
        .map(|node| max_experts(node))
        .collect::<Vec<_>>();
    let total_capacity = capacities.iter().try_fold(0_u64, |total, capacity| {
        total
            .checked_add(*capacity)
            .ok_or_else(|| EKError::RuntimeError("active Worker capacity overflow".to_owned()))
    })?;
    let required = u64::try_from(required_experts)
        .map_err(|_| EKError::RuntimeError("model expert count is too large".to_owned()))?;
    if total_capacity < required {
        return Err(EKError::RuntimeError(format!(
            "active Worker capacity {total_capacity} cannot cover {required} routed experts"
        )));
    }

    let mut loads = vec![0_u64; workers.len()];
    let mut planned = Vec::with_capacity(required_experts);
    for _ in 0..required_experts {
        let selected = (0..workers.len())
            .filter(|&index| loads[index] < capacities[index])
            .min_by(|&left, &right| {
                let left_ratio = u128::from(loads[left]) * u128::from(capacities[right]);
                let right_ratio = u128::from(loads[right]) * u128::from(capacities[left]);
                left_ratio.cmp(&right_ratio).then_with(|| left.cmp(&right))
            })
            .expect("aggregate capacity was validated before planning");
        loads[selected] += 1;
        planned.push(workers[selected].id);
    }

    for (worker, load) in workers.iter().zip(loads) {
        log::info!(
            "rebalance target worker={} assigned_experts={} max_experts={}",
            worker.hostname,
            load,
            max_experts(worker)
        );
    }
    Ok(planned)
}

async fn execute_duplicate_schedule(hostnames: Vec<String>) -> EKResult<()> {
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings.weight.server.as_ref().unwrap().addr.clone();
    log::info!(
        "Running duplicate schedule for model: {model_name}, instance: {instance_name}, weight server: {ws_addr}"
    );
    let cli = WeightSrvClient::new(ws_addr);
    let vital = cli.load_meta_vital(&model_name).await?;
    log::info!("model info : {:?}", &vital);

    let reader = StateReaderImpl::new();
    let model = reader
        .model_by_name(&model_name)
        .await?
        .ok_or(EKError::NotFound("model not found".to_string()))?;

    let writer = StateWriterImpl::new();
    let all_nodes = reader.active_nodes().await?;

    let node_ids = if hostnames.is_empty() {
        log::info!("No specific hostnames provided, duplicating to all active nodes");
        all_nodes.into_iter().map(|x| x.id).collect::<Vec<_>>()
    } else {
        log::info!("Filtering nodes by hostnames: {hostnames:?}");
        all_nodes
            .into_iter()
            .filter(|node| hostnames.contains(&node.hostname))
            .map(|x| x.id)
            .collect::<Vec<_>>()
    };

    if node_ids.is_empty() {
        return Err(EKError::NotFound(
            "No matching nodes found for the specified hostnames".to_string(),
        ));
    }

    let instance_obj = writer
        .instance_upsert(NewInstance {
            model_id: model.id,
            name: instance_name,
        })
        .await?;

    let mut experts = vec![];
    for layer in vital.moe_layers.0..vital.moe_layers.1 {
        for expert in 0..vital.routed_experts {
            experts.push(ExpertKey::new(model_name.clone(), layer, expert));
        }
    }
    log::info!(
        "total experts to schedule: {}, target nodes: {}",
        experts.len(),
        node_ids.len()
    );
    log::info!("duplicating all experts to {} nodes", node_ids.len());

    writer.expert_del_by_instance(instance_obj.id).await?;

    let mut js = JoinSet::new();
    for e in experts {
        for &node_id in &node_ids {
            let e = e.clone();
            js.spawn(async move {
                let writer = StateWriterImpl::new();
                writer
                    .expert_upsert(NewExpert {
                        instance_id: instance_obj.id,
                        node_id,
                        expert_id: e.as_object_key(),
                        replica: 1,
                        state: serde_json::json!({}),
                    })
                    .await
                    .unwrap();
            });
        }
    }
    js.join_all().await;
    log::info!("all experts duplicated to target nodes");

    Ok(())
}

fn parse_layer_ranges(layers_str: &str) -> EKResult<Vec<u32>> {
    let mut layers = Vec::new();

    for range_part in layers_str.split(',') {
        let range_part = range_part.trim();
        if range_part.contains('-') {
            let parts: Vec<&str> = range_part.split('-').collect();
            if parts.len() != 2 {
                return Err(EKError::InvalidInput(format!(
                    "Invalid range format: {range_part}. Expected format like '1-5'"
                )));
            }

            let start: u32 = parts[0]
                .parse()
                .map_err(|_| EKError::InvalidInput(format!("Invalid number: {}", parts[0])))?;
            let end: u32 = parts[1]
                .parse()
                .map_err(|_| EKError::InvalidInput(format!("Invalid number: {}", parts[1])))?;

            if start > end {
                return Err(EKError::InvalidInput(format!(
                    "Invalid range: {start} > {end}. Start must be <= end"
                )));
            }

            for layer in start..=end {
                layers.push(layer);
            }
        } else {
            let layer: u32 = range_part
                .parse()
                .map_err(|_| EKError::InvalidInput(format!("Invalid number: {range_part}")))?;
            layers.push(layer);
        }
    }

    layers.sort();
    layers.dedup();
    Ok(layers)
}

async fn execute_manual_schedule(hostnames: Vec<String>, layers_str: String) -> EKResult<()> {
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings.weight.server.as_ref().unwrap().addr.clone();
    log::info!(
        "Running manual schedule for model: {model_name}, instance: {instance_name}, target nodes: {hostnames:?}, layers: {layers_str}"
    );

    // Parse layer ranges
    let target_layers = parse_layer_ranges(&layers_str)?;
    log::info!("Parsed layers: {target_layers:?}");

    let cli = WeightSrvClient::new(ws_addr);
    let vital = cli.load_meta_vital(&model_name).await?;
    log::info!("model info : {:?}", &vital);

    let reader = StateReaderImpl::new();
    let model = reader
        .model_by_name(&model_name)
        .await?
        .ok_or(EKError::NotFound("model not found".to_string()))?;

    let writer = StateWriterImpl::new();
    let all_nodes = reader.active_nodes().await?;

    // Find target nodes by hostnames
    let target_nodes: Vec<_> = all_nodes
        .into_iter()
        .filter(|node| hostnames.contains(&node.hostname))
        .collect();

    if target_nodes.is_empty() {
        return Err(EKError::NotFound(
            "No matching nodes found for the specified hostnames".to_string(),
        ));
    }

    let found_hostnames: Vec<_> = target_nodes.iter().map(|n| &n.hostname).collect();
    log::info!("Target nodes found: {found_hostnames:?}");

    let instance_obj = writer
        .instance_upsert(NewInstance {
            model_id: model.id,
            name: instance_name,
        })
        .await?;

    // Clear experts on the target nodes
    log::info!(
        "Removing existing experts from {} target nodes",
        target_nodes.len()
    );
    for node in &target_nodes {
        writer.del_experts_by_node(node.id, instance_obj.id).await?;
    }

    // Generate experts only for the specified layers
    let mut experts_to_assign = Vec::new();
    for layer in target_layers {
        let layer = layer as usize;
        // Validate layer is within model bounds
        if layer < vital.moe_layers.0 || layer >= vital.moe_layers.1 {
            return Err(EKError::InvalidInput(format!(
                "Layer {} is out of bounds. Model supports layers {}-{}",
                layer,
                vital.moe_layers.0,
                vital.moe_layers.1 - 1
            )));
        }

        for expert in 0..vital.routed_experts {
            experts_to_assign.push(ExpertKey::new(model_name.clone(), layer, expert));
        }
    }

    log::info!(
        "Assigning {} experts to each of {} target nodes",
        experts_to_assign.len(),
        target_nodes.len()
    );

    // Assign experts to all target nodes
    let mut js = JoinSet::new();
    for expert_key in experts_to_assign {
        for target_node in &target_nodes {
            let expert_key = expert_key.clone();
            let node_id = target_node.id;
            js.spawn(async move {
                let writer = StateWriterImpl::new();
                writer
                    .expert_upsert(NewExpert {
                        instance_id: instance_obj.id,
                        node_id,
                        expert_id: expert_key.as_object_key(),
                        replica: 1,
                        state: serde_json::json!({}),
                    })
                    .await
                    .unwrap();
            });
        }
    }
    js.join_all().await;

    log::info!(
        "Manual assignment completed: assigned specified layers to {} nodes",
        target_nodes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn worker(id: i32, hostname: &str, instance_id: i32, capacity: u64) -> Node {
        Node {
            id,
            hostname: hostname.to_owned(),
            device: format!("cuda:{id}"),
            config: serde_json::json!({
                "instance_id": instance_id,
                "max_experts": capacity,
            }),
        }
    }

    fn counts(plan: &[i32]) -> HashMap<i32, usize> {
        let mut counts = HashMap::new();
        for node_id in plan {
            *counts.entry(*node_id).or_default() += 1;
        }
        counts
    }

    #[test]
    fn rebalance_requires_capacity_for_the_configured_instance() {
        let nodes = vec![worker(1, "wrong-instance", 8, 10)];
        let error = plan_rebalance_nodes(7, &nodes, 1).unwrap_err();

        assert!(matches!(error, EKError::NotFound(_)));
        assert!(error.to_string().contains("no active Worker"));
    }

    #[test]
    fn rebalance_rejects_an_empty_model_before_replacing_placement() {
        let nodes = vec![worker(1, "worker-a", 7, 10)];
        let error = plan_rebalance_nodes(7, &nodes, 0).unwrap_err();

        assert!(error.to_string().contains("no routed experts"));
    }

    #[test]
    fn rebalance_rejects_insufficient_aggregate_capacity() {
        let nodes = vec![worker(1, "worker-a", 7, 2), worker(2, "worker-b", 7, 1)];
        let error = plan_rebalance_nodes(7, &nodes, 4).unwrap_err();

        assert!(error.to_string().contains("capacity 3 cannot cover 4"));
    }

    #[test]
    fn rebalance_is_deterministic_and_balances_equal_workers() {
        let nodes = vec![worker(2, "worker-b", 7, 3), worker(1, "worker-a", 7, 3)];

        let first = plan_rebalance_nodes(7, &nodes, 6).unwrap();
        let second = plan_rebalance_nodes(7, &nodes, 6).unwrap();

        assert_eq!(first, vec![1, 2, 1, 2, 1, 2]);
        assert_eq!(second, first);
    }

    #[test]
    fn rebalance_distributes_by_reported_capacity_without_exceeding_it() {
        let nodes = vec![worker(1, "worker-a", 7, 4), worker(2, "worker-b", 7, 2)];

        let plan = plan_rebalance_nodes(7, &nodes, 6).unwrap();

        assert_eq!(plan.len(), 6);
        assert_eq!(counts(&plan), HashMap::from([(1, 4), (2, 2)]));
    }
}
