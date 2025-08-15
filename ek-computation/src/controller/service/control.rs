use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use ek_db::{safetensor::ExpertKey, weight_srv::client::WeightSrvClient};
use rand::random;
use tokio::task::JoinSet;

use crate::{
    controller::registry::get_registry, proto::ek::control::v1::{self}, state::{
        io::StateReaderImpl,
        models::{NewExpert, NewInstance},
        writer::StateWriterImpl,
    }
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
        let registry = get_registry();
        registry.lock().await.reset().await?;
        let resp = v1::RebalanceResp {};
        Ok(tonic::Response::new(resp))
    }

    async fn duplicate(
        &self,
        request: tonic::Request<v1::DuplicateReq>,
    ) -> Result<tonic::Response<v1::DuplicateResp>, tonic::Status> {
        let req = request.into_inner();
        execute_duplicate_schedule(req.hostnames).await?;
        let registry = get_registry();
        registry.lock().await.reset().await?;
        let resp = v1::DuplicateResp {};
        Ok(tonic::Response::new(resp))
    }

    async fn manual(
        &self,
        request: tonic::Request<v1::ManualReq>,
    ) -> Result<tonic::Response<v1::ManualResp>, tonic::Status> {
        let req = request.into_inner();
        execute_manual_schedule(req.hostnames, req.layers).await?;
        let registry = get_registry();
        registry.lock().await.reset().await?;
        let resp = v1::ManualResp {};
        Ok(tonic::Response::new(resp))
    }
}

async fn execute_rebalance() -> EKResult<()> {
    // implement the static scheduling logic here
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings.weight.server.as_ref().unwrap().addr.clone();
    log::info!(
        "Running static schedule for model: {model_name}, instance: {instance_name}, weight server: {ws_addr}"
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
    let node_ids = reader
        .active_nodes()
        .await?
        .into_iter()
        .map(|x| x.id)
        .collect::<Vec<_>>();

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
    log::info!("total experts to schedule {}", experts.len());

    writer.expert_del_by_instance(instance_obj.id).await?;

    let mut js = JoinSet::new();
    for e in experts {
        let e = e.clone();
        let node_ids = node_ids.clone();
        js.spawn(async move {
            let writer = StateWriterImpl::new();
            let rand = random::<u16>();
            writer
                .expert_upsert(NewExpert {
                    instance_id: instance_obj.id,
                    node_id: node_ids[(rand % node_ids.len() as u16) as usize],
                    expert_id: e.as_object_key(),
                    replica: 1,
                    state: serde_json::json!({}),
                })
                .await
                .unwrap();
        });
    }
    js.join_all().await;
    log::info!("all experts scheduled");

    Ok(())
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
        return Err(EKError::NotFound("No matching nodes found for the specified hostnames".to_string()));
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
    log::info!("total experts to schedule: {}, target nodes: {}", experts.len(), node_ids.len());
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
            
            let start: u32 = parts[0].parse().map_err(|_| {
                EKError::InvalidInput(format!("Invalid number: {}", parts[0]))
            })?;
            let end: u32 = parts[1].parse().map_err(|_| {
                EKError::InvalidInput(format!("Invalid number: {}", parts[1]))
            })?;
            
            if start > end {
                return Err(EKError::InvalidInput(format!(
                    "Invalid range: {start} > {end}. Start must be <= end"
                )));
            }
            
            for layer in start..=end {
                layers.push(layer);
            }
        } else {
            let layer: u32 = range_part.parse().map_err(|_| {
                EKError::InvalidInput(format!("Invalid number: {range_part}"))
            })?;
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
        return Err(EKError::NotFound("No matching nodes found for the specified hostnames".to_string()));
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
    log::info!("Removing existing experts from {} target nodes", target_nodes.len());
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
                layer, vital.moe_layers.0, vital.moe_layers.1 - 1
            )));
        }
        
        for expert in 0..vital.routed_experts {
            experts_to_assign.push(ExpertKey::new(model_name.clone(), layer, expert));
        }
    }
    
    log::info!("Assigning {} experts to each of {} target nodes", experts_to_assign.len(), target_nodes.len());

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
    
    log::info!("Manual assignment completed: assigned specified layers to {} nodes", target_nodes.len());
    Ok(())
}
