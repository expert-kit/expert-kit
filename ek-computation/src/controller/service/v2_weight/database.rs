//! Database and Dispatcher adapter for the v2 weight-control service.

use std::collections::{BTreeSet, HashMap};

use async_trait::async_trait;
use ek_db::safetensor::ExpertKey as StoredExpertKey;
use tokio::sync::mpsc;
use tonic::Status;

use crate::{
    controller::{dispatcher::DISPATCHER, runtime_state::ExpertKey},
    proto::ek::control::v2::{RegisterWorkerRequest, TargetExpert},
    state::{
        io::{StateReader, StateReaderImpl},
        models::{Expert, Node},
        writer::StateWriterImpl,
    },
};

use super::{TargetSubscription, WeightControlHooks};

const PLACEMENT_UPDATE_BUFFER: usize = 4;

pub struct DatabaseWeightControlHooks;

impl DatabaseWeightControlHooks {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DatabaseWeightControlHooks {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WeightControlHooks for DatabaseWeightControlHooks {
    async fn subscribe(
        &self,
        registration: &RegisterWorkerRequest,
    ) -> Result<TargetSubscription, Status> {
        let (subscription_id, mut raw_updates) = DISPATCHER
            .lock()
            .await
            .subscribe_with_lease(&registration.worker_id);
        let initial = match load_current_targets(registration).await {
            Ok(initial) => initial,
            Err(status) => {
                DISPATCHER
                    .lock()
                    .await
                    .unsubscribe_with_lease(&registration.worker_id, subscription_id);
                return Err(status);
            }
        };
        let registration = registration.clone();
        let (sender, updates) = mpsc::channel(PLACEMENT_UPDATE_BUFFER);
        tokio::spawn(async move {
            while raw_updates.recv().await.is_some() {
                let targets = load_current_targets(&registration).await;
                if sender.send(targets).await.is_err() {
                    return;
                }
            }
        });
        Ok(TargetSubscription {
            subscription_id,
            initial,
            updates,
        })
    }

    async fn unsubscribe(&self, worker_id: &str, subscription_id: u64) {
        DISPATCHER
            .lock()
            .await
            .unsubscribe_with_lease(worker_id, subscription_id);
    }

    async fn persist_ready(
        &self,
        worker_id: &str,
        ready: &BTreeSet<ExpertKey>,
    ) -> Result<(), Status> {
        let model_name = &ek_base::config::get_ek_settings().inference.model_name;
        let loaded: Vec<String> = ready
            .iter()
            .map(|key| {
                StoredExpertKey::new(
                    model_name.clone(),
                    key.layer_id as usize,
                    key.expert_id as usize,
                )
                .as_object_key()
            })
            .collect();
        StateWriterImpl::new()
            .update_expert_load_states(worker_id, &loaded)
            .await
            .map_err(internal_status)?;
        Ok(())
    }
}

async fn load_current_targets(
    registration: &RegisterWorkerRequest,
) -> Result<Vec<TargetExpert>, Status> {
    let reader = StateReaderImpl::new();
    let node = reader
        .node_by_hostname(&registration.worker_id)
        .await
        .map_err(internal_status)?
        .ok_or_else(|| Status::failed_precondition("registered worker is missing from database"))?;
    let assignments = reader
        .experts_by_node(node.id)
        .await
        .map_err(internal_status)?
        .into_iter()
        .filter(|expert| {
            expert.state.get("status").and_then(|state| state.as_str()) != Some("scheduled")
        })
        .collect();
    resolve_targets(registration, assignments, &reader, node.id).await
}

async fn resolve_targets(
    registration: &RegisterWorkerRequest,
    assignments: Vec<Expert>,
    reader: &StateReaderImpl,
    own_node_id: i32,
) -> Result<Vec<TargetExpert>, Status> {
    let model_name = &ek_base::config::get_ek_settings().inference.model_name;
    let instance_id = i32::try_from(registration.instance_id)
        .map_err(|_| Status::invalid_argument("instance_id does not fit the database schema"))?;
    let active_nodes: HashMap<i32, Node> = reader
        .active_nodes()
        .await
        .map_err(internal_status)?
        .into_iter()
        .map(|node| (node.id, node))
        .collect();
    let observed_experts = reader
        .experts_by_instance(instance_id)
        .await
        .map_err(internal_status)?;
    build_targets(
        registration,
        model_name,
        own_node_id,
        assignments,
        &active_nodes,
        observed_experts,
    )
    .map_err(|error| Status::internal(error.0))
}

#[derive(Debug)]
struct TargetBuildError(String);

fn build_targets(
    registration: &RegisterWorkerRequest,
    model_name: &str,
    own_node_id: i32,
    assignments: Vec<Expert>,
    active_nodes: &HashMap<i32, Node>,
    observed_experts: Vec<Expert>,
) -> Result<Vec<TargetExpert>, TargetBuildError> {
    let mut peers: HashMap<String, Vec<String>> = HashMap::new();
    for expert in observed_experts {
        if !is_loaded(&expert) || expert.node_id == own_node_id {
            continue;
        }
        let Some(node) = active_nodes.get(&expert.node_id) else {
            continue;
        };
        let Some(endpoint) = node.config.get("wm_addr").and_then(|value| value.as_str()) else {
            continue;
        };
        if !endpoint.is_empty() {
            peers
                .entry(expert.expert_id)
                .or_default()
                .push(endpoint.to_owned());
        }
    }

    let target_device = registration
        .device
        .as_ref()
        .expect("validated registration has one device")
        .device
        .clone();
    let mut targets = Vec::with_capacity(assignments.len());
    for assignment in assignments {
        let key = StoredExpertKey::from_expert_id(model_name, &assignment.expert_id)
            .map_err(|error| TargetBuildError(error.to_string()))?;
        let layer_id = u32::try_from(key.layer())
            .map_err(|_| TargetBuildError("expert layer does not fit the v2 protocol".into()))?;
        let expert_id = u32::try_from(key.idx())
            .map_err(|_| TargetBuildError("expert index does not fit the v2 protocol".into()))?;
        let mut peer_weight_endpoints = peers.remove(&assignment.expert_id).unwrap_or_default();
        peer_weight_endpoints.sort();
        peer_weight_endpoints.dedup();
        targets.push(TargetExpert {
            layer_id,
            expert_id,
            target_device: target_device.clone(),
            peer_weight_endpoints,
        });
    }
    targets.sort_by_key(|target| (target.layer_id, target.expert_id));
    targets.dedup_by_key(|target| (target.layer_id, target.expert_id));
    Ok(targets)
}

fn is_loaded(expert: &Expert) -> bool {
    expert.state.get("status").and_then(|state| state.as_str()) == Some("loaded")
}

fn internal_status(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::ek::{control::v2::WorkerDevice, worker::v2::ActivationDType};

    fn registration() -> RegisterWorkerRequest {
        RegisterWorkerRequest {
            worker_id: "worker-0".to_owned(),
            start_id: "start-0".to_owned(),
            instance_id: 7,
            computation_endpoint: "127.0.0.1:50051".to_owned(),
            peer_weight_endpoint: "http://127.0.0.1:50052".to_owned(),
            backend: "torch".to_owned(),
            activation_dtype: ActivationDType::ActivationDtypeBf16 as i32,
            device: Some(WorkerDevice {
                device: "cuda:0".to_owned(),
                max_experts: 8,
            }),
            max_batch_tokens: 64,
            max_active_batches_per_device: 1,
            max_pending_batches_per_device: 1,
            transport_type: crate::proto::ek::control::v2::WorkerTransportType::WorkerTransportGrpc
                as i32,
        }
    }

    fn expert(node_id: i32, status: &str) -> Expert {
        Expert {
            id: node_id,
            instance_id: 7,
            node_id,
            expert_id: "model/l1-e2".to_owned(),
            replica: 0,
            state: serde_json::json!({"status": status}),
        }
    }

    #[test]
    fn target_builder_includes_only_ready_active_peers() {
        let active_nodes = HashMap::from([
            (
                1,
                Node {
                    id: 1,
                    hostname: "worker-0".to_owned(),
                    device: "cuda:0".to_owned(),
                    config: serde_json::json!({"wm_addr": "http://worker-0:8000"}),
                },
            ),
            (
                2,
                Node {
                    id: 2,
                    hostname: "worker-1".to_owned(),
                    device: "cuda:1".to_owned(),
                    config: serde_json::json!({"wm_addr": "http://worker-1:8000"}),
                },
            ),
        ]);
        let targets = build_targets(
            &registration(),
            "model",
            1,
            vec![expert(1, "pending")],
            &active_nodes,
            vec![
                expert(1, "loaded"),
                expert(2, "loaded"),
                expert(3, "loaded"),
            ],
        )
        .unwrap();

        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].layer_id, 1);
        assert_eq!(targets[0].expert_id, 2);
        assert_eq!(targets[0].target_device, "cuda:0");
        assert_eq!(targets[0].peer_weight_endpoints, ["http://worker-1:8000"]);
    }
}
