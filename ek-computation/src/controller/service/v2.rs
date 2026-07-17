//! v2 Worker lifecycle and Frontend topology gRPC services.

use std::{collections::HashMap, pin::Pin, sync::Arc, time::Duration};

use async_trait::async_trait;

use tokio::sync::{Mutex, mpsc, watch};
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

use crate::{
    controller::{
        elastic::{progressive, recovery::recover_unique_experts},
        poller::request_immediate_poll,
        registry::get_registry,
        routing_broadcaster::get_broadcaster,
        v2_state::{ControllerStateError, ControllerV2State, HeartbeatResult, RegistrationResult},
    },
    proto::ek::control::v2::{
        HeartbeatRequest, HeartbeatSummary, RegisterWorkerRequest, RegisterWorkerResponse,
        TopologyMessage, WatchTopologyRequest, topology_service_server::TopologyService,
        worker_lifecycle_service_server::WorkerLifecycleService,
    },
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewNode,
        writer::StateWriterImpl,
    },
};

const TOPOLOGY_STREAM_BUFFER: usize = 16;

#[async_trait]
pub trait WorkerLifecycleHooks: Send + Sync + 'static {
    async fn registered(
        &self,
        registration: &RegisterWorkerRequest,
        result: &RegistrationResult,
    ) -> Result<(), Status>;

    async fn heartbeat(&self, worker_id: &str, start_id: &str) -> Result<(), Status>;

    async fn shutting_down(&self, worker_id: &str, start_id: &str) -> Result<(), Status>;

    async fn unavailable(
        &self,
        worker_id: &str,
        start_id: &str,
        graceful: bool,
    ) -> Result<(), Status>;
}

pub struct DatabaseLifecycleHooks {
    graceful_recovery: Mutex<HashMap<(String, String), watch::Receiver<bool>>>,
}

impl DatabaseLifecycleHooks {
    pub fn new() -> Self {
        Self {
            graceful_recovery: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for DatabaseLifecycleHooks {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl WorkerLifecycleHooks for DatabaseLifecycleHooks {
    async fn registered(
        &self,
        registration: &RegisterWorkerRequest,
        result: &RegistrationResult,
    ) -> Result<(), Status> {
        let reader = StateReaderImpl::new();
        let writer = StateWriterImpl::new();
        let existing = reader
            .node_by_hostname(&registration.worker_id)
            .await
            .map_err(internal_status)?;
        let stored_start_id = existing
            .as_ref()
            .and_then(|node| node.config.get("start_id"))
            .and_then(|value| value.as_str());
        let new_start = stored_start_id != Some(registration.start_id.as_str());
        let mut config = existing
            .as_ref()
            .map(|node| node.config.clone())
            .unwrap_or_else(|| serde_json::json!({}));
        let device = registration
            .device
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("registration has no device"))?;

        config["addr"] = serde_json::json!(registration.computation_endpoint);
        config["channel"] = serde_json::json!("grpc");
        config["wm_addr"] = serde_json::json!(registration.peer_weight_endpoint);
        config["start_id"] = serde_json::json!(registration.start_id);
        config["instance_id"] = serde_json::json!(registration.instance_id);
        config["backend"] = serde_json::json!(registration.backend);
        config["activation_dtype"] = serde_json::json!(registration.activation_dtype);
        config["max_experts"] = serde_json::json!(device.max_experts);
        config["max_batch_tokens"] = serde_json::json!(registration.max_batch_tokens);
        config["max_active_batches"] =
            serde_json::json!(registration.max_active_batches_per_device);
        config["max_pending_batches"] =
            serde_json::json!(registration.max_pending_batches_per_device);

        writer
            .node_upsert(NewNode {
                hostname: registration.worker_id.clone(),
                device: device.device.clone(),
                config,
            })
            .await
            .map_err(internal_status)?;
        writer
            .node_update_seen(&registration.worker_id)
            .await
            .map_err(internal_status)?;
        get_registry()
            .lock()
            .await
            .reregister(&registration.worker_id);

        if new_start || result.replaced_start_id.is_some() {
            let worker_id = registration.worker_id.clone();
            tokio::spawn(async move {
                progressive::progressive_assign(&worker_id).await;
            });
        }
        request_immediate_poll();
        Ok(())
    }

    async fn heartbeat(&self, worker_id: &str, _start_id: &str) -> Result<(), Status> {
        StateWriterImpl::new()
            .node_update_seen(worker_id)
            .await
            .map_err(internal_status)
    }

    async fn shutting_down(&self, worker_id: &str, start_id: &str) -> Result<(), Status> {
        let key = (worker_id.to_owned(), start_id.to_owned());
        let mut recoveries = self.graceful_recovery.lock().await;
        if recoveries.contains_key(&key) {
            return Ok(());
        }
        let (completed, receiver) = watch::channel(false);
        recoveries.insert(key, receiver);
        let worker_id = worker_id.to_owned();
        tokio::spawn(async move {
            recover_unique_experts(&worker_id).await;
            completed.send_replace(true);
        });
        Ok(())
    }

    async fn unavailable(
        &self,
        worker_id: &str,
        start_id: &str,
        graceful: bool,
    ) -> Result<(), Status> {
        StateWriterImpl::new()
            .deactivate_node(worker_id)
            .await
            .map_err(internal_status)?;
        get_registry().lock().await.deregister(worker_id).await;
        get_broadcaster().remove_node(worker_id).await;
        request_immediate_poll();

        let worker_id_owned = worker_id.to_owned();
        if graceful {
            let recovery = self
                .graceful_recovery
                .lock()
                .await
                .remove(&(worker_id.to_owned(), start_id.to_owned()));
            tokio::spawn(async move {
                if let Some(mut completed) = recovery {
                    while !*completed.borrow_and_update() && completed.changed().await.is_ok() {}
                } else {
                    recover_unique_experts(&worker_id_owned).await;
                }
                let _ = StateWriterImpl::new()
                    .delete_experts_by_node_hostname(&worker_id_owned)
                    .await;
            });
        } else {
            tokio::spawn(async move {
                recover_unique_experts(&worker_id_owned).await;
                let _ = StateWriterImpl::new()
                    .delete_experts_by_node_hostname(&worker_id_owned)
                    .await;
            });
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct WorkerLifecycleServiceImpl {
    state: ControllerV2State,
    hooks: Arc<dyn WorkerLifecycleHooks>,
    heartbeat_timeout: Duration,
}

impl WorkerLifecycleServiceImpl {
    pub fn new(
        state: ControllerV2State,
        hooks: Arc<dyn WorkerLifecycleHooks>,
        heartbeat_timeout: Duration,
    ) -> Self {
        assert!(
            !heartbeat_timeout.is_zero(),
            "heartbeat timeout must be positive"
        );
        Self {
            state,
            hooks,
            heartbeat_timeout,
        }
    }

    async fn run_heartbeat<S>(&self, mut messages: S) -> Result<HeartbeatSummary, Status>
    where
        S: Stream<Item = Result<HeartbeatRequest, Status>> + Unpin,
    {
        let first = tokio::time::timeout(self.heartbeat_timeout, messages.next())
            .await
            .map_err(|_| Status::deadline_exceeded("first heartbeat timed out"))?
            .ok_or_else(|| Status::invalid_argument("heartbeat stream is empty"))??;
        let worker_id = first.worker_id.clone();
        let start_id = first.start_id.clone();
        let lease = self
            .state
            .open_heartbeat(&worker_id, &start_id)
            .await
            .map_err(state_status)?;
        let mut last_sequence = 0;
        let mut graceful = false;
        let mut next = Some(first);
        let stream_result = loop {
            let message = if let Some(message) = next.take() {
                message
            } else {
                match tokio::time::timeout(self.heartbeat_timeout, messages.next()).await {
                    Ok(Some(Ok(message))) => message,
                    Ok(Some(Err(status))) => break Err(status),
                    Ok(None) => break Ok(()),
                    Err(_) => break Err(Status::deadline_exceeded("heartbeat timed out")),
                }
            };
            if message.worker_id != worker_id || message.start_id != start_id {
                break Err(Status::invalid_argument(
                    "heartbeat identity changed within one stream",
                ));
            }
            match self
                .state
                .heartbeat(
                    &worker_id,
                    &start_id,
                    lease,
                    message.sequence,
                    message.state,
                )
                .await
            {
                Ok(HeartbeatResult::Applied) => {
                    if let Err(status) = self.hooks.heartbeat(&worker_id, &start_id).await {
                        break Err(status);
                    }
                    last_sequence = message.sequence;
                }
                Ok(HeartbeatResult::Duplicate) => {
                    last_sequence = message.sequence;
                }
                Err(error) => break Err(state_status(error)),
            }
            if message.state
                == crate::proto::ek::control::v2::WorkerRunState::WorkerShuttingDown as i32
                && !graceful
            {
                graceful = true;
                if let Err(status) = self.hooks.shutting_down(&worker_id, &start_id).await {
                    break Err(status);
                }
            }
        };

        if self
            .state
            .close_heartbeat(&worker_id, &start_id, lease)
            .await
            .map_err(state_status)?
        {
            self.hooks
                .unavailable(&worker_id, &start_id, graceful)
                .await?;
        }
        stream_result?;
        Ok(HeartbeatSummary { last_sequence })
    }
}

#[tonic::async_trait]
impl WorkerLifecycleService for WorkerLifecycleServiceImpl {
    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
        let registration = request.into_inner();
        let result = self
            .state
            .register(registration.clone())
            .await
            .map_err(state_status)?;
        self.hooks.registered(&registration, &result).await?;
        Ok(Response::new(result.response))
    }

    async fn heartbeat(
        &self,
        request: Request<Streaming<HeartbeatRequest>>,
    ) -> Result<Response<HeartbeatSummary>, Status> {
        self.run_heartbeat(request.into_inner())
            .await
            .map(Response::new)
    }
}

#[derive(Clone)]
pub struct TopologyServiceImpl {
    state: ControllerV2State,
}

impl TopologyServiceImpl {
    pub fn new(state: ControllerV2State) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl TopologyService for TopologyServiceImpl {
    type WatchTopologyStream =
        Pin<Box<dyn Stream<Item = Result<TopologyMessage, Status>> + Send + 'static>>;

    async fn watch_topology(
        &self,
        request: Request<WatchTopologyRequest>,
    ) -> Result<Response<Self::WatchTopologyStream>, Status> {
        let request = request.into_inner();
        if request.instance_id == 0 {
            return Err(Status::invalid_argument("instance_id must be positive"));
        }
        let newest = self.state.topology_version(request.instance_id).await;
        if request.current_version > newest {
            return Err(Status::invalid_argument(format!(
                "requested topology version {} is newer than current version {newest}",
                request.current_version
            )));
        }
        let state = self.state.clone();
        let mut changed = state.subscribe();
        let (sender, receiver) = mpsc::channel(TOPOLOGY_STREAM_BUFFER);
        tokio::spawn(async move {
            let mut installed_version = request.current_version;
            let mut sent_initial = false;
            loop {
                let newest = state.topology_version(request.instance_id).await;
                if !sent_initial || installed_version < newest {
                    let messages = match state
                        .topology_messages(request.instance_id, installed_version)
                        .await
                    {
                        Ok(messages) => messages,
                        Err(error) => {
                            let _ = sender.send(Err(state_status(error))).await;
                            return;
                        }
                    };
                    for message in messages {
                        if sender.send(Ok(message)).await.is_err() {
                            return;
                        }
                    }
                    installed_version = newest;
                    sent_initial = true;
                    continue;
                }
                if changed.changed().await.is_err() {
                    return;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}

pub(super) fn state_status(error: ControllerStateError) -> Status {
    match error {
        ControllerStateError::InvalidRegistration(_)
        | ControllerStateError::InvalidHeartbeatState
        | ControllerStateError::InvalidExpertState
        | ControllerStateError::FutureTopologyVersion { .. } => {
            Status::invalid_argument(error.to_string())
        }
        ControllerStateError::UnknownWorker
        | ControllerStateError::ReplacedWorker
        | ControllerStateError::StaleHeartbeatStream
        | ControllerStateError::HeartbeatSequenceRollback { .. }
        | ControllerStateError::PlacementTooLarge { .. }
        | ControllerStateError::PlacementGenerationMismatch { .. }
        | ControllerStateError::ReportSequenceRollback { .. }
        | ControllerStateError::ConflictingReportSequence(_)
        | ControllerStateError::UnknownDrain(_) => Status::failed_precondition(error.to_string()),
        ControllerStateError::Persistence(_) => Status::internal(error.to_string()),
    }
}

fn internal_status(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::ek::{
        control::v2::{WorkerDevice, WorkerRunState, topology_message},
        worker::v2::ActivationDType,
    };

    #[derive(Default)]
    struct FakeHooks {
        events: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl WorkerLifecycleHooks for FakeHooks {
        async fn registered(
            &self,
            registration: &RegisterWorkerRequest,
            _result: &RegistrationResult,
        ) -> Result<(), Status> {
            self.events
                .lock()
                .await
                .push(format!("registered:{}", registration.worker_id));
            Ok(())
        }

        async fn heartbeat(&self, worker_id: &str, _start_id: &str) -> Result<(), Status> {
            self.events
                .lock()
                .await
                .push(format!("heartbeat:{worker_id}"));
            Ok(())
        }

        async fn shutting_down(&self, worker_id: &str, _start_id: &str) -> Result<(), Status> {
            self.events
                .lock()
                .await
                .push(format!("shutdown:{worker_id}"));
            Ok(())
        }

        async fn unavailable(
            &self,
            worker_id: &str,
            _start_id: &str,
            graceful: bool,
        ) -> Result<(), Status> {
            self.events
                .lock()
                .await
                .push(format!("unavailable:{worker_id}:{graceful}"));
            Ok(())
        }
    }

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
                max_experts: 16,
            }),
            max_batch_tokens: 64,
            max_active_batches_per_device: 1,
            max_pending_batches_per_device: 1,
        }
    }

    fn heartbeat(sequence: u64, state: WorkerRunState) -> HeartbeatRequest {
        HeartbeatRequest {
            worker_id: "worker-0".to_owned(),
            start_id: "start-0".to_owned(),
            sequence,
            state: state as i32,
        }
    }

    #[tokio::test]
    async fn lifecycle_service_registers_and_closes_a_graceful_stream() {
        let state = ControllerV2State::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service =
            WorkerLifecycleServiceImpl::new(state, hooks.clone(), Duration::from_millis(100));
        let response = service
            .register_worker(Request::new(registration()))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.current_topology_version, 0);

        let messages = tokio_stream::iter(vec![
            Ok(heartbeat(1, WorkerRunState::WorkerRunning)),
            Ok(heartbeat(2, WorkerRunState::WorkerShuttingDown)),
        ]);
        let summary = service.run_heartbeat(messages).await.unwrap();
        assert_eq!(summary.last_sequence, 2);
        assert_eq!(
            *hooks.events.lock().await,
            [
                "registered:worker-0",
                "heartbeat:worker-0",
                "heartbeat:worker-0",
                "shutdown:worker-0",
                "unavailable:worker-0:true",
            ]
        );
    }

    #[tokio::test]
    async fn lifecycle_service_times_out_and_marks_worker_unavailable() {
        let state = ControllerV2State::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service = WorkerLifecycleServiceImpl::new(
            state.clone(),
            hooks.clone(),
            Duration::from_millis(10),
        );
        state.register(registration()).await.unwrap();
        let messages = tokio_stream::pending::<Result<HeartbeatRequest, Status>>();
        let error = service.run_heartbeat(messages).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
        assert!(hooks.events.lock().await.is_empty());
    }

    #[tokio::test]
    async fn lifecycle_service_closes_a_live_worker_after_heartbeat_timeout() {
        let state = ControllerV2State::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service = WorkerLifecycleServiceImpl::new(
            state.clone(),
            hooks.clone(),
            Duration::from_millis(10),
        );
        state.register(registration()).await.unwrap();
        let (sender, receiver) = mpsc::channel(1);
        sender
            .send(heartbeat(1, WorkerRunState::WorkerRunning))
            .await
            .unwrap();
        let messages = ReceiverStream::new(receiver).map(Ok);
        let error = service.run_heartbeat(messages).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
        assert_eq!(
            *hooks.events.lock().await,
            ["heartbeat:worker-0", "unavailable:worker-0:false"]
        );
        drop(sender);
    }

    #[tokio::test]
    async fn topology_service_sends_an_initial_empty_snapshot() {
        let state = ControllerV2State::new(8);
        let service = TopologyServiceImpl::new(state);
        let response = service
            .watch_topology(Request::new(WatchTopologyRequest {
                instance_id: 7,
                current_version: 0,
            }))
            .await
            .unwrap();
        let mut stream = response.into_inner();
        let message = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let Some(topology_message::Message::Snapshot(snapshot)) = message.message else {
            panic!("expected a snapshot");
        };
        assert_eq!(snapshot.topology_version, 0);
        assert_eq!(snapshot.part_count, 1);
        assert!(snapshot.routes.is_empty());
    }
}
