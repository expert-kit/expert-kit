//! Worker registration, heartbeat, and availability lifecycle service.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;

use tokio::sync::{Mutex, watch};
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, Streaming};

use crate::{
    controller::{
        elastic::{progressive, recovery::recover_unique_experts},
        poller::request_immediate_poll,
        runtime_state::{ControllerRuntimeState, HeartbeatResult, RegistrationResult},
        service::instance::DefaultInstanceResolver,
    },
    proto::ek::control::v2::{
        HeartbeatRequest, HeartbeatSummary, RegisterWorkerRequest, RegisterWorkerResponse,
        worker_lifecycle_service_server::WorkerLifecycleService,
    },
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewNode,
        writer::StateWriterImpl,
    },
};

use super::status::state_status;

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
        request_immediate_poll();
        log::info!("Worker left: id={worker_id} start_id={start_id} graceful={graceful}");

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
                delete_stale_experts(&worker_id_owned).await;
            });
        } else {
            tokio::spawn(async move {
                recover_unique_experts(&worker_id_owned).await;
                delete_stale_experts(&worker_id_owned).await;
            });
        }
        Ok(())
    }
}

async fn delete_stale_experts(worker_id: &str) {
    match StateWriterImpl::new()
        .delete_experts_by_node_hostname(worker_id)
        .await
    {
        Ok(count) => {
            log::info!("Cleaned {count} stale expert rows for node {worker_id}");
        }
        Err(error) => {
            log::error!("Failed to clean stale expert rows for node {worker_id}: {error}");
        }
    }
}

#[derive(Clone)]
pub struct WorkerLifecycleServiceImpl {
    state: ControllerRuntimeState,
    hooks: Arc<dyn WorkerLifecycleHooks>,
    instance_resolver: Arc<dyn DefaultInstanceResolver>,
    heartbeat_timeout: Duration,
}

struct HeartbeatLeaseGuard {
    state: ControllerRuntimeState,
    hooks: Arc<dyn WorkerLifecycleHooks>,
    worker_id: String,
    start_id: String,
    lease: u64,
    graceful: bool,
    state_closed: bool,
    hook_completed: bool,
}

impl HeartbeatLeaseGuard {
    fn new(
        state: ControllerRuntimeState,
        hooks: Arc<dyn WorkerLifecycleHooks>,
        worker_id: String,
        start_id: String,
        lease: u64,
    ) -> Self {
        Self {
            state,
            hooks,
            worker_id,
            start_id,
            lease,
            graceful: false,
            state_closed: false,
            hook_completed: false,
        }
    }

    async fn close(&mut self) -> Result<(), Status> {
        if !self.state_closed {
            let unavailable = self
                .state
                .close_heartbeat(&self.worker_id, &self.start_id, self.lease)
                .await
                .map_err(state_status)?;
            self.state_closed = true;
            if !unavailable {
                self.hook_completed = true;
            }
        }
        if !self.hook_completed {
            self.hooks
                .unavailable(&self.worker_id, &self.start_id, self.graceful)
                .await?;
            self.hook_completed = true;
        }
        Ok(())
    }
}

impl Drop for HeartbeatLeaseGuard {
    fn drop(&mut self) {
        if self.hook_completed {
            return;
        }
        let state = self.state.clone();
        let hooks = self.hooks.clone();
        let worker_id = self.worker_id.clone();
        let start_id = self.start_id.clone();
        let lease = self.lease;
        let graceful = self.graceful;
        let state_closed = self.state_closed;
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let unavailable = if state_closed {
                    true
                } else {
                    match state.close_heartbeat(&worker_id, &start_id, lease).await {
                        Ok(unavailable) => unavailable,
                        Err(error) => {
                            tracing::error!(
                                worker_id,
                                start_id,
                                error = %error,
                                "failed to close a cancelled Worker heartbeat lease"
                            );
                            false
                        }
                    }
                };
                if unavailable
                    && let Err(error) = hooks.unavailable(&worker_id, &start_id, graceful).await
                {
                    tracing::error!(
                        worker_id,
                        start_id,
                        graceful,
                        error = %error,
                        "failed to apply cancelled Worker heartbeat cleanup"
                    );
                }
            });
        }
    }
}

impl WorkerLifecycleServiceImpl {
    pub fn new(
        state: ControllerRuntimeState,
        hooks: Arc<dyn WorkerLifecycleHooks>,
        instance_resolver: Arc<dyn DefaultInstanceResolver>,
        heartbeat_timeout: Duration,
    ) -> Self {
        assert!(
            !heartbeat_timeout.is_zero(),
            "heartbeat timeout must be positive"
        );
        Self {
            state,
            hooks,
            instance_resolver,
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
        let mut lease_guard = HeartbeatLeaseGuard::new(
            self.state.clone(),
            self.hooks.clone(),
            worker_id.clone(),
            start_id.clone(),
            lease,
        );
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
                lease_guard.graceful = true;
                if let Err(status) = self.hooks.shutting_down(&worker_id, &start_id).await {
                    break Err(status);
                }
                log::info!(
                    "Worker requested graceful shutdown: id={worker_id} start_id={start_id}"
                );
            }
        };

        lease_guard.close().await?;
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
        if registration.instance_id == 0 {
            return Err(Status::invalid_argument("instance_id must be positive"));
        }
        self.instance_resolver
            .resolve(registration.instance_id)
            .await?;
        let result = self
            .state
            .register(registration.clone())
            .await
            .map_err(state_status)?;
        self.hooks.registered(&registration, &result).await?;
        let device = registration
            .device
            .as_ref()
            .expect("validated Worker registration has one device");
        log::info!(
            "Worker joined: id={} start_id={} backend={} device={} max_experts={} \
             placement_generation={} topology_version={} replaced={}",
            registration.worker_id,
            registration.start_id,
            registration.backend,
            device.device,
            device.max_experts,
            result.response.current_placement_generation,
            result.response.current_topology_version,
            result.replaced_start_id.is_some(),
        );
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

fn internal_status(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    use super::*;
    use crate::controller::{
        runtime_state::ControllerStateError, service::instance::ResolvedDefaultInstance,
    };
    use crate::proto::ek::{
        control::v2::{WorkerDevice, WorkerRunState},
        worker::v2::ActivationDType,
    };

    #[derive(Default)]
    struct FakeHooks {
        events: Mutex<Vec<String>>,
    }

    struct FakeInstanceResolver;

    #[async_trait]
    impl DefaultInstanceResolver for FakeInstanceResolver {
        async fn resolve(
            &self,
            requested_instance_id: u64,
        ) -> Result<ResolvedDefaultInstance, Status> {
            if requested_instance_id != 0 && requested_instance_id != 7 {
                return Err(Status::failed_precondition(
                    "requested instance does not match Controller default",
                ));
            }
            Ok(ResolvedDefaultInstance {
                instance_id: 7,
                model_name: "model".to_owned(),
                instance_name: "default".to_owned(),
            })
        }
    }

    fn instance_resolver() -> Arc<dyn DefaultInstanceResolver> {
        Arc::new(FakeInstanceResolver)
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
            transport_type: crate::proto::ek::control::v2::WorkerTransportType::WorkerTransportGrpc
                as i32,
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
        let state = ControllerRuntimeState::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service = WorkerLifecycleServiceImpl::new(
            state,
            hooks.clone(),
            instance_resolver(),
            Duration::from_millis(100),
        );
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
        let state = ControllerRuntimeState::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service = WorkerLifecycleServiceImpl::new(
            state.clone(),
            hooks.clone(),
            instance_resolver(),
            Duration::from_millis(10),
        );
        state.register(registration()).await.unwrap();
        let messages = tokio_stream::pending::<Result<HeartbeatRequest, Status>>();
        let error = service.run_heartbeat(messages).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
        assert!(hooks.events.lock().await.is_empty());
    }

    #[tokio::test]
    async fn cancelled_heartbeat_handler_still_marks_worker_unavailable() {
        let state = ControllerRuntimeState::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service = WorkerLifecycleServiceImpl::new(
            state.clone(),
            hooks.clone(),
            instance_resolver(),
            Duration::from_secs(1),
        );
        state.register(registration()).await.unwrap();
        let (sender, receiver) = mpsc::channel(2);
        sender
            .send(Ok(heartbeat(1, WorkerRunState::WorkerRunning)))
            .await
            .unwrap();
        let task =
            tokio::spawn(async move { service.run_heartbeat(ReceiverStream::new(receiver)).await });
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if hooks
                    .events
                    .lock()
                    .await
                    .iter()
                    .any(|event| event == "heartbeat:worker-0")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if hooks
                    .events
                    .lock()
                    .await
                    .iter()
                    .any(|event| event == "unavailable:worker-0:false")
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        drop(sender);
    }

    #[tokio::test]
    async fn lifecycle_service_closes_a_live_worker_after_heartbeat_timeout() {
        let state = ControllerRuntimeState::new(8);
        let hooks = Arc::new(FakeHooks::default());
        let service = WorkerLifecycleServiceImpl::new(
            state.clone(),
            hooks.clone(),
            instance_resolver(),
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
    async fn lifecycle_rejects_a_nondefault_instance_before_state_mutation() {
        let state = ControllerRuntimeState::new(8);
        let service = WorkerLifecycleServiceImpl::new(
            state.clone(),
            Arc::new(FakeHooks::default()),
            instance_resolver(),
            Duration::from_millis(100),
        );
        let mut request = registration();
        request.instance_id = 8;

        let error = service
            .register_worker(Request::new(request))
            .await
            .unwrap_err();

        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
        assert!(matches!(
            state.registration("worker-0", "start-0").await,
            Err(ControllerStateError::UnknownWorker)
        ));
    }
}
