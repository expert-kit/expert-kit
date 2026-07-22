//! Cross-language recovery tests for the Rust Controller and Python Workers.

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use ek_computation::{
    controller::{
        service::{
            instance::{DefaultInstanceResolver, InstanceServiceImpl, ResolvedDefaultInstance},
            v2::{TopologyServiceImpl, WorkerLifecycleHooks, WorkerLifecycleServiceImpl},
            v2_weight::{TargetSubscription, WeightControlHooks, WeightControlServiceImpl},
        },
        v2_state::{ControllerV2State, ExpertKey, RegistrationResult},
    },
    proto::ek::{
        control::v2::{
            RegisterWorkerRequest, TargetExpert, TopologyMessage, WatchTopologyRequest,
            WorkerRoute, instance_service_server::InstanceServiceServer, topology_message,
            topology_service_client::TopologyServiceClient,
            topology_service_server::TopologyServiceServer,
            weight_control_service_server::WeightControlServiceServer,
            worker_lifecycle_service_server::WorkerLifecycleServiceServer,
        },
        worker::v2::{
            ActivationDType, ComputeErrorCode, ExecuteRequest, ExecuteResponse,
            computation_service_client::ComputationServiceClient, execute_response,
        },
    },
};
use tokio::sync::{Mutex, Notify, mpsc, oneshot};
use tonic::{Request, Status, Streaming, transport::Channel};

const INSTANCE_ID: u64 = 7;
const WORKER_A: &str = "worker-a";
const WORKER_B: &str = "worker-b";
const TEST_TIMEOUT: Duration = Duration::from_secs(15);

struct TestInstanceResolver;

#[async_trait]
impl DefaultInstanceResolver for TestInstanceResolver {
    async fn resolve(&self, requested_instance_id: u64) -> Result<ResolvedDefaultInstance, Status> {
        if requested_instance_id != 0 && requested_instance_id != INSTANCE_ID {
            return Err(Status::failed_precondition("unexpected model instance"));
        }
        Ok(ResolvedDefaultInstance {
            instance_id: INSTANCE_ID,
            model_name: "test-model".to_owned(),
            instance_name: "default".to_owned(),
        })
    }
}

#[derive(Clone)]
struct Subscriber {
    id: u64,
    updates: mpsc::Sender<Result<Vec<TargetExpert>, Status>>,
}

#[derive(Default)]
struct HookState {
    next_subscription_id: u64,
    subscribers: HashMap<String, Subscriber>,
    replacement_requested: bool,
    shutdown_seen: bool,
    unavailable: Vec<(String, bool)>,
    live_workers: HashSet<String>,
    ready: HashMap<String, BTreeSet<ExpertKey>>,
}

#[derive(Default)]
struct RecoveryHooks {
    state: Mutex<HookState>,
    changed: Notify,
}

impl RecoveryHooks {
    fn target() -> TargetExpert {
        TargetExpert {
            layer_id: 0,
            expert_id: 0,
            target_device: "cpu".to_owned(),
            peer_weight_endpoints: Vec::new(),
        }
    }

    async fn request_replacement(&self) -> Result<(), Status> {
        let update = {
            let mut state = self.state.lock().await;
            if state.replacement_requested {
                return Ok(());
            }
            state.replacement_requested = true;
            state
                .subscribers
                .get(WORKER_B)
                .map(|subscriber| subscriber.updates.clone())
        };
        self.changed.notify_waiters();
        if let Some(update) = update {
            update
                .send(Ok(vec![Self::target()]))
                .await
                .map_err(|_| Status::unavailable("replacement weight stream closed"))?;
        }
        Ok(())
    }

    async fn wait_for_subscribers(&self) {
        tokio::time::timeout(TEST_TIMEOUT, async {
            loop {
                let notified = self.changed.notified();
                let ready = {
                    let state = self.state.lock().await;
                    state.subscribers.contains_key(WORKER_A)
                        && state.subscribers.contains_key(WORKER_B)
                };
                if ready {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("Workers did not open both weight streams");
    }

    async fn wait_for_unavailable(&self, worker_id: &str, graceful: bool) {
        tokio::time::timeout(TEST_TIMEOUT, async {
            loop {
                let notified = self.changed.notified();
                let observed = {
                    let state = self.state.lock().await;
                    state
                        .unavailable
                        .iter()
                        .any(|event| event == &(worker_id.to_owned(), graceful))
                };
                if observed {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("Controller did not observe the expected Worker loss");
    }

    async fn wait_until_routable(&self, worker_id: &str) {
        tokio::time::timeout(TEST_TIMEOUT, async {
            loop {
                let notified = self.changed.notified();
                let routable = {
                    let state = self.state.lock().await;
                    state.live_workers.contains(worker_id)
                        && state.ready.get(worker_id).is_some_and(|ready| {
                            ready.contains(&ExpertKey {
                                layer_id: 0,
                                expert_id: 0,
                            })
                        })
                };
                if routable {
                    return;
                }
                notified.await;
            }
        })
        .await
        .expect("Worker did not become live with its assigned expert ready");
    }

    async fn shutdown_seen(&self) -> bool {
        self.state.lock().await.shutdown_seen
    }
}

#[async_trait]
impl WorkerLifecycleHooks for RecoveryHooks {
    async fn registered(
        &self,
        _registration: &RegisterWorkerRequest,
        _result: &RegistrationResult,
    ) -> Result<(), Status> {
        Ok(())
    }

    async fn heartbeat(&self, worker_id: &str, _start_id: &str) -> Result<(), Status> {
        self.state
            .lock()
            .await
            .live_workers
            .insert(worker_id.to_owned());
        self.changed.notify_waiters();
        Ok(())
    }

    async fn shutting_down(&self, worker_id: &str, _start_id: &str) -> Result<(), Status> {
        if worker_id == WORKER_A {
            self.state.lock().await.shutdown_seen = true;
            self.changed.notify_waiters();
            self.request_replacement().await?;
        }
        Ok(())
    }

    async fn unavailable(
        &self,
        worker_id: &str,
        _start_id: &str,
        graceful: bool,
    ) -> Result<(), Status> {
        self.state
            .lock()
            .await
            .unavailable
            .push((worker_id.to_owned(), graceful));
        self.changed.notify_waiters();
        if worker_id == WORKER_A {
            self.request_replacement().await?;
        }
        Ok(())
    }
}

#[async_trait]
impl WeightControlHooks for RecoveryHooks {
    async fn subscribe(
        &self,
        registration: &RegisterWorkerRequest,
    ) -> Result<TargetSubscription, Status> {
        let (updates, receiver) = mpsc::channel(4);
        let (subscription_id, initial) = {
            let mut state = self.state.lock().await;
            state.next_subscription_id += 1;
            let subscription_id = state.next_subscription_id;
            let initial = if registration.worker_id == WORKER_A
                || (registration.worker_id == WORKER_B && state.replacement_requested)
            {
                vec![Self::target()]
            } else if registration.worker_id == WORKER_B {
                Vec::new()
            } else {
                return Err(Status::invalid_argument("unexpected test Worker ID"));
            };
            state.subscribers.insert(
                registration.worker_id.clone(),
                Subscriber {
                    id: subscription_id,
                    updates,
                },
            );
            (subscription_id, initial)
        };
        self.changed.notify_waiters();
        Ok(TargetSubscription::new(subscription_id, initial, receiver))
    }

    async fn unsubscribe(&self, worker_id: &str, subscription_id: u64) {
        let mut state = self.state.lock().await;
        if state
            .subscribers
            .get(worker_id)
            .is_some_and(|subscriber| subscriber.id == subscription_id)
        {
            state.subscribers.remove(worker_id);
        }
        drop(state);
        self.changed.notify_waiters();
    }

    async fn persist_ready(
        &self,
        worker_id: &str,
        ready: &BTreeSet<ExpertKey>,
    ) -> Result<(), Status> {
        self.state
            .lock()
            .await
            .ready
            .insert(worker_id.to_owned(), ready.clone());
        self.changed.notify_waiters();
        Ok(())
    }
}

struct ControllerHarness {
    endpoint: String,
    hooks: Arc<RecoveryHooks>,
    shutdown: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl ControllerHarness {
    async fn start() -> Self {
        let port = unused_port();
        let address = format!("127.0.0.1:{port}").parse().unwrap();
        let endpoint = format!("127.0.0.1:{port}");
        let state = ControllerV2State::new(32);
        let hooks = Arc::new(RecoveryHooks::default());
        let instance_resolver: Arc<dyn DefaultInstanceResolver> = Arc::new(TestInstanceResolver);
        let lifecycle = WorkerLifecycleServiceImpl::new(
            state.clone(),
            hooks.clone(),
            instance_resolver.clone(),
            Duration::from_millis(500),
        );
        let weights = WeightControlServiceImpl::new(state.clone(), hooks.clone());
        let topology = TopologyServiceImpl::new(state, instance_resolver.clone());
        let instance = InstanceServiceImpl::new(instance_resolver);
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(InstanceServiceServer::new(instance))
                .add_service(WorkerLifecycleServiceServer::new(lifecycle))
                .add_service(WeightControlServiceServer::new(weights))
                .add_service(TopologyServiceServer::new(topology))
                .serve_with_shutdown(address, async {
                    let _ = stopped.await;
                })
                .await
                .unwrap();
        });
        wait_for_controller(&endpoint).await;
        Self {
            endpoint,
            hooks,
            shutdown: Some(shutdown),
            task,
        }
    }

    async fn close(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        self.task.await.unwrap();
    }
}

struct TopologyObserver {
    stream: Streaming<TopologyMessage>,
    version: u64,
    routes: HashMap<(u32, u32), Vec<WorkerRoute>>,
}

impl TopologyObserver {
    async fn connect(endpoint: &str) -> Self {
        let mut client = TopologyServiceClient::connect(format!("http://{endpoint}"))
            .await
            .unwrap();
        let stream = client
            .watch_topology(WatchTopologyRequest {
                instance_id: INSTANCE_ID,
                current_version: 0,
            })
            .await
            .unwrap()
            .into_inner();
        Self {
            stream,
            version: 0,
            routes: HashMap::new(),
        }
    }

    async fn wait_for_only(&mut self, worker_id: &str) -> (u64, WorkerRoute) {
        tokio::time::timeout(TEST_TIMEOUT, async {
            loop {
                if let Some(routes) = self.routes.get(&(0, 0))
                    && routes.len() == 1
                    && routes[0].worker_id == worker_id
                {
                    return (self.version, routes[0].clone());
                }
                let message = self
                    .stream
                    .message()
                    .await
                    .unwrap()
                    .expect("Topology stream ended unexpectedly");
                self.apply(message);
            }
        })
        .await
        .expect("Topology did not publish the expected sole Worker")
    }

    fn apply(&mut self, message: TopologyMessage) {
        match message.message.expect("Topology message has no payload") {
            topology_message::Message::Snapshot(snapshot) => {
                assert_eq!(snapshot.part_count, 1);
                assert_eq!(snapshot.part_index, 0);
                self.routes.clear();
                for route in snapshot.routes {
                    self.routes
                        .insert((route.layer_id, route.expert_id), route.replicas);
                }
                self.version = snapshot.topology_version;
            }
            topology_message::Message::Update(update) => {
                assert_eq!(update.part_count, 1);
                assert_eq!(update.part_index, 0);
                assert_eq!(update.previous_version, self.version);
                for change in update.changes {
                    if change.replicas.is_empty() {
                        self.routes.remove(&(change.layer_id, change.expert_id));
                    } else {
                        self.routes
                            .insert((change.layer_id, change.expert_id), change.replicas);
                    }
                }
                self.version = update.topology_version;
            }
        }
    }
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new(name: &str) -> Self {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("expertkit-{name}-{}-{unique}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct WorkerProcess {
    child: Child,
}

impl WorkerProcess {
    fn spawn(
        worker_id: &str,
        controller: &str,
        computation_endpoint: &str,
        root: &Path,
        gate: Option<&Path>,
    ) -> Self {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let python = repository.join(".venv/bin/python");
        let fixture = repository.join("ek-worker/tests/fixtures/cpu_worker_process.py");
        assert!(
            python.is_file(),
            "run `uv sync --locked --package expertkit-worker` first"
        );
        let mut command = Command::new(python);
        command
            .current_dir(repository)
            .arg(fixture)
            .arg("--worker-id")
            .arg(worker_id)
            .arg("--controller")
            .arg(controller)
            .arg("--computation-listen")
            .arg(computation_endpoint)
            .arg("--cache-root")
            .arg(root.join(format!("{worker_id}-cache")))
            .arg("--active-marker")
            .arg(root.join(format!("{worker_id}-active")))
            .arg("--pending-marker")
            .arg(root.join(format!("{worker_id}-pending")))
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        if let Some(gate) = gate {
            command.arg("--gate").arg(gate);
        }
        Self {
            child: command.spawn().unwrap(),
        }
    }

    fn signal_term(&self) {
        let status = Command::new("kill")
            .arg("-TERM")
            .arg(self.child.id().to_string())
            .status()
            .unwrap();
        assert!(status.success());
    }

    fn kill(&mut self) -> ExitStatus {
        self.child.kill().unwrap();
        self.child.wait().unwrap()
    }

    async fn wait(&mut self) -> ExitStatus {
        tokio::time::timeout(TEST_TIMEOUT, async {
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    return status;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("Worker process did not exit")
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn unused_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn wait_for_controller(endpoint: &str) {
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if TopologyServiceClient::connect(format!("http://{endpoint}"))
                .await
                .is_ok()
            {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("Controller did not start");
}

async fn wait_for_marker(path: &Path, expected: &str) {
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if fs::read_to_string(path).ok().as_deref() == Some(expected) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("Worker did not reach the expected Backend submission");
}

async fn wait_for_path(path: &Path) {
    tokio::time::timeout(TEST_TIMEOUT, async {
        while !path.exists() {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("Worker did not retain the expected pending call");
}

fn execute_request(topology_version: u64) -> Request<ExecuteRequest> {
    let hidden_states = [1.0_f32, 2.0_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut request = Request::new(ExecuteRequest {
        instance_id: INSTANCE_ID,
        layer_id: 0,
        topology_version,
        token_count: 1,
        hidden_dim: 2,
        top_k: 1,
        dtype: ActivationDType::ActivationDtypeFp32 as i32,
        hidden_states,
        expert_ids: 0_i32.to_le_bytes().to_vec(),
        routing_weights: 0.5_f32.to_le_bytes().to_vec(),
    });
    request.set_timeout(Duration::from_secs(10));
    request
}

async fn computation_client(endpoint: &str) -> ComputationServiceClient<Channel> {
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            if let Ok(client) =
                ComputationServiceClient::connect(format!("http://{endpoint}")).await
            {
                return client;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("Worker computation endpoint did not start")
}

async fn execute(
    mut client: ComputationServiceClient<Channel>,
    topology_version: u64,
) -> Result<ExecuteResponse, tonic::Status> {
    client
        .execute(execute_request(topology_version))
        .await
        .map(tonic::Response::into_inner)
}

fn assert_output(response: ExecuteResponse) {
    let Some(execute_response::Result::PartialOutput(bytes)) = response.result else {
        panic!("expected a successful Worker response");
    };
    assert_eq!(bytes.len(), 8);
    let values = [
        f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    ];
    let expected = [0.3655293_f32, 1.761594_f32];
    for (actual, expected) in values.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

async fn wait_for_draining(client: ComputationServiceClient<Channel>, topology_version: u64) {
    tokio::time::timeout(TEST_TIMEOUT, async {
        loop {
            match execute(client.clone(), topology_version).await {
                Ok(ExecuteResponse {
                    result: Some(execute_response::Result::Error(error)),
                }) if error.code == ComputeErrorCode::ComputeErrorDraining as i32 => {
                    assert!(error.retryable);
                    assert_eq!(error.min_topology_version, Some(topology_version));
                    return;
                }
                Ok(ExecuteResponse {
                    result: Some(execute_response::Result::Error(error)),
                }) if error.code == ComputeErrorCode::ComputeErrorBusy as i32 => {}
                Err(status) if status.code() == tonic::Code::ResourceExhausted => {}
                other => panic!("unexpected response while waiting for drain: {other:?}"),
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("Worker did not apply the drain authorization");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the synchronized Python Worker environment and direct I/O"]
async fn graceful_shutdown_preserves_admitted_batches_and_cuts_over_topology() {
    let root = TestDirectory::new("graceful-worker-recovery");
    let controller = ControllerHarness::start().await;
    let mut topology = TopologyObserver::connect(&controller.endpoint).await;
    let endpoint_a = format!("127.0.0.1:{}", unused_port());
    let endpoint_b = format!("127.0.0.1:{}", unused_port());
    let gate = root.0.join("worker-a-gate");
    let active = root.0.join("worker-a-active");
    let pending = root.0.join("worker-a-pending");
    let mut worker_a = WorkerProcess::spawn(
        WORKER_A,
        &controller.endpoint,
        &endpoint_a,
        &root.0,
        Some(&gate),
    );
    let mut worker_b =
        WorkerProcess::spawn(WORKER_B, &controller.endpoint, &endpoint_b, &root.0, None);

    controller.hooks.wait_for_subscribers().await;
    controller.hooks.wait_until_routable(WORKER_A).await;
    let (version_a, route_a) = topology.wait_for_only(WORKER_A).await;
    assert_eq!(route_a.computation_endpoint, endpoint_a);
    let client_a = computation_client(&endpoint_a).await;
    let first = tokio::spawn(execute(client_a.clone(), version_a));
    wait_for_marker(&active, "1").await;
    let second = tokio::spawn(execute(client_a.clone(), version_a));
    wait_for_path(&pending).await;

    worker_a.signal_term();
    let (version_b, route_b) = topology.wait_for_only(WORKER_B).await;
    assert!(version_b > version_a);
    assert_eq!(route_b.computation_endpoint, endpoint_b);

    fs::write(&gate, b"run").unwrap();
    assert_output(first.await.unwrap().unwrap());
    wait_for_marker(&active, "2").await;
    wait_for_draining(client_a, version_b).await;
    fs::write(&gate, b"run").unwrap();
    assert_output(second.await.unwrap().unwrap());

    assert!(worker_a.wait().await.success());
    controller.hooks.wait_for_unavailable(WORKER_A, true).await;
    assert!(controller.hooks.shutdown_seen().await);
    let client_b = computation_client(&endpoint_b).await;
    assert_output(execute(client_b, version_b).await.unwrap());

    let _ = worker_b.kill();
    drop(topology);
    controller.close().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires the synchronized Python Worker environment and direct I/O"]
async fn abnormal_loss_reassigns_ready_weight_and_restores_computation() {
    let root = TestDirectory::new("abnormal-worker-recovery");
    let controller = ControllerHarness::start().await;
    let mut topology = TopologyObserver::connect(&controller.endpoint).await;
    let endpoint_a = format!("127.0.0.1:{}", unused_port());
    let endpoint_b = format!("127.0.0.1:{}", unused_port());
    let mut worker_a =
        WorkerProcess::spawn(WORKER_A, &controller.endpoint, &endpoint_a, &root.0, None);
    let mut worker_b =
        WorkerProcess::spawn(WORKER_B, &controller.endpoint, &endpoint_b, &root.0, None);

    controller.hooks.wait_for_subscribers().await;
    controller.hooks.wait_until_routable(WORKER_A).await;
    let (version_a, _) = topology.wait_for_only(WORKER_A).await;
    let client_a = computation_client(&endpoint_a).await;
    assert_output(execute(client_a, version_a).await.unwrap());

    assert!(!worker_a.kill().success());
    controller.hooks.wait_for_unavailable(WORKER_A, false).await;
    let (version_b, route_b) = topology.wait_for_only(WORKER_B).await;
    assert!(version_b > version_a);
    assert_eq!(route_b.computation_endpoint, endpoint_b);
    assert!(!controller.hooks.shutdown_seen().await);
    let client_b = computation_client(&endpoint_b).await;
    assert_output(execute(client_b, version_b).await.unwrap());

    let _ = worker_b.kill();
    drop(topology);
    controller.close().await;
}
