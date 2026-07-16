//! In-memory state shared by the v2 Controller services.
//!
//! The database remains the durable source for placement policy. This module owns
//! process-lifetime protocol state: Worker starts, heartbeat leases, placement and
//! report sequences, and the versioned routes consumed by Frontends.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, VecDeque},
    fmt,
    sync::Arc,
};

use tokio::sync::{RwLock, watch};

use crate::proto::ek::{
    control::v2::{
        DrainAuthorizationPart, ExpertRoute, ExpertState, ExpertStateKind, RegisterWorkerRequest,
        RegisterWorkerResponse, RouteChange, TargetExpert, TargetExpertListPart, TopologyMessage,
        TopologySnapshotPart, TopologyUpdatePart, WorkerRoute, WorkerRunState, topology_message,
    },
    worker::v2::{ActivationDType, ExpertKey as ProtoExpertKey},
};

pub const MAX_CONTROL_ENTRIES_PER_PART: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ExpertKey {
    pub layer_id: u32,
    pub expert_id: u32,
}

impl ExpertKey {
    fn from_state(state: &ExpertState) -> Self {
        Self {
            layer_id: state.layer_id,
            expert_id: state.expert_id,
        }
    }

    fn from_target(target: &TargetExpert) -> Self {
        Self {
            layer_id: target.layer_id,
            expert_id: target.expert_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegistrationResult {
    pub response: RegisterWorkerResponse,
    pub replaced_start_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub generation: u64,
    pub changed: bool,
    pub parts: Vec<TargetExpertListPart>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrainAuthorization {
    pub drain_id: u64,
    pub parts: Vec<DrainAuthorizationPart>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeartbeatResult {
    Applied,
    Duplicate,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateReportResult {
    Applied,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ControllerStateError {
    InvalidRegistration(&'static str),
    UnknownWorker,
    ReplacedWorker,
    StaleHeartbeatStream,
    InvalidHeartbeatState,
    HeartbeatSequenceRollback { last: u64, received: u64 },
    PlacementTooLarge { maximum: u32, received: usize },
    PlacementGenerationMismatch { expected: u64, received: u64 },
    ReportSequenceRollback { last: u64, received: u64 },
    ConflictingReportSequence(u64),
    InvalidExpertState,
    UnknownDrain(u64),
    FutureTopologyVersion { current: u64, requested: u64 },
}

impl fmt::Display for ControllerStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRegistration(message) => {
                write!(formatter, "invalid registration: {message}")
            }
            Self::UnknownWorker => formatter.write_str("worker is not registered"),
            Self::ReplacedWorker => formatter.write_str("worker start has been replaced"),
            Self::StaleHeartbeatStream => formatter.write_str("heartbeat stream has been replaced"),
            Self::InvalidHeartbeatState => formatter.write_str("heartbeat state is invalid"),
            Self::HeartbeatSequenceRollback { last, received } => write!(
                formatter,
                "heartbeat sequence rolled back from {last} to {received}"
            ),
            Self::PlacementTooLarge { maximum, received } => write!(
                formatter,
                "placement contains {received} experts but worker capacity is {maximum}"
            ),
            Self::PlacementGenerationMismatch { expected, received } => write!(
                formatter,
                "placement generation is {received}, expected {expected}"
            ),
            Self::ReportSequenceRollback { last, received } => write!(
                formatter,
                "state report sequence rolled back from {last} to {received}"
            ),
            Self::ConflictingReportSequence(sequence) => {
                write!(
                    formatter,
                    "state report sequence {sequence} was reused with new data"
                )
            }
            Self::InvalidExpertState => formatter.write_str("expert state is invalid"),
            Self::UnknownDrain(drain_id) => write!(formatter, "drain {drain_id} is unknown"),
            Self::FutureTopologyVersion { current, requested } => write!(
                formatter,
                "requested topology version {requested} is newer than current version {current}"
            ),
        }
    }
}

impl std::error::Error for ControllerStateError {}

#[derive(Clone)]
pub struct ControllerV2State {
    inner: Arc<RwLock<Inner>>,
    changed: watch::Sender<u64>,
}

struct Inner {
    workers: HashMap<String, WorkerRecord>,
    topologies: HashMap<u64, InstanceTopology>,
    next_heartbeat_lease: u64,
    next_drain_id: u64,
    notification_sequence: u64,
    history_limit: usize,
}

struct WorkerRecord {
    registration: RegisterWorkerRequest,
    live: bool,
    heartbeat_lease: u64,
    heartbeat_open: bool,
    last_heartbeat_sequence: Option<u64>,
    run_state: WorkerRunState,
    placement_generation: u64,
    targets: BTreeMap<ExpertKey, TargetExpert>,
    expert_states: BTreeMap<ExpertKey, ExpertState>,
    last_report: Option<StoredReport>,
    drains: VecDeque<DrainRecord>,
}

#[derive(Clone, PartialEq)]
struct StoredReport {
    sequence: u64,
    full: bool,
    states: Vec<ExpertState>,
}

struct DrainRecord {
    drain_id: u64,
    placement_generation: u64,
    min_topology_version: u64,
    stop_all: bool,
    experts: BTreeSet<ExpertKey>,
    completed: bool,
}

#[derive(Default)]
struct InstanceTopology {
    version: u64,
    routes: BTreeMap<ExpertKey, Vec<WorkerRoute>>,
    history: VecDeque<TopologyRevision>,
}

#[derive(Clone)]
struct TopologyRevision {
    previous_version: u64,
    topology_version: u64,
    changes: Vec<RouteChange>,
}

impl ControllerV2State {
    pub fn new(history_limit: usize) -> Self {
        assert!(history_limit > 0, "topology history limit must be positive");
        let (changed, _receiver) = watch::channel(0);
        Self {
            inner: Arc::new(RwLock::new(Inner {
                workers: HashMap::new(),
                topologies: HashMap::new(),
                next_heartbeat_lease: 0,
                next_drain_id: 0,
                notification_sequence: 0,
                history_limit,
            })),
            changed,
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    pub async fn register(
        &self,
        registration: RegisterWorkerRequest,
    ) -> Result<RegistrationResult, ControllerStateError> {
        validate_registration(&registration)?;

        let worker_id = registration.worker_id.clone();
        let start_id = registration.start_id.clone();
        let instance_id = registration.instance_id;
        let mut inner = self.inner.write().await;

        if let Some(existing) = inner.workers.get(&worker_id)
            && existing.registration.start_id == start_id
        {
            if existing.registration != registration {
                return Err(ControllerStateError::InvalidRegistration(
                    "static fields changed for an existing start_id",
                ));
            }
            let topology_version = inner
                .topologies
                .get(&instance_id)
                .map_or(0, |topology| topology.version);
            return Ok(RegistrationResult {
                response: RegisterWorkerResponse {
                    current_topology_version: topology_version,
                    current_placement_generation: existing.placement_generation,
                },
                replaced_start_id: None,
            });
        }

        let replaced = inner.workers.remove(&worker_id);
        let replaced_start_id = replaced
            .as_ref()
            .map(|worker| worker.registration.start_id.clone());
        let affected = replaced.as_ref().map(ready_keys).unwrap_or_default();
        let (placement_generation, targets) = replaced
            .map(|worker| (worker.placement_generation, worker.targets))
            .unwrap_or_default();

        inner.workers.insert(
            worker_id,
            WorkerRecord {
                registration,
                live: false,
                heartbeat_lease: 0,
                heartbeat_open: false,
                last_heartbeat_sequence: None,
                run_state: WorkerRunState::WorkerRunning,
                placement_generation,
                targets,
                expert_states: BTreeMap::new(),
                last_report: None,
                drains: VecDeque::new(),
            },
        );
        if publish_routes(&mut inner, instance_id, affected) {
            notify(&mut inner, &self.changed);
        }

        let topology_version = inner
            .topologies
            .get(&instance_id)
            .map_or(0, |topology| topology.version);
        Ok(RegistrationResult {
            response: RegisterWorkerResponse {
                current_topology_version: topology_version,
                current_placement_generation: placement_generation,
            },
            replaced_start_id,
        })
    }

    pub async fn open_heartbeat(
        &self,
        worker_id: &str,
        start_id: &str,
    ) -> Result<u64, ControllerStateError> {
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        inner.next_heartbeat_lease = inner.next_heartbeat_lease.wrapping_add(1).max(1);
        let lease = inner.next_heartbeat_lease;
        let worker = inner
            .workers
            .get_mut(worker_id)
            .expect("validated worker must exist");
        worker.heartbeat_lease = lease;
        worker.heartbeat_open = true;
        Ok(lease)
    }

    pub async fn heartbeat(
        &self,
        worker_id: &str,
        start_id: &str,
        lease: u64,
        sequence: u64,
        state: i32,
    ) -> Result<HeartbeatResult, ControllerStateError> {
        let run_state = WorkerRunState::try_from(state)
            .ok()
            .filter(|state| *state != WorkerRunState::Unspecified)
            .ok_or(ControllerStateError::InvalidHeartbeatState)?;
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;

        let (instance_id, became_live, state_changed, affected) = {
            let worker = inner
                .workers
                .get_mut(worker_id)
                .expect("validated worker must exist");
            if worker.heartbeat_lease != lease {
                return Err(ControllerStateError::StaleHeartbeatStream);
            }
            if let Some(last) = worker.last_heartbeat_sequence {
                if sequence < last {
                    return Err(ControllerStateError::HeartbeatSequenceRollback {
                        last,
                        received: sequence,
                    });
                }
                if sequence == last {
                    if worker.run_state != run_state {
                        return Err(ControllerStateError::InvalidHeartbeatState);
                    }
                    return Ok(HeartbeatResult::Duplicate);
                }
            }
            if worker.run_state == WorkerRunState::WorkerShuttingDown
                && run_state != WorkerRunState::WorkerShuttingDown
            {
                return Err(ControllerStateError::InvalidHeartbeatState);
            }
            let became_live = !worker.live;
            let state_changed = worker.run_state != run_state;
            worker.live = true;
            worker.last_heartbeat_sequence = Some(sequence);
            worker.run_state = run_state;
            (
                worker.registration.instance_id,
                became_live,
                state_changed,
                ready_keys(worker),
            )
        };
        let topology_changed = became_live && publish_routes(&mut inner, instance_id, affected);
        if topology_changed || state_changed {
            notify(&mut inner, &self.changed);
        }
        Ok(HeartbeatResult::Applied)
    }

    pub async fn close_heartbeat(
        &self,
        worker_id: &str,
        start_id: &str,
        lease: u64,
    ) -> Result<bool, ControllerStateError> {
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        let (instance_id, affected) = {
            let worker = inner
                .workers
                .get_mut(worker_id)
                .expect("validated worker must exist");
            if worker.heartbeat_lease != lease {
                return Ok(false);
            }
            if !worker.heartbeat_open {
                return Ok(false);
            }
            worker.heartbeat_open = false;
            worker.live = false;
            (worker.registration.instance_id, ready_keys(worker))
        };
        if publish_routes(&mut inner, instance_id, affected) {
            notify(&mut inner, &self.changed);
        }
        Ok(true)
    }

    pub async fn placement(
        &self,
        worker_id: &str,
        start_id: &str,
    ) -> Result<Placement, ControllerStateError> {
        let inner = self.inner.read().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        let worker = inner
            .workers
            .get(worker_id)
            .expect("validated worker must exist");
        Ok(Placement {
            generation: worker.placement_generation,
            changed: false,
            parts: target_parts(worker.placement_generation, worker.targets.values()),
        })
    }

    pub async fn registration(
        &self,
        worker_id: &str,
        start_id: &str,
    ) -> Result<RegisterWorkerRequest, ControllerStateError> {
        let inner = self.inner.read().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        Ok(inner
            .workers
            .get(worker_id)
            .expect("validated worker must exist")
            .registration
            .clone())
    }

    pub async fn ready_experts(
        &self,
        worker_id: &str,
        start_id: &str,
    ) -> Result<BTreeSet<ExpertKey>, ControllerStateError> {
        let inner = self.inner.read().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        Ok(ready_keys(
            inner
                .workers
                .get(worker_id)
                .expect("validated worker must exist"),
        ))
    }

    pub async fn set_targets(
        &self,
        worker_id: &str,
        start_id: &str,
        targets: Vec<TargetExpert>,
    ) -> Result<Placement, ControllerStateError> {
        let normalized = normalize_targets(targets);
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        let (instance_id, affected, removed_ready, generation, changed, values) = {
            let worker = inner
                .workers
                .get_mut(worker_id)
                .expect("validated worker must exist");
            let maximum = worker
                .registration
                .device
                .as_ref()
                .expect("registration validation requires a device")
                .max_experts;
            if normalized.len() > maximum as usize {
                return Err(ControllerStateError::PlacementTooLarge {
                    maximum,
                    received: normalized.len(),
                });
            }
            let changed = worker.targets != normalized;
            let mut affected: BTreeSet<ExpertKey> = worker.targets.keys().copied().collect();
            affected.extend(normalized.keys().copied());
            let removed_ready = worker
                .targets
                .keys()
                .filter(|key| !normalized.contains_key(key))
                .filter(|key| {
                    worker
                        .expert_states
                        .get(key)
                        .is_some_and(|state| state.state == ExpertStateKind::ExpertReady as i32)
                })
                .copied()
                .collect::<BTreeSet<_>>();
            if changed {
                worker.placement_generation = worker.placement_generation.wrapping_add(1).max(1);
                worker.targets = normalized;
            }
            (
                worker.registration.instance_id,
                affected,
                removed_ready,
                worker.placement_generation,
                changed,
                worker.targets.values().cloned().collect::<Vec<_>>(),
            )
        };
        if changed {
            let topology_changed = publish_routes(&mut inner, instance_id, affected);
            let drain_created = if removed_ready.is_empty() {
                false
            } else {
                let min_topology_version = inner
                    .topologies
                    .get(&instance_id)
                    .map_or(0, |topology| topology.version);
                let drain_id = next_drain_id(&mut inner);
                inner
                    .workers
                    .get_mut(worker_id)
                    .expect("validated worker must exist")
                    .drains
                    .push_back(DrainRecord {
                        drain_id,
                        placement_generation: generation,
                        min_topology_version,
                        stop_all: false,
                        experts: removed_ready,
                        completed: false,
                    });
                true
            };
            if topology_changed || drain_created {
                notify(&mut inner, &self.changed);
            }
        }
        Ok(Placement {
            generation,
            changed,
            parts: target_parts(generation, values.iter()),
        })
    }

    pub async fn apply_state_report(
        &self,
        worker_id: &str,
        start_id: &str,
        placement_generation: u64,
        report_sequence: u64,
        full: bool,
        states: Vec<ExpertState>,
    ) -> Result<StateReportResult, ControllerStateError> {
        let normalized = normalize_states(states)?;
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;

        let (instance_id, affected) = {
            let worker = inner
                .workers
                .get_mut(worker_id)
                .expect("validated worker must exist");
            if placement_generation != worker.placement_generation {
                return Err(ControllerStateError::PlacementGenerationMismatch {
                    expected: worker.placement_generation,
                    received: placement_generation,
                });
            }
            if let Some(last) = &worker.last_report {
                if report_sequence < last.sequence {
                    return Err(ControllerStateError::ReportSequenceRollback {
                        last: last.sequence,
                        received: report_sequence,
                    });
                }
                if report_sequence == last.sequence {
                    if last.full == full && last.states == normalized {
                        return Ok(StateReportResult::Duplicate);
                    }
                    return Err(ControllerStateError::ConflictingReportSequence(
                        report_sequence,
                    ));
                }
            }

            let mut affected: BTreeSet<ExpertKey> = if full {
                worker.expert_states.keys().copied().collect()
            } else {
                BTreeSet::new()
            };
            affected.extend(normalized.iter().map(ExpertKey::from_state));
            if full {
                worker.expert_states.clear();
            }
            for state in &normalized {
                worker
                    .expert_states
                    .insert(ExpertKey::from_state(state), state.clone());
            }
            worker.last_report = Some(StoredReport {
                sequence: report_sequence,
                full,
                states: normalized,
            });
            (worker.registration.instance_id, affected)
        };

        if publish_routes(&mut inner, instance_id, affected) {
            notify(&mut inner, &self.changed);
        }
        Ok(StateReportResult::Applied)
    }

    pub async fn drain_authorizations(
        &self,
        worker_id: &str,
        start_id: &str,
    ) -> Result<Vec<DrainAuthorization>, ControllerStateError> {
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        let should_create_shutdown = {
            let worker = inner
                .workers
                .get(worker_id)
                .expect("validated worker must exist");
            worker.run_state == WorkerRunState::WorkerShuttingDown
                && !worker.drains.iter().any(|drain| drain.stop_all)
                && shutdown_replacements_ready(&inner, worker_id)
        };
        if should_create_shutdown {
            let (instance_id, placement_generation, experts) = {
                let worker = inner
                    .workers
                    .get(worker_id)
                    .expect("validated worker must exist");
                (
                    worker.registration.instance_id,
                    worker.placement_generation,
                    ready_keys(worker),
                )
            };
            let min_topology_version = inner
                .topologies
                .get(&instance_id)
                .map_or(0, |topology| topology.version);
            let drain_id = next_drain_id(&mut inner);
            inner
                .workers
                .get_mut(worker_id)
                .expect("validated worker must exist")
                .drains
                .push_back(DrainRecord {
                    drain_id,
                    placement_generation,
                    min_topology_version,
                    stop_all: true,
                    experts,
                    completed: false,
                });
        }
        let worker = inner
            .workers
            .get(worker_id)
            .expect("validated worker must exist");
        Ok(worker
            .drains
            .iter()
            .filter(|drain| !drain.completed)
            .map(drain_authorization)
            .collect())
    }

    pub async fn complete_drain(
        &self,
        worker_id: &str,
        start_id: &str,
        drain_id: u64,
    ) -> Result<bool, ControllerStateError> {
        let mut inner = self.inner.write().await;
        ensure_current_start(&inner, worker_id, start_id)?;
        let (instance_id, stop_all, affected) = {
            let worker = inner
                .workers
                .get_mut(worker_id)
                .expect("validated worker must exist");
            let Some(drain) = worker
                .drains
                .iter_mut()
                .find(|drain| drain.drain_id == drain_id)
            else {
                return Err(ControllerStateError::UnknownDrain(drain_id));
            };
            if drain.completed {
                return Ok(false);
            }
            drain.completed = true;
            let stop_all = drain.stop_all;
            let affected = if stop_all {
                worker.live = false;
                ready_keys(worker)
            } else {
                BTreeSet::new()
            };
            (worker.registration.instance_id, stop_all, affected)
        };
        if stop_all && publish_routes(&mut inner, instance_id, affected) {
            notify(&mut inner, &self.changed);
        }
        Ok(true)
    }

    pub async fn topology_version(&self, instance_id: u64) -> u64 {
        self.inner
            .read()
            .await
            .topologies
            .get(&instance_id)
            .map_or(0, |topology| topology.version)
    }

    pub async fn topology_messages(
        &self,
        instance_id: u64,
        current_version: u64,
    ) -> Result<Vec<TopologyMessage>, ControllerStateError> {
        let inner = self.inner.read().await;
        let Some(topology) = inner.topologies.get(&instance_id) else {
            if current_version > 0 {
                return Err(ControllerStateError::FutureTopologyVersion {
                    current: 0,
                    requested: current_version,
                });
            }
            return Ok(snapshot_messages(instance_id, &InstanceTopology::default()));
        };
        if current_version > topology.version {
            return Err(ControllerStateError::FutureTopologyVersion {
                current: topology.version,
                requested: current_version,
            });
        }
        if current_version == topology.version {
            return Ok(Vec::new());
        }

        let mut cursor = current_version;
        let mut revisions = Vec::new();
        for revision in &topology.history {
            if revision.previous_version == cursor {
                revisions.push(revision);
                cursor = revision.topology_version;
            }
        }
        if cursor == topology.version {
            return Ok(revisions
                .into_iter()
                .flat_map(|revision| update_messages(instance_id, revision))
                .collect());
        }
        Ok(snapshot_messages(instance_id, topology))
    }
}

fn ensure_current_start(
    inner: &Inner,
    worker_id: &str,
    start_id: &str,
) -> Result<(), ControllerStateError> {
    let worker = inner
        .workers
        .get(worker_id)
        .ok_or(ControllerStateError::UnknownWorker)?;
    if worker.registration.start_id != start_id {
        return Err(ControllerStateError::ReplacedWorker);
    }
    Ok(())
}

fn validate_registration(registration: &RegisterWorkerRequest) -> Result<(), ControllerStateError> {
    if registration.worker_id.trim().is_empty()
        || registration.start_id.trim().is_empty()
        || registration.instance_id == 0
        || registration.backend.trim().is_empty()
    {
        return Err(ControllerStateError::InvalidRegistration(
            "identity, instance, and backend are required",
        ));
    }
    if !matches!(registration.backend.as_str(), "torch" | "ggml" | "fused") {
        return Err(ControllerStateError::InvalidRegistration(
            "backend is not supported",
        ));
    }
    if ActivationDType::try_from(registration.activation_dtype)
        .ok()
        .filter(|dtype| *dtype != ActivationDType::ActivationDtypeUnspecified)
        .is_none()
    {
        return Err(ControllerStateError::InvalidRegistration(
            "activation dtype is invalid",
        ));
    }
    let Some(device) = registration.device.as_ref() else {
        return Err(ControllerStateError::InvalidRegistration(
            "one device is required",
        ));
    };
    if device.device.trim().is_empty()
        || device.max_experts == 0
        || registration.max_batch_tokens == 0
        || registration.max_active_batches_per_device == 0
        || registration.max_pending_batches_per_device == 0
    {
        return Err(ControllerStateError::InvalidRegistration(
            "device and capacity limits must be positive",
        ));
    }
    validate_host_port(&registration.computation_endpoint)?;
    let peer = url::Url::parse(&registration.peer_weight_endpoint).map_err(|_| {
        ControllerStateError::InvalidRegistration("peer weight endpoint must be an HTTP URL")
    })?;
    if !matches!(peer.scheme(), "http" | "https") || peer.host().is_none() || peer.port().is_none()
    {
        return Err(ControllerStateError::InvalidRegistration(
            "peer weight endpoint must include an HTTP host and port",
        ));
    }
    Ok(())
}

fn validate_host_port(value: &str) -> Result<(), ControllerStateError> {
    if value.trim() != value || value.contains("://") {
        return Err(ControllerStateError::InvalidRegistration(
            "computation endpoint must use host:port without a scheme",
        ));
    }
    let uri = format!("http://{value}").parse::<tonic::codegen::http::Uri>();
    let Ok(uri) = uri else {
        return Err(ControllerStateError::InvalidRegistration(
            "computation endpoint is invalid",
        ));
    };
    if uri.host().is_none() || uri.port_u16().is_none_or(|port| port == 0) {
        return Err(ControllerStateError::InvalidRegistration(
            "computation endpoint must include a host and port",
        ));
    }
    Ok(())
}

fn normalize_targets(targets: Vec<TargetExpert>) -> BTreeMap<ExpertKey, TargetExpert> {
    targets
        .into_iter()
        .map(|mut target| {
            target.peer_weight_endpoints.sort();
            target.peer_weight_endpoints.dedup();
            (ExpertKey::from_target(&target), target)
        })
        .collect()
}

fn normalize_states(states: Vec<ExpertState>) -> Result<Vec<ExpertState>, ControllerStateError> {
    let mut normalized = BTreeMap::new();
    for state in states {
        let kind = ExpertStateKind::try_from(state.state)
            .ok()
            .filter(|kind| *kind != ExpertStateKind::ExpertStateUnspecified)
            .ok_or(ControllerStateError::InvalidExpertState)?;
        if kind != ExpertStateKind::ExpertFailed && state.failure.is_some() {
            return Err(ControllerStateError::InvalidExpertState);
        }
        normalized.insert(ExpertKey::from_state(&state), state);
    }
    Ok(normalized.into_values().collect())
}

fn next_drain_id(inner: &mut Inner) -> u64 {
    inner.next_drain_id = inner.next_drain_id.wrapping_add(1).max(1);
    inner.next_drain_id
}

fn shutdown_replacements_ready(inner: &Inner, worker_id: &str) -> bool {
    let worker = inner
        .workers
        .get(worker_id)
        .expect("validated worker must exist");
    let topology = inner.topologies.get(&worker.registration.instance_id);
    ready_keys(worker).into_iter().all(|key| {
        topology
            .and_then(|topology| topology.routes.get(&key))
            .is_some_and(|replicas| {
                replicas.iter().any(|replica| {
                    replica.worker_id != worker.registration.worker_id
                        || replica.start_id != worker.registration.start_id
                })
            })
    })
}

fn drain_authorization(drain: &DrainRecord) -> DrainAuthorization {
    let experts: Vec<ProtoExpertKey> = drain
        .experts
        .iter()
        .map(|key| ProtoExpertKey {
            layer_id: key.layer_id,
            expert_id: key.expert_id,
        })
        .collect();
    let part_count = experts.len().div_ceil(MAX_CONTROL_ENTRIES_PER_PART).max(1);
    let parts = (0..part_count)
        .map(|part_index| {
            let start = part_index * MAX_CONTROL_ENTRIES_PER_PART;
            let end = (start + MAX_CONTROL_ENTRIES_PER_PART).min(experts.len());
            DrainAuthorizationPart {
                drain_id: drain.drain_id,
                placement_generation: drain.placement_generation,
                min_topology_version: drain.min_topology_version,
                stop_accepting_all_computation: drain.stop_all,
                part_index: part_index as u32,
                part_count: part_count as u32,
                experts: experts[start..end].to_vec(),
            }
        })
        .collect();
    DrainAuthorization {
        drain_id: drain.drain_id,
        parts,
    }
}

fn ready_keys(worker: &WorkerRecord) -> BTreeSet<ExpertKey> {
    worker
        .expert_states
        .iter()
        .filter_map(|(key, state)| {
            (state.state == ExpertStateKind::ExpertReady as i32).then_some(*key)
        })
        .collect()
}

fn publish_routes(inner: &mut Inner, instance_id: u64, affected: BTreeSet<ExpertKey>) -> bool {
    if affected.is_empty() {
        return false;
    }

    let mut changes = Vec::new();
    for key in affected {
        let mut replicas: Vec<WorkerRoute> = inner
            .workers
            .values()
            .filter(|worker| worker.registration.instance_id == instance_id && worker.live)
            .filter(|worker| worker.targets.contains_key(&key))
            .filter(|worker| {
                worker
                    .expert_states
                    .get(&key)
                    .is_some_and(|state| state.state == ExpertStateKind::ExpertReady as i32)
            })
            .map(worker_route)
            .collect();
        replicas.sort_by(|left, right| {
            (&left.worker_id, &left.start_id).cmp(&(&right.worker_id, &right.start_id))
        });

        let topology = inner.topologies.entry(instance_id).or_default();
        let previous = topology.routes.get(&key).cloned().unwrap_or_default();
        if previous == replicas {
            continue;
        }
        if replicas.is_empty() {
            topology.routes.remove(&key);
        } else {
            topology.routes.insert(key, replicas.clone());
        }
        changes.push(RouteChange {
            layer_id: key.layer_id,
            expert_id: key.expert_id,
            replicas,
        });
    }
    if changes.is_empty() {
        return false;
    }

    let topology = inner.topologies.entry(instance_id).or_default();
    let previous_version = topology.version;
    topology.version = topology.version.wrapping_add(1).max(1);
    topology.history.push_back(TopologyRevision {
        previous_version,
        topology_version: topology.version,
        changes,
    });
    while topology.history.len() > inner.history_limit {
        topology.history.pop_front();
    }
    true
}

fn notify(inner: &mut Inner, changed: &watch::Sender<u64>) {
    inner.notification_sequence = inner.notification_sequence.wrapping_add(1).max(1);
    changed.send_replace(inner.notification_sequence);
}

fn worker_route(worker: &WorkerRecord) -> WorkerRoute {
    let device = worker
        .registration
        .device
        .as_ref()
        .expect("registration validation requires a device");
    WorkerRoute {
        worker_id: worker.registration.worker_id.clone(),
        start_id: worker.registration.start_id.clone(),
        computation_endpoint: worker.registration.computation_endpoint.clone(),
        device: device.device.clone(),
        max_active_batches: worker.registration.max_active_batches_per_device,
        max_pending_batches: worker.registration.max_pending_batches_per_device,
        max_batch_tokens: worker.registration.max_batch_tokens,
    }
}

fn target_parts<'a>(
    generation: u64,
    targets: impl IntoIterator<Item = &'a TargetExpert>,
) -> Vec<TargetExpertListPart> {
    let targets: Vec<TargetExpert> = targets.into_iter().cloned().collect();
    let part_count = targets.len().div_ceil(MAX_CONTROL_ENTRIES_PER_PART).max(1);
    (0..part_count)
        .map(|part_index| {
            let start = part_index * MAX_CONTROL_ENTRIES_PER_PART;
            let end = (start + MAX_CONTROL_ENTRIES_PER_PART).min(targets.len());
            TargetExpertListPart {
                placement_generation: generation,
                part_index: part_index as u32,
                part_count: part_count as u32,
                experts: targets[start..end].to_vec(),
            }
        })
        .collect()
}

fn snapshot_messages(instance_id: u64, topology: &InstanceTopology) -> Vec<TopologyMessage> {
    let routes: Vec<ExpertRoute> = topology
        .routes
        .iter()
        .map(|(key, replicas)| ExpertRoute {
            layer_id: key.layer_id,
            expert_id: key.expert_id,
            replicas: replicas.clone(),
        })
        .collect();
    let part_count = routes.len().div_ceil(MAX_CONTROL_ENTRIES_PER_PART).max(1);
    (0..part_count)
        .map(|part_index| {
            let start = part_index * MAX_CONTROL_ENTRIES_PER_PART;
            let end = (start + MAX_CONTROL_ENTRIES_PER_PART).min(routes.len());
            TopologyMessage {
                message: Some(topology_message::Message::Snapshot(TopologySnapshotPart {
                    instance_id,
                    topology_version: topology.version,
                    part_index: part_index as u32,
                    part_count: part_count as u32,
                    routes: routes[start..end].to_vec(),
                })),
            }
        })
        .collect()
}

fn update_messages(instance_id: u64, revision: &TopologyRevision) -> Vec<TopologyMessage> {
    let part_count = revision
        .changes
        .len()
        .div_ceil(MAX_CONTROL_ENTRIES_PER_PART)
        .max(1);
    (0..part_count)
        .map(|part_index| {
            let start = part_index * MAX_CONTROL_ENTRIES_PER_PART;
            let end = (start + MAX_CONTROL_ENTRIES_PER_PART).min(revision.changes.len());
            TopologyMessage {
                message: Some(topology_message::Message::Update(TopologyUpdatePart {
                    instance_id,
                    previous_version: revision.previous_version,
                    topology_version: revision.topology_version,
                    part_index: part_index as u32,
                    part_count: part_count as u32,
                    changes: revision.changes[start..end].to_vec(),
                })),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::ek::control::v2::WorkerDevice;

    fn registration(worker_id: &str, start_id: &str) -> RegisterWorkerRequest {
        RegisterWorkerRequest {
            worker_id: worker_id.to_owned(),
            start_id: start_id.to_owned(),
            instance_id: 7,
            computation_endpoint: "127.0.0.1:50051".to_owned(),
            peer_weight_endpoint: "http://127.0.0.1:50052".to_owned(),
            backend: "torch".to_owned(),
            activation_dtype: ActivationDType::ActivationDtypeBf16 as i32,
            device: Some(WorkerDevice {
                device: "cuda:0".to_owned(),
                max_experts: 256,
            }),
            max_batch_tokens: 4096,
            max_active_batches_per_device: 2,
            max_pending_batches_per_device: 2,
        }
    }

    fn target(layer_id: u32, expert_id: u32) -> TargetExpert {
        TargetExpert {
            layer_id,
            expert_id,
            target_device: "cuda:0".to_owned(),
            peer_weight_endpoints: vec!["http://peer:8000".to_owned()],
        }
    }

    fn ready(layer_id: u32, expert_id: u32) -> ExpertState {
        ExpertState {
            layer_id,
            expert_id,
            state: ExpertStateKind::ExpertReady as i32,
            failure: None,
        }
    }

    async fn register_live_worker(state: &ControllerV2State) -> u64 {
        register_live_named(state, "worker-0", "start-0").await
    }

    async fn register_live_named(
        state: &ControllerV2State,
        worker_id: &str,
        start_id: &str,
    ) -> u64 {
        state
            .register(registration(worker_id, start_id))
            .await
            .unwrap();
        let lease = state.open_heartbeat(worker_id, start_id).await.unwrap();
        state
            .heartbeat(
                worker_id,
                start_id,
                lease,
                1,
                WorkerRunState::WorkerRunning as i32,
            )
            .await
            .unwrap();
        lease
    }

    #[tokio::test]
    async fn registration_is_idempotent_but_a_new_start_replaces_routes() {
        let state = ControllerV2State::new(8);
        register_live_worker(&state).await;
        let placement = state
            .set_targets("worker-0", "start-0", vec![target(1, 2)])
            .await
            .unwrap();
        state
            .apply_state_report(
                "worker-0",
                "start-0",
                placement.generation,
                1,
                true,
                vec![ready(1, 2)],
            )
            .await
            .unwrap();
        assert_eq!(state.topology_version(7).await, 1);

        let repeated = state
            .register(registration("worker-0", "start-0"))
            .await
            .unwrap();
        assert_eq!(repeated.response.current_placement_generation, 1);
        assert_eq!(state.topology_version(7).await, 1);

        let replacement = state
            .register(registration("worker-0", "start-1"))
            .await
            .unwrap();
        assert_eq!(replacement.replaced_start_id.as_deref(), Some("start-0"));
        assert_eq!(replacement.response.current_placement_generation, 1);
        assert_eq!(state.topology_version(7).await, 2);
        assert_eq!(
            state.open_heartbeat("worker-0", "start-0").await,
            Err(ControllerStateError::ReplacedWorker)
        );
    }

    #[tokio::test]
    async fn heartbeat_sequence_and_stream_lease_are_monotonic() {
        let state = ControllerV2State::new(8);
        state
            .register(registration("worker-0", "start-0"))
            .await
            .unwrap();
        let first = state.open_heartbeat("worker-0", "start-0").await.unwrap();
        assert_eq!(
            state
                .heartbeat(
                    "worker-0",
                    "start-0",
                    first,
                    4,
                    WorkerRunState::WorkerRunning as i32,
                )
                .await,
            Ok(HeartbeatResult::Applied)
        );
        assert_eq!(
            state
                .heartbeat(
                    "worker-0",
                    "start-0",
                    first,
                    4,
                    WorkerRunState::WorkerRunning as i32,
                )
                .await,
            Ok(HeartbeatResult::Duplicate)
        );
        assert_eq!(
            state
                .heartbeat(
                    "worker-0",
                    "start-0",
                    first,
                    3,
                    WorkerRunState::WorkerRunning as i32,
                )
                .await,
            Err(ControllerStateError::HeartbeatSequenceRollback {
                last: 4,
                received: 3,
            })
        );

        let second = state.open_heartbeat("worker-0", "start-0").await.unwrap();
        assert!(
            !state
                .close_heartbeat("worker-0", "start-0", first)
                .await
                .unwrap()
        );
        assert!(
            state
                .close_heartbeat("worker-0", "start-0", second)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn ready_routes_require_target_and_live_worker() {
        let state = ControllerV2State::new(8);
        let lease = register_live_worker(&state).await;
        let placement = state
            .set_targets("worker-0", "start-0", vec![target(1, 2)])
            .await
            .unwrap();
        assert_eq!(placement.parts[0].experts.len(), 1);
        assert_eq!(state.topology_version(7).await, 0);

        state
            .apply_state_report(
                "worker-0",
                "start-0",
                placement.generation,
                1,
                true,
                vec![ready(1, 2), ready(1, 3)],
            )
            .await
            .unwrap();
        let messages = state.topology_messages(7, 0).await.unwrap();
        let topology_message::Message::Update(update) = messages[0].message.as_ref().unwrap()
        else {
            panic!("expected retained update");
        };
        assert_eq!(update.changes.len(), 1);
        assert_eq!(update.changes[0].expert_id, 2);
        assert_eq!(update.changes[0].replicas[0].worker_id, "worker-0");

        assert!(
            state
                .close_heartbeat("worker-0", "start-0", lease)
                .await
                .unwrap()
        );
        let messages = state.topology_messages(7, 1).await.unwrap();
        let topology_message::Message::Update(update) = messages[0].message.as_ref().unwrap()
        else {
            panic!("expected removal update");
        };
        assert!(update.changes[0].replicas.is_empty());
    }

    #[tokio::test]
    async fn state_reports_reject_generation_and_sequence_regressions() {
        let state = ControllerV2State::new(8);
        register_live_worker(&state).await;
        let placement = state
            .set_targets("worker-0", "start-0", vec![target(1, 2)])
            .await
            .unwrap();
        let report = vec![ready(1, 2)];
        assert_eq!(
            state
                .apply_state_report(
                    "worker-0",
                    "start-0",
                    placement.generation,
                    2,
                    true,
                    report.clone(),
                )
                .await,
            Ok(StateReportResult::Applied)
        );
        assert_eq!(
            state
                .apply_state_report("worker-0", "start-0", placement.generation, 2, true, report,)
                .await,
            Ok(StateReportResult::Duplicate)
        );
        assert!(matches!(
            state
                .apply_state_report(
                    "worker-0",
                    "start-0",
                    placement.generation,
                    1,
                    false,
                    vec![],
                )
                .await,
            Err(ControllerStateError::ReportSequenceRollback { .. })
        ));
        assert!(matches!(
            state
                .apply_state_report("worker-0", "start-0", 99, 3, false, vec![])
                .await,
            Err(ControllerStateError::PlacementGenerationMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn topology_replays_history_or_falls_back_to_snapshot() {
        let state = ControllerV2State::new(1);
        let lease = register_live_worker(&state).await;
        let placement = state
            .set_targets("worker-0", "start-0", vec![target(1, 2)])
            .await
            .unwrap();
        state
            .apply_state_report(
                "worker-0",
                "start-0",
                placement.generation,
                1,
                true,
                vec![ready(1, 2)],
            )
            .await
            .unwrap();
        state
            .close_heartbeat("worker-0", "start-0", lease)
            .await
            .unwrap();

        let retained = state.topology_messages(7, 1).await.unwrap();
        assert!(matches!(
            retained[0].message,
            Some(topology_message::Message::Update(_))
        ));
        let recovered = state.topology_messages(7, 0).await.unwrap();
        assert!(matches!(
            recovered[0].message,
            Some(topology_message::Message::Snapshot(_))
        ));
        let topology_message::Message::Snapshot(snapshot) = recovered[0].message.as_ref().unwrap()
        else {
            unreachable!();
        };
        assert_eq!(snapshot.topology_version, 2);
        assert!(snapshot.routes.is_empty());
    }

    #[tokio::test]
    async fn topology_and_target_parts_are_bounded() {
        let state = ControllerV2State::new(8);
        register_live_worker(&state).await;
        let targets: Vec<TargetExpert> = (0..129).map(|expert_id| target(1, expert_id)).collect();
        let placement = state
            .set_targets("worker-0", "start-0", targets)
            .await
            .unwrap();
        assert_eq!(placement.parts.len(), 3);
        assert_eq!(placement.parts[0].experts.len(), 64);
        assert_eq!(placement.parts[2].experts.len(), 1);

        state
            .apply_state_report(
                "worker-0",
                "start-0",
                placement.generation,
                1,
                true,
                (0..129).map(|expert_id| ready(1, expert_id)).collect(),
            )
            .await
            .unwrap();
        let messages = state.topology_messages(7, 0).await.unwrap();
        assert_eq!(messages.len(), 3);
    }

    #[tokio::test]
    async fn target_removal_is_authorized_after_the_route_update() {
        let state = ControllerV2State::new(8);
        register_live_worker(&state).await;
        let placement = state
            .set_targets("worker-0", "start-0", vec![target(1, 2), target(1, 3)])
            .await
            .unwrap();
        state
            .apply_state_report(
                "worker-0",
                "start-0",
                placement.generation,
                1,
                true,
                vec![ready(1, 2), ready(1, 3)],
            )
            .await
            .unwrap();
        let next = state
            .set_targets("worker-0", "start-0", vec![target(1, 2)])
            .await
            .unwrap();

        let drains = state
            .drain_authorizations("worker-0", "start-0")
            .await
            .unwrap();
        assert_eq!(drains.len(), 1);
        assert!(!drains[0].parts[0].stop_accepting_all_computation);
        assert_eq!(drains[0].parts[0].placement_generation, next.generation);
        assert_eq!(drains[0].parts[0].min_topology_version, 2);
        assert_eq!(drains[0].parts[0].experts[0].expert_id, 3);
        assert!(
            state
                .complete_drain("worker-0", "start-0", drains[0].drain_id)
                .await
                .unwrap()
        );
        assert!(
            !state
                .complete_drain("worker-0", "start-0", drains[0].drain_id)
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn whole_worker_drain_waits_for_ready_replacements() {
        let state = ControllerV2State::new(8);
        let first_lease = register_live_named(&state, "worker-0", "start-0").await;
        let first_placement = state
            .set_targets("worker-0", "start-0", vec![target(1, 2)])
            .await
            .unwrap();
        state
            .apply_state_report(
                "worker-0",
                "start-0",
                first_placement.generation,
                1,
                true,
                vec![ready(1, 2)],
            )
            .await
            .unwrap();
        state
            .heartbeat(
                "worker-0",
                "start-0",
                first_lease,
                2,
                WorkerRunState::WorkerShuttingDown as i32,
            )
            .await
            .unwrap();
        assert!(
            state
                .drain_authorizations("worker-0", "start-0")
                .await
                .unwrap()
                .is_empty()
        );

        register_live_named(&state, "worker-1", "start-1").await;
        let replacement = state
            .set_targets("worker-1", "start-1", vec![target(1, 2)])
            .await
            .unwrap();
        state
            .apply_state_report(
                "worker-1",
                "start-1",
                replacement.generation,
                1,
                true,
                vec![ready(1, 2)],
            )
            .await
            .unwrap();
        let drains = state
            .drain_authorizations("worker-0", "start-0")
            .await
            .unwrap();
        assert_eq!(drains.len(), 1);
        assert!(drains[0].parts[0].stop_accepting_all_computation);
        assert_eq!(drains[0].parts[0].min_topology_version, 2);

        state
            .complete_drain("worker-0", "start-0", drains[0].drain_id)
            .await
            .unwrap();
        assert!(
            state
                .close_heartbeat("worker-0", "start-0", first_lease)
                .await
                .unwrap()
        );
    }
}
