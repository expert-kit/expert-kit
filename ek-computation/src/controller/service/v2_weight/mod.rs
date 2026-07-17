//! v2 placement, expert-state, and drain synchronization service.

use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    pin::Pin,
    sync::Arc,
};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

use crate::{
    controller::v2_state::{
        ControllerV2State, ExpertKey, MAX_CONTROL_ENTRIES_PER_PART, Placement, StateReportResult,
    },
    proto::ek::control::v2::{
        ControllerWeightMessage, ExpertState, FullExpertStatePart, RegisterWorkerRequest,
        StateReportAck, TargetExpert, WorkerWeightMessage, controller_weight_message,
        weight_control_service_server::WeightControlService, worker_weight_message,
    },
};

use super::v2::state_status;

mod database;

pub use database::DatabaseWeightControlHooks;

const WEIGHT_STREAM_BUFFER: usize = 16;

pub struct TargetSubscription {
    subscription_id: u64,
    initial: Vec<TargetExpert>,
    updates: mpsc::Receiver<Result<Vec<TargetExpert>, Status>>,
}

impl TargetSubscription {
    /// Build one placement subscription supplied by a Controller integration.
    ///
    /// The identifier must remain unique for the lifetime of the matching Worker
    /// weight stream. Closing `updates` ends that stream and causes `unsubscribe`
    /// to be called with the same identifier.
    pub fn new(
        subscription_id: u64,
        initial: Vec<TargetExpert>,
        updates: mpsc::Receiver<Result<Vec<TargetExpert>, Status>>,
    ) -> Self {
        assert!(subscription_id > 0, "subscription ID must be positive");
        Self {
            subscription_id,
            initial,
            updates,
        }
    }
}

#[async_trait]
pub trait WeightControlHooks: Send + Sync + 'static {
    async fn subscribe(
        &self,
        registration: &RegisterWorkerRequest,
    ) -> Result<TargetSubscription, Status>;

    async fn unsubscribe(&self, worker_id: &str, subscription_id: u64);

    async fn persist_ready(
        &self,
        worker_id: &str,
        ready: &BTreeSet<ExpertKey>,
    ) -> Result<(), Status>;
}

#[derive(Clone)]
pub struct WeightControlServiceImpl {
    state: ControllerV2State,
    hooks: Arc<dyn WeightControlHooks>,
}

impl WeightControlServiceImpl {
    pub fn new(state: ControllerV2State, hooks: Arc<dyn WeightControlHooks>) -> Self {
        Self { state, hooks }
    }
}

struct WeightSession {
    state: ControllerV2State,
    hooks: Arc<dyn WeightControlHooks>,
    worker_id: String,
    start_id: String,
    max_experts: u32,
    responses: mpsc::Sender<Result<ControllerWeightMessage, Status>>,
}

impl WeightSession {
    async fn run<S>(
        self,
        mut requests: S,
        mut subscription: TargetSubscription,
    ) -> Result<(), Status>
    where
        S: Stream<Item = Result<WorkerWeightMessage, Status>> + Unpin,
    {
        let mut assembler = FullStateAssembler::new(self.max_experts);
        let mut state_changed = self.state.subscribe();
        let mut sent_drains = HashSet::new();

        let placement = self
            .state
            .set_targets(&self.worker_id, &self.start_id, subscription.initial)
            .await
            .map_err(state_status)?;
        send_placement(&self.responses, &placement).await?;
        send_new_drains(
            &self.state,
            &self.worker_id,
            &self.start_id,
            &self.responses,
            &mut sent_drains,
        )
        .await?;

        loop {
            tokio::select! {
                request = requests.next() => {
                    let Some(request) = request else {
                        return Ok(());
                    };
                    let request = request?;
                    handle_worker_message(
                        &self.state,
                        self.hooks.as_ref(),
                        &self.worker_id,
                        &self.start_id,
                        &self.responses,
                        &mut assembler,
                        request,
                    ).await?;
                    send_new_drains(
                        &self.state,
                        &self.worker_id,
                        &self.start_id,
                        &self.responses,
                        &mut sent_drains,
                    ).await?;
                }
                targets = subscription.updates.recv() => {
                    let Some(targets) = targets else {
                        return Ok(());
                    };
                    let placement = self.state
                        .set_targets(&self.worker_id, &self.start_id, targets?)
                        .await
                        .map_err(state_status)?;
                    if placement.changed {
                        send_placement(&self.responses, &placement).await?;
                    }
                    send_new_drains(
                        &self.state,
                        &self.worker_id,
                        &self.start_id,
                        &self.responses,
                        &mut sent_drains,
                    ).await?;
                }
                changed = state_changed.changed() => {
                    if changed.is_err() {
                        return Ok(());
                    }
                    send_new_drains(
                        &self.state,
                        &self.worker_id,
                        &self.start_id,
                        &self.responses,
                        &mut sent_drains,
                    ).await?;
                }
            }
        }
    }
}

#[tonic::async_trait]
impl WeightControlService for WeightControlServiceImpl {
    type SyncStream =
        Pin<Box<dyn Stream<Item = Result<ControllerWeightMessage, Status>> + Send + 'static>>;

    async fn sync(
        &self,
        request: Request<Streaming<WorkerWeightMessage>>,
    ) -> Result<Response<Self::SyncStream>, Status> {
        let mut requests = request.into_inner();
        let first = requests
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("weight stream is empty"))?;
        let Some(worker_weight_message::Message::Open(open)) = first.message else {
            return Err(Status::invalid_argument(
                "the first weight-stream message must be open",
            ));
        };
        let registration = self
            .state
            .registration(&open.worker_id, &open.start_id)
            .await
            .map_err(state_status)?;
        let max_experts = registration
            .device
            .as_ref()
            .expect("validated registration has one device")
            .max_experts;
        let subscription = self.hooks.subscribe(&registration).await?;
        let subscription_id = subscription.subscription_id;
        let worker_id = open.worker_id;
        let start_id = open.start_id;
        let state = self.state.clone();
        let hooks = self.hooks.clone();
        let (sender, receiver) = mpsc::channel(WEIGHT_STREAM_BUFFER);
        tokio::spawn(async move {
            let result = WeightSession {
                state,
                hooks: hooks.clone(),
                worker_id: worker_id.clone(),
                start_id,
                max_experts,
                responses: sender.clone(),
            }
            .run(requests, subscription)
            .await;
            hooks.unsubscribe(&worker_id, subscription_id).await;
            if let Err(status) = result {
                let _ = sender.send(Err(status)).await;
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}

async fn handle_worker_message(
    state: &ControllerV2State,
    hooks: &dyn WeightControlHooks,
    worker_id: &str,
    start_id: &str,
    responses: &mpsc::Sender<Result<ControllerWeightMessage, Status>>,
    assembler: &mut FullStateAssembler,
    request: WorkerWeightMessage,
) -> Result<(), Status> {
    match request.message {
        Some(worker_weight_message::Message::FullState(part)) => {
            if let Some(report) = assembler
                .add(part)
                .map_err(|error| Status::invalid_argument(error.0))?
            {
                apply_report(state, hooks, worker_id, start_id, responses, report).await?;
            }
        }
        Some(worker_weight_message::Message::StateUpdates(updates)) => {
            if updates.report_sequence == 0 || updates.experts.len() > MAX_CONTROL_ENTRIES_PER_PART
            {
                return Err(Status::invalid_argument("state update is not bounded"));
            }
            apply_report(
                state,
                hooks,
                worker_id,
                start_id,
                responses,
                CompleteStateReport {
                    placement_generation: updates.placement_generation,
                    report_sequence: updates.report_sequence,
                    full: false,
                    experts: updates.experts,
                },
            )
            .await?;
        }
        Some(worker_weight_message::Message::DrainComplete(completion)) => {
            state
                .complete_drain(worker_id, start_id, completion.drain_id)
                .await
                .map_err(state_status)?;
        }
        Some(worker_weight_message::Message::Open(_)) => {
            return Err(Status::invalid_argument(
                "open may appear only as the first stream message",
            ));
        }
        None => return Err(Status::invalid_argument("weight message has no payload")),
    }
    Ok(())
}

async fn apply_report(
    state: &ControllerV2State,
    hooks: &dyn WeightControlHooks,
    worker_id: &str,
    start_id: &str,
    responses: &mpsc::Sender<Result<ControllerWeightMessage, Status>>,
    report: CompleteStateReport,
) -> Result<(), Status> {
    let result = state
        .apply_state_report(
            worker_id,
            start_id,
            report.placement_generation,
            report.report_sequence,
            report.full,
            report.experts,
        )
        .await
        .map_err(state_status)?;
    if result == StateReportResult::Applied {
        let ready = state
            .ready_experts(worker_id, start_id)
            .await
            .map_err(state_status)?;
        hooks.persist_ready(worker_id, &ready).await?;
    }
    send_response(
        responses,
        ControllerWeightMessage {
            message: Some(controller_weight_message::Message::StateAck(
                StateReportAck {
                    report_sequence: report.report_sequence,
                },
            )),
        },
    )
    .await
}

async fn send_placement(
    responses: &mpsc::Sender<Result<ControllerWeightMessage, Status>>,
    placement: &Placement,
) -> Result<(), Status> {
    for part in &placement.parts {
        send_response(
            responses,
            ControllerWeightMessage {
                message: Some(controller_weight_message::Message::Targets(part.clone())),
            },
        )
        .await?;
    }
    Ok(())
}

async fn send_new_drains(
    state: &ControllerV2State,
    worker_id: &str,
    start_id: &str,
    responses: &mpsc::Sender<Result<ControllerWeightMessage, Status>>,
    sent_drains: &mut HashSet<u64>,
) -> Result<(), Status> {
    for authorization in state
        .drain_authorizations(worker_id, start_id)
        .await
        .map_err(state_status)?
    {
        if !sent_drains.insert(authorization.drain_id) {
            continue;
        }
        for part in authorization.parts {
            send_response(
                responses,
                ControllerWeightMessage {
                    message: Some(controller_weight_message::Message::Drain(part)),
                },
            )
            .await?;
        }
    }
    Ok(())
}

async fn send_response(
    responses: &mpsc::Sender<Result<ControllerWeightMessage, Status>>,
    response: ControllerWeightMessage,
) -> Result<(), Status> {
    responses
        .send(Ok(response))
        .await
        .map_err(|_| Status::cancelled("weight response stream closed"))
}

#[derive(Debug)]
struct CompleteStateReport {
    placement_generation: u64,
    report_sequence: u64,
    full: bool,
    experts: Vec<ExpertState>,
}

struct FullStateAssembler {
    max_parts: u32,
    active: Option<PartialStateReport>,
}

struct PartialStateReport {
    placement_generation: u64,
    report_sequence: u64,
    part_count: u32,
    parts: BTreeMap<u32, Vec<ExpertState>>,
}

#[derive(Debug)]
struct FullStateError(&'static str);

impl FullStateAssembler {
    fn new(max_experts: u32) -> Self {
        let max_parts = (max_experts as usize)
            .div_ceil(MAX_CONTROL_ENTRIES_PER_PART)
            .max(1) as u32;
        Self {
            max_parts,
            active: None,
        }
    }

    fn add(
        &mut self,
        part: FullExpertStatePart,
    ) -> Result<Option<CompleteStateReport>, FullStateError> {
        if part.report_sequence == 0
            || part.part_count == 0
            || part.part_count > self.max_parts
            || part.part_index >= part.part_count
            || part.experts.len() > MAX_CONTROL_ENTRIES_PER_PART
        {
            return Err(FullStateError("full state part is not bounded"));
        }
        let active = self.active.get_or_insert_with(|| PartialStateReport {
            placement_generation: part.placement_generation,
            report_sequence: part.report_sequence,
            part_count: part.part_count,
            parts: BTreeMap::new(),
        });
        if active.placement_generation != part.placement_generation
            || active.report_sequence != part.report_sequence
            || active.part_count != part.part_count
        {
            return Err(FullStateError(
                "full state parts describe different reports",
            ));
        }
        if let Some(existing) = active.parts.get(&part.part_index) {
            if existing == &part.experts {
                return Ok(None);
            }
            return Err(FullStateError(
                "full state part was repeated with different data",
            ));
        }
        active.parts.insert(part.part_index, part.experts);
        if active.parts.len() != active.part_count as usize {
            return Ok(None);
        }

        let active = self
            .active
            .take()
            .expect("active report was just completed");
        let experts: Vec<ExpertState> = active.parts.into_values().flatten().collect();
        let distinct: BTreeSet<(u32, u32)> = experts
            .iter()
            .map(|expert| (expert.layer_id, expert.expert_id))
            .collect();
        if distinct.len() != experts.len() {
            return Err(FullStateError("full state report repeats an expert"));
        }
        Ok(Some(CompleteStateReport {
            placement_generation: active.placement_generation,
            report_sequence: active.report_sequence,
            full: true,
            experts,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::ek::{
        control::v2::{
            ExpertStateKind, ExpertStateUpdates, WorkerDevice, controller_weight_message,
        },
        worker::v2::ActivationDType,
    };
    use tokio::sync::Mutex;

    #[derive(Default)]
    struct FakeHooks {
        persisted: Mutex<Vec<BTreeSet<ExpertKey>>>,
    }

    #[async_trait]
    impl WeightControlHooks for FakeHooks {
        async fn subscribe(
            &self,
            _registration: &RegisterWorkerRequest,
        ) -> Result<TargetSubscription, Status> {
            unreachable!()
        }

        async fn unsubscribe(&self, _worker_id: &str, _subscription_id: u64) {}

        async fn persist_ready(
            &self,
            _worker_id: &str,
            ready: &BTreeSet<ExpertKey>,
        ) -> Result<(), Status> {
            self.persisted.lock().await.push(ready.clone());
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
                max_experts: 128,
            }),
            max_batch_tokens: 64,
            max_active_batches_per_device: 1,
            max_pending_batches_per_device: 1,
        }
    }

    fn ready(expert_id: u32) -> ExpertState {
        ExpertState {
            layer_id: 1,
            expert_id,
            state: ExpertStateKind::ExpertReady as i32,
            failure: None,
        }
    }

    #[test]
    fn full_state_assembler_accepts_reordered_parts() {
        let mut assembler = FullStateAssembler::new(128);
        assert!(
            assembler
                .add(FullExpertStatePart {
                    placement_generation: 3,
                    report_sequence: 9,
                    part_index: 1,
                    part_count: 2,
                    experts: vec![ready(2)],
                })
                .unwrap()
                .is_none()
        );
        let report = assembler
            .add(FullExpertStatePart {
                placement_generation: 3,
                report_sequence: 9,
                part_index: 0,
                part_count: 2,
                experts: vec![ready(1)],
            })
            .unwrap()
            .unwrap();
        assert_eq!(report.report_sequence, 9);
        assert_eq!(
            report
                .experts
                .iter()
                .map(|expert| expert.expert_id)
                .collect::<Vec<_>>(),
            [1, 2]
        );
    }

    #[test]
    fn full_state_assembler_rejects_conflicting_duplicates_and_excess_parts() {
        let mut conflicting = FullStateAssembler::new(128);
        let first = FullExpertStatePart {
            placement_generation: 1,
            report_sequence: 1,
            part_index: 0,
            part_count: 2,
            experts: vec![ready(1)],
        };
        assert!(conflicting.add(first.clone()).unwrap().is_none());
        let error = conflicting
            .add(FullExpertStatePart {
                experts: vec![ready(2)],
                ..first
            })
            .unwrap_err();
        assert_eq!(error.0, "full state part was repeated with different data");

        let mut assembler = FullStateAssembler::new(64);
        let part = FullExpertStatePart {
            placement_generation: 1,
            report_sequence: 1,
            part_index: 0,
            part_count: 1,
            experts: vec![ready(1)],
        };
        assembler.add(part.clone()).unwrap().unwrap();
        let error = assembler
            .add(FullExpertStatePart {
                part_count: 2,
                ..part
            })
            .unwrap_err();
        assert_eq!(error.0, "full state part is not bounded");
    }

    #[tokio::test]
    async fn session_sends_targets_persists_ready_state_and_acks_report() {
        let state = ControllerV2State::new(8);
        state.register(registration()).await.unwrap();
        let hooks = Arc::new(FakeHooks::default());
        let (_update_sender, updates) = mpsc::channel(1);
        let subscription = TargetSubscription {
            subscription_id: 1,
            initial: vec![TargetExpert {
                layer_id: 1,
                expert_id: 2,
                target_device: "cuda:0".to_owned(),
                peer_weight_endpoints: Vec::new(),
            }],
            updates,
        };
        let requests = tokio_stream::iter(vec![Ok(WorkerWeightMessage {
            message: Some(worker_weight_message::Message::StateUpdates(
                ExpertStateUpdates {
                    placement_generation: 1,
                    report_sequence: 1,
                    experts: vec![ready(2)],
                },
            )),
        })]);
        let (sender, mut responses) = mpsc::channel(8);
        WeightSession {
            state,
            hooks: hooks.clone(),
            worker_id: "worker-0".to_owned(),
            start_id: "start-0".to_owned(),
            max_experts: 128,
            responses: sender,
        }
        .run(requests, subscription)
        .await
        .unwrap();

        let target_message = responses.recv().await.unwrap().unwrap();
        let Some(controller_weight_message::Message::Targets(targets)) = target_message.message
        else {
            panic!("expected targets");
        };
        assert_eq!(targets.placement_generation, 1);
        assert_eq!(targets.experts[0].expert_id, 2);
        let ack_message = responses.recv().await.unwrap().unwrap();
        let Some(controller_weight_message::Message::StateAck(ack)) = ack_message.message else {
            panic!("expected state ack");
        };
        assert_eq!(ack.report_sequence, 1);
        assert_eq!(
            hooks.persisted.lock().await[0],
            BTreeSet::from([ExpertKey {
                layer_id: 1,
                expert_id: 2,
            }])
        );
    }

    #[tokio::test]
    async fn session_sends_target_removal_before_drain_authorization() {
        let state = ControllerV2State::new(8);
        state.register(registration()).await.unwrap();
        let lease = state.open_heartbeat("worker-0", "start-0").await.unwrap();
        state
            .heartbeat(
                "worker-0",
                "start-0",
                lease,
                1,
                crate::proto::ek::control::v2::WorkerRunState::WorkerRunning as i32,
            )
            .await
            .unwrap();
        let hooks = Arc::new(FakeHooks::default());
        let (update_sender, updates) = mpsc::channel(1);
        let subscription = TargetSubscription {
            subscription_id: 1,
            initial: vec![TargetExpert {
                layer_id: 1,
                expert_id: 2,
                target_device: "cuda:0".to_owned(),
                peer_weight_endpoints: Vec::new(),
            }],
            updates,
        };
        let (request_sender, requests) = mpsc::channel(2);
        let (response_sender, mut responses) = mpsc::channel(8);
        let session = tokio::spawn(
            WeightSession {
                state,
                hooks,
                worker_id: "worker-0".to_owned(),
                start_id: "start-0".to_owned(),
                max_experts: 128,
                responses: response_sender,
            }
            .run(ReceiverStream::new(requests), subscription),
        );

        let initial = responses.recv().await.unwrap().unwrap();
        assert!(matches!(
            initial.message,
            Some(controller_weight_message::Message::Targets(_))
        ));
        request_sender
            .send(Ok(WorkerWeightMessage {
                message: Some(worker_weight_message::Message::StateUpdates(
                    ExpertStateUpdates {
                        placement_generation: 1,
                        report_sequence: 1,
                        experts: vec![ready(2)],
                    },
                )),
            }))
            .await
            .unwrap();
        let ack = responses.recv().await.unwrap().unwrap();
        assert!(matches!(
            ack.message,
            Some(controller_weight_message::Message::StateAck(_))
        ));

        update_sender.send(Ok(Vec::new())).await.unwrap();
        let removal = responses.recv().await.unwrap().unwrap();
        let Some(controller_weight_message::Message::Targets(targets)) = removal.message else {
            panic!("expected target removal");
        };
        assert_eq!(targets.placement_generation, 2);
        assert!(targets.experts.is_empty());
        let authorization = responses.recv().await.unwrap().unwrap();
        let Some(controller_weight_message::Message::Drain(drain)) = authorization.message else {
            panic!("expected drain authorization");
        };
        assert_eq!(drain.placement_generation, 2);
        assert_eq!(drain.min_topology_version, 2);
        assert_eq!(drain.experts[0].expert_id, 2);

        request_sender
            .send(Ok(WorkerWeightMessage {
                message: Some(worker_weight_message::Message::DrainComplete(
                    crate::proto::ek::control::v2::DrainComplete {
                        drain_id: drain.drain_id,
                    },
                )),
            }))
            .await
            .unwrap();
        drop(request_sender);
        session.await.unwrap().unwrap();
        drop(update_sender);
    }
}
