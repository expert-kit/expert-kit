use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{Arc, OnceLock},
    time,
};

use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
    utils::{Defers, PerfTimer},
};
use safetensors::SafeTensors;
use tch::{IndexOp, Tensor};
use tokio::{
    sync::{Mutex, mpsc},
    task::JoinHandle,
};
use tracing::{Instrument, Span, span};

use crate::{
    backend::{EkTensor, torch::TchTensor},
    controller::registry::{ExpertClient, ExpertId, ExpertIdRef, ShmqWorkerReq, ShmqWorkerResp},
    metrics::METRIC_CONTROLLER_INTRA_REQ,
    observability::{StageTimer, TraceLabels, inject_current_trace_context, next_expert_call_id},
    proto::ek::worker::v1::{self},
};

use super::registry::{GlobalWorkerRegistry, get_registry};

#[async_trait::async_trait]
pub trait Executor {
    async fn submit(
        &mut self,
        req: &v1::ForwardReq,
    ) -> EKResult<mpsc::Receiver<Arc<v1::ForwardResp>>>;

    async fn exec(&mut self) -> EKResult<()>;
}

type ReqId = u64;
type GlobalSeqId = u64;
type LocalSeqIdx = usize;

struct IngressMeta {
    tensor: Tensor,
    sender: mpsc::Sender<Arc<v1::ForwardResp>>,
    result: Vec<Vec<ExpertResult>>,
}

unsafe impl Sync for IngressMeta {}

#[derive(Clone)]
struct EgressMeta {
    req_id: ReqId,
    seq_gid: GlobalSeqId,
    expert_idx: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum ExpertError {
    NoWorker { expert_id: String, message: String },
    Timeout { expert_id: String, message: String },
    Transport { expert_id: String, message: String },
    Compute { expert_id: String, message: String },
}

impl fmt::Display for ExpertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpertError::NoWorker { expert_id, message } => {
                write!(f, "no worker for expert {expert_id}: {message}")
            }
            ExpertError::Timeout { expert_id, message } => {
                write!(f, "timeout for expert {expert_id}: {message}")
            }
            ExpertError::Transport { expert_id, message } => {
                write!(f, "transport error for expert {expert_id}: {message}")
            }
            ExpertError::Compute { expert_id, message } => {
                write!(f, "compute error for expert {expert_id}: {message}")
            }
        }
    }
}

enum ExpertResult {
    Pending,
    Running,
    Success(Tensor),
    Failed(ExpertError),
}

impl ExpertResult {
    fn is_terminal(&self) -> bool {
        matches!(self, ExpertResult::Success(_) | ExpertResult::Failed(_))
    }
}

enum ForwardResponse {
    Grpc(v1::ForwardResp),
    Shm(ShmqWorkerResp),
    Rdma(ShmqWorkerResp),
}

impl ForwardResponse {
    fn output_tensor(&self) -> &[u8] {
        match self {
            ForwardResponse::Grpc(resp) => &resp.output_tensor,
            ForwardResponse::Shm(resp) => resp.output_tensor(),
            ForwardResponse::Rdma(resp) => resp.output_tensor(),
        }
    }
}

#[derive(Debug, Clone)]
enum PendingResponse {
    Shm(ShmqWorkerResp),
    Rdma(ShmqWorkerResp),
}

impl PendingResponse {
    fn into_shm(self) -> Option<ShmqWorkerResp> {
        match self {
            PendingResponse::Shm(resp) => Some(resp),
            PendingResponse::Rdma(_) => None,
        }
    }

    fn into_rdma(self) -> Option<ShmqWorkerResp> {
        match self {
            PendingResponse::Shm(_) => None,
            PendingResponse::Rdma(resp) => Some(resp),
        }
    }
}

pub struct NaiveExecutor {
    pending_egress: BTreeMap<ExpertId, Vec<EgressMeta>>,
    pending_ingress: BTreeMap<ReqId, IngressMeta>,
    pending_resp: Arc<Mutex<HashMap<usize, PendingResponse>>>,

    seq_mapping: BTreeMap<GlobalSeqId, (ReqId, LocalSeqIdx)>,
    seq_gid_cursor: u64,
    req_id_cursor: u64,
    registry: GlobalWorkerRegistry,
}

impl fmt::Debug for NaiveExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NaiveExecutor").finish()
    }
}

#[async_trait::async_trait]
impl Executor for NaiveExecutor {
    async fn submit(
        &mut self,
        req: &v1::ForwardReq,
    ) -> EKResult<mpsc::Receiver<Arc<v1::ForwardResp>>> {
        self.inner_submit(req).await
    }

    async fn exec(&mut self) -> EKResult<()> {
        self.inner_execute().await
    }
}

impl NaiveExecutor {
    async fn inner_submit(
        &mut self,
        req: &v1::ForwardReq,
    ) -> EKResult<mpsc::Receiver<Arc<v1::ForwardResp>>> {
        let (sender, receiver) = mpsc::channel(1);
        log::debug!("submit request, seq_len {:?}", req.sequences.len());
        let span = span!(
            tracing::Level::INFO,
            "naive_executor_submit",
            seq_len = req.sequences.len(),
            instance_id = req.instance_id.as_str()
        );
        let _enter = span.enter();

        let inp_safetensor = SafeTensors::deserialize(&req.tensor)?;
        let inp_view = inp_safetensor.tensor("data")?;
        let inp_tensor = TchTensor::from(&inp_view);
        let mut result = vec![];

        for i in &req.sequences {
            let mut experts = Vec::new();
            for _ in &i.experts {
                experts.push(ExpertResult::Pending);
            }
            result.push(experts);
        }

        let meta = IngressMeta {
            tensor: inp_tensor.inner(),
            sender,
            result,
        };

        self.req_id_cursor += 1;

        self.pending_ingress.insert(self.req_id_cursor, meta);
        self.break_down_to_egress(req, self.req_id_cursor);

        Ok(receiver)
    }

    fn assemble_seq_tensors(&self, gids: Vec<GlobalSeqId>) -> EKResult<Tensor> {
        let mut tensors = vec![];
        for gid in gids {
            let (rid, lid) = self
                .seq_mapping
                .get(&gid)
                .ok_or(EKError::NotFound("seq not found".into()))?;
            let ingress_meta = self
                .pending_ingress
                .get(rid)
                .ok_or(EKError::NotFound("req tensor not found".into()))?;
            let hidden = ingress_meta.tensor.i(*lid as i64);
            tensors.push(hidden);
        }
        let out = Tensor::stack(&tensors, 0);
        log::debug!(
            "assemble seq tensor, vec_len={} shape={:?}",
            tensors.len(),
            out.size()
        );
        Ok(out)
    }

    fn mark_egress_running(&mut self, egress_meta: &[EgressMeta]) -> EKResult<()> {
        for meta in egress_meta {
            let (_, lid) = self
                .seq_mapping
                .get(&meta.seq_gid)
                .ok_or(EKError::NotFound("no seq mapping".into()))?;
            let ingress_meta = self
                .pending_ingress
                .get_mut(&meta.req_id)
                .ok_or(EKError::NotFound("no ingress req found".into()))?;
            ingress_meta.result[*lid][meta.expert_idx] = ExpertResult::Running;
        }
        Ok(())
    }

    fn mark_egress_failed(&mut self, egress_meta: &[EgressMeta], err: ExpertError) -> EKResult<()> {
        for meta in egress_meta {
            let (_, lid) = self
                .seq_mapping
                .get(&meta.seq_gid)
                .ok_or(EKError::NotFound("no seq mapping".into()))?;
            let ingress_meta = self
                .pending_ingress
                .get_mut(&meta.req_id)
                .ok_or(EKError::NotFound("no ingress req found".into()))?;
            ingress_meta.result[*lid][meta.expert_idx] = ExpertResult::Failed(err.clone());
        }
        Ok(())
    }

    fn cleanup_egress_requests(&mut self, egress_meta: &[EgressMeta]) {
        let failed_req_ids = egress_meta
            .iter()
            .map(|meta| meta.req_id)
            .collect::<Vec<_>>();
        self.pending_ingress
            .retain(|req_id, _| !failed_req_ids.contains(req_id));
        self.seq_mapping
            .retain(|_, (req_id, _)| !failed_req_ids.contains(req_id));
        for metas in self.pending_egress.values_mut() {
            metas.retain(|meta| !failed_req_ids.contains(&meta.req_id));
        }
        self.pending_egress.retain(|_, metas| !metas.is_empty());
    }

    pub async fn inner_execute(&mut self) -> EKResult<()> {
        let mut tit = PerfTimer::new("inner_execute");
        let mut handles: Vec<JoinHandle<Result<ForwardResponse, ExpertError>>> = vec![];
        let mut chips: Vec<(ExpertId, Vec<EgressMeta>, StageTimer, Span)> = vec![];
        let settings = get_ek_settings();

        while let Some((expert_id, egress_meta)) = self.pending_egress.pop_first() {
            let expert_id: ExpertIdRef = expert_id.as_ref();
            let expert_call_id = next_expert_call_id();
            let num_tokens = egress_meta.len();
            let expert_timer = StageTimer::start(
                TraceLabels::new("controller", "controller.expert_call")
                    .path("fallback_controller")
                    .expert_call_id(expert_call_id)
                    .expert_id(expert_id)
                    .num_tokens(num_tokens),
            );
            let expert_span = expert_timer.span();

            // Retry selecting a worker — the expert may be in recovery
            // (recently assigned to a surviving worker but not yet loaded).
            // Wait up to ~62 s for it to become available before giving up.
            const MAX_ATTEMPTS: u32 = 6;
            let client = {
                let schedule_timer = {
                    let _entered = expert_span.enter();
                    StageTimer::start(
                        TraceLabels::new("controller", "controller.schedule")
                            .path("fallback_controller")
                            .expert_call_id(expert_call_id)
                            .expert_id(expert_id)
                            .num_tokens(num_tokens),
                    )
                };
                let schedule_span = schedule_timer.span();
                let selected = async {
                    let mut last_err = None;
                    let mut found = None;
                    for attempt in 0..MAX_ATTEMPTS {
                        match self.registry.lock().await.select(expert_id).await {
                            Ok(c) => {
                                if attempt > 0 {
                                    log::info!(
                                        "controller executor: worker found for {expert_id} on attempt {}",
                                        attempt + 1
                                    );
                                }
                                found = Some(c);
                                break;
                            }
                            Err(e) => {
                                let backoff =
                                    std::time::Duration::from_secs(1u64 << attempt.min(4));
                                log::info!(
                                    "controller executor: no worker for {expert_id} \
                                     (attempt {}/{MAX_ATTEMPTS}), retrying in {:?}: {e}",
                                    attempt + 1,
                                    backoff
                                );
                                last_err = Some(e);
                                tokio::time::sleep(backoff).await;
                            }
                        }
                    }
                    match found {
                        Some(c) => Ok(c),
                        None => {
                            let err = ExpertError::NoWorker {
                                expert_id: expert_id.to_owned(),
                                message: format!(
                                    "no worker after {MAX_ATTEMPTS} attempts: {last_err:?}"
                                ),
                            };
                            log::error!("controller executor: {err}");
                            self.mark_egress_failed(&egress_meta, err.clone())?;
                            for handle in handles.iter() {
                                handle.abort();
                            }
                            self.cleanup_egress_requests(&egress_meta);
                            Err(EKError::NotFound(err.to_string()))
                        }
                    }
                };
                let selected = selected.instrument(schedule_span).await;
                drop(schedule_timer);
                selected?
            };
            self.mark_egress_running(&egress_meta)?;
            chips.push((
                expert_id.to_owned(),
                egress_meta.to_owned(),
                expert_timer,
                expert_span.clone(),
            ));

            let seq_gids = egress_meta
                .iter()
                .map(|e| e.seq_gid)
                .collect::<Vec<GlobalSeqId>>();

            let egress_tensor = {
                let timer = {
                    let _entered = expert_span.enter();
                    StageTimer::start(
                        TraceLabels::new("controller", "controller.assemble")
                            .path("fallback_controller")
                            .expert_call_id(expert_call_id)
                            .expert_id(expert_id)
                            .num_tokens(num_tokens),
                    )
                };
                let span = timer.span();
                let _entered = span.enter();
                let tensor = self.assemble_seq_tensors(seq_gids)?;
                drop(timer);
                tensor
            };
            log::debug!("egress tensor shape={:?}", egress_tensor.size());
            let serialized_tensor = {
                let timer = {
                    let _entered = expert_span.enter();
                    StageTimer::start(
                        TraceLabels::new("controller", "controller.serialize")
                            .path("fallback_controller")
                            .expert_call_id(expert_call_id)
                            .expert_id(expert_id)
                            .num_tokens(num_tokens),
                    )
                };
                let span = timer.span();
                let _entered = span.enter();
                let tensor = TchTensor::from(egress_tensor).serialize();
                drop(timer);
                tensor
            };
            let seqs = egress_meta
                .iter()
                .map(|_e| v1::forward_req::SequenceInfo {
                    experts: vec![expert_id.to_owned()],
                })
                .collect::<Vec<_>>();

            let pending_resp = self.pending_resp.clone();
            let expert_id = expert_id.to_owned();
            match client {
                ExpertClient::Grpc(grpc_channel) => {
                    let mut cli =
                        v1::computation_service_client::ComputationServiceClient::new(grpc_channel)
                            .max_decoding_message_size(1024 * 1024 * 1024)
                            .max_encoding_message_size(1024 * 1024 * 1024);

                    let f = tokio::spawn(
                        async move {
                            let req = v1::ForwardReq {
                                // TODO: hardcode instance id.
                                instance_id: "0".into(),
                                tensor: serialized_tensor,
                                sequences: seqs,
                            };
                            let send_timer = StageTimer::start(
                                TraceLabels::new("controller", "controller.send_worker")
                                    .path("fallback_controller")
                                    .expert_call_id(expert_call_id)
                                    .expert_id(expert_id.as_str())
                                    .num_tokens(num_tokens),
                            );
                            let send_span = send_timer.span();

                            let start = time::Instant::now();
                            let _d = Defers::defer(Box::new(move || {
                                let elapsed = start.elapsed();
                                // TODO: hardcode metric name
                                METRIC_CONTROLLER_INTRA_REQ
                                    .with_label_values(&[settings.inference.model_name.as_str()])
                                    .observe(elapsed.as_micros() as f64);
                            }));
                            let result = async {
                                let mut req = tonic::Request::new(req);
                                inject_current_trace_context(req.metadata_mut());
                                cli.forward(req).await
                            }
                            .instrument(send_span)
                            .await
                            .map(|resp| ForwardResponse::Grpc(resp.into_inner()))
                            .map_err(|e| {
                                let message = format!("gRPC forward error: {e}");
                                if e.code() == tonic::Code::Internal {
                                    ExpertError::Compute { expert_id, message }
                                } else {
                                    ExpertError::Transport { expert_id, message }
                                }
                            });
                            drop(send_timer);
                            result
                        }
                        .instrument(expert_span.clone()),
                    );
                    handles.push(f);
                }
                ExpertClient::Shm((send_channel, recv_channel)) => {
                    let fu = async move {
                        let req = ShmqWorkerReq::new(expert_id.as_ref(), &serialized_tensor);

                        let start = time::Instant::now();
                        let _d = Defers::defer(Box::new(move || {
                            let elapsed = start.elapsed();
                            // TODO: hardcode metric name
                            METRIC_CONTROLLER_INTRA_REQ
                                .with_label_values(&[settings.inference.model_name.as_str()])
                                .observe(elapsed.as_micros() as f64);
                        }));

                        while send_channel.lock().await.send(&req).is_err() {
                            log::warn!("failed to send request to expert {expert_id}");
                            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
                        }

                        log::debug!(
                            "request sent for expert {}, waiting for response",
                            expert_id
                        );
                        let resp = loop {
                            if let Some(pending_resp) = pending_resp.lock().await.remove(&req.id())
                                && let Some(resp) = pending_resp.into_shm()
                            {
                                break resp;
                            }
                            match recv_channel.lock().await.recv() {
                                Ok(resp) => {
                                    if resp.id() != req.id() {
                                        log::debug!(
                                            "received response for expert {} but id not correct",
                                            expert_id
                                        );
                                        pending_resp
                                            .lock()
                                            .await
                                            .insert(resp.id(), PendingResponse::Shm(resp));
                                        tokio::task::yield_now().await;
                                        continue;
                                    }
                                    break resp;
                                }
                                Err(_) => {
                                    tokio::task::yield_now().await;
                                }
                            }
                        };
                        Ok(ForwardResponse::Shm(resp))
                    }
                    .in_current_span();
                    handles.push(tokio::spawn(fu));
                }
                ExpertClient::Rdma((send_channel, recv_channel)) => {
                    let fu = async move {
                        let req = ShmqWorkerReq::new(expert_id.as_ref(), &serialized_tensor);

                        let start = time::Instant::now();
                        let _d = Defers::defer(Box::new(move || {
                            let elapsed = start.elapsed();
                            METRIC_CONTROLLER_INTRA_REQ
                                .with_label_values(&[settings.inference.model_name.as_str()])
                                .observe(elapsed.as_micros() as f64);
                        }));

                        // Send request via RDMA
                        loop {
                           match send_channel.lock().await.send(&req) {
                                Ok(_) => break,
                                Err(e) => {
                                    log::warn!("failed to send RDMA request to expert {expert_id}: {e}");
                                    tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
                                }
                            }
                        }

                        log::debug!(
                            "RDMA request sent for expert {}, waiting for response",
                            expert_id
                        );

                        // Wait for response via RDMA
                        let resp = loop {
                            if let Some(pending_resp) = pending_resp.lock().await.remove(&req.id())
                                && let Some(resp) = pending_resp.into_rdma() {
                                    break resp;
                                }
                            match recv_channel.lock().await.recv() {
                                Ok(resp) => {
                                    if resp.id() != req.id() {
                                        log::debug!(
                                            "received RDMA response for expert {} but id not correct",
                                            expert_id
                                        );
                                        pending_resp.lock().await.insert(resp.id(), PendingResponse::Rdma(resp));
                                        tokio::task::yield_now().await;
                                        continue;
                                    }
                                    break resp;
                                }
                                Err(_) => {
                                    tokio::task::yield_now().await;
                                }
                            }
                        };
                        Ok(ForwardResponse::Rdma(resp))
                    }
                    .in_current_span();
                    handles.push(tokio::spawn(fu));
                }
            }
        }

        tit.stop("egress_req_sent");

        for (egress_idx, handle) in handles.into_iter().enumerate() {
            let (expert_id_for_chip, egress_meta_for_chip, _expert_timer, expert_span) =
                &chips[egress_idx];
            let egress = (expert_id_for_chip.clone(), egress_meta_for_chip.clone());
            let res = match handle.await {
                Ok(Ok(res)) => res,
                Ok(Err(err)) => {
                    self.mark_egress_failed(&egress.1, err.clone())?;
                    self.cleanup_egress_requests(&egress.1);
                    return Err(EKError::RuntimeError(err.to_string()));
                }
                Err(e) => {
                    let err = ExpertError::Transport {
                        expert_id: egress.0.clone(),
                        message: format!("worker task join error: {e}"),
                    };
                    self.mark_egress_failed(&egress.1, err.clone())?;
                    self.cleanup_egress_requests(&egress.1);
                    return Err(EKError::RuntimeError(err.to_string()));
                }
            };
            let res_safetensor = {
                let timer = {
                    let _entered = expert_span.enter();
                    StageTimer::start(
                        TraceLabels::new("controller", "controller.deserialize")
                            .path("fallback_controller")
                            .expert_id(egress.0.as_str())
                            .num_tokens(egress.1.len()),
                    )
                };
                let span = timer.span();
                let _entered = span.enter();
                let parsed = SafeTensors::deserialize(res.output_tensor());
                drop(timer);
                match parsed {
                    Ok(res) => res,
                    Err(e) => {
                        let err = ExpertError::Compute {
                            expert_id: egress.0.clone(),
                            message: format!("invalid output tensor: {e}"),
                        };
                        self.mark_egress_failed(&egress.1, err.clone())?;
                        self.cleanup_egress_requests(&egress.1);
                        return Err(EKError::RuntimeError(err.to_string()));
                    }
                }
            };
            // TODO: hardcode safe tensor name
            let view = match res_safetensor.tensor("data") {
                Ok(view) => view,
                Err(e) => {
                    let err = ExpertError::Compute {
                        expert_id: egress.0.clone(),
                        message: format!("missing output tensor data: {e}"),
                    };
                    self.mark_egress_failed(&egress.1, err.clone())?;
                    self.cleanup_egress_requests(&egress.1);
                    return Err(EKError::RuntimeError(err.to_string()));
                }
            };
            let res_tensor = {
                let timer = {
                    let _entered = expert_span.enter();
                    StageTimer::start(
                        TraceLabels::new("controller", "controller.merge")
                            .path("fallback_controller")
                            .expert_id(egress.0.as_str())
                            .num_tokens(egress.1.len()),
                    )
                };
                let span = timer.span();
                let _entered = span.enter();
                let tensor = TchTensor::from(&view).inner();
                drop(timer);
                tensor
            };

            log::debug!("received tensor shape={:?}", res_tensor.size());
            for (seq_idx, egress_meta) in egress.1.iter().enumerate() {
                let id_mapping = self
                    .seq_mapping
                    .get(&egress_meta.seq_gid)
                    .ok_or(EKError::NotFound("no seq mapping".into()))?;
                assert!(id_mapping.0 == egress_meta.req_id);
                let lid = id_mapping.1;
                let meta = self
                    .pending_ingress
                    .get_mut(&egress_meta.req_id)
                    .ok_or(EKError::NotFound("no ingress req found".into()))?;
                let seq_completion = &mut meta.result[lid];
                seq_completion[egress_meta.expert_idx] =
                    ExpertResult::Success(res_tensor.i(seq_idx as i64));
            }
        }

        tit.stop("remote resp joined");
        self.output().await?;
        tit.stop("output generated");

        Ok(())
    }

    async fn output(&mut self) -> EKResult<()> {
        let mut removed = vec![];
        let mut failed = None;
        for (req_id, meta) in self.pending_ingress.iter() {
            let completed = meta
                .result
                .iter()
                .all(|x| x.iter().all(ExpertResult::is_terminal));
            if !completed {
                continue;
            }
            if let Some(err) = meta.result.iter().flatten().find_map(|v| match v {
                ExpertResult::Failed(err) => Some(err.clone()),
                _ => None,
            }) {
                failed = Some(EKError::RuntimeError(err.to_string()));
                removed.push(*req_id);
                continue;
            }
            let res_tensors = meta
                .result
                .iter()
                .map(|x| {
                    let must_tensor = x
                        .iter()
                        .map(|x| match x {
                            ExpertResult::Success(tensor) => tensor,
                            _ => unreachable!(
                                "completed request without failed slots must be all success"
                            ),
                        })
                        .collect::<Vec<_>>();
                    Tensor::stack(&must_tensor, 0)
                })
                .collect::<Vec<_>>();

            let output_tensor = Tensor::stack(&res_tensors, 0);
            log::debug!("output tensor shape: {:?}", output_tensor.size());
            let serialized_tensor = TchTensor::from(output_tensor).serialize();

            let resp = v1::ForwardResp {
                output_tensor: serialized_tensor,
            };

            let send_res = meta
                .sender
                .send_timeout(Arc::new(resp), std::time::Duration::from_secs(5))
                .await;
            if let Err(e) = send_res {
                log::error!("send forward response  error: {e}");
            }
            removed.push(*req_id);
        }

        for rid in removed {
            self.pending_ingress.remove(&rid);
            let gids_to_remove = self
                .seq_mapping
                .iter()
                .filter(|x| x.1.0 == rid)
                .map(|x| *x.0)
                .collect::<Vec<_>>();
            for key in gids_to_remove {
                self.seq_mapping.remove(&key);
            }
        }
        if let Some(err) = failed {
            return Err(err);
        }
        Ok(())
    }

    fn break_down_to_egress(&mut self, req: &v1::ForwardReq, req_id: ReqId) {
        for (idx, seq) in req.sequences.iter().enumerate() {
            // update pending_seq
            let seq_gid = self.add_seq(req_id, idx as LocalSeqIdx);
            // update pending_req
            for (idx, expert) in seq.experts.iter().enumerate() {
                self.pending_egress
                    .entry(expert.clone())
                    .or_default()
                    .push(EgressMeta {
                        req_id,
                        seq_gid,
                        expert_idx: idx,
                    });
            }
        }
    }
    fn add_seq(&mut self, rid: ReqId, seq_lid: LocalSeqIdx) -> GlobalSeqId {
        self.seq_gid_cursor += 1;
        self.seq_mapping.insert(self.seq_gid_cursor, (rid, seq_lid));
        self.seq_gid_cursor
    }
}

impl Default for NaiveExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl NaiveExecutor {
    pub fn new() -> Self {
        Self {
            pending_egress: BTreeMap::new(),
            pending_ingress: BTreeMap::new(),
            seq_mapping: BTreeMap::new(),
            seq_gid_cursor: 0,
            req_id_cursor: 0,
            registry: get_registry(),
            pending_resp: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

pub fn get_executor() -> Arc<Mutex<dyn Executor + Send>> {
    static INSTANCE: OnceLock<Arc<Mutex<dyn Executor + Send>>> = OnceLock::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = NaiveExecutor::new();
        Arc::new(Mutex::new(inner))
    });
    (res.clone()) as _
}
