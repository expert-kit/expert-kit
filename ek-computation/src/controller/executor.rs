use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    sync::{Arc, OnceLock},
};

use ek_base::{
    error::{EKError, EKResult},
    utils::PerfTimer,
};
use safetensors::SafeTensors;
use tch::{IndexOp, Tensor};
use tokio::sync::{Mutex, mpsc};
use tracing::{instrument, span};

use crate::{
    backend::{EkTensor, torch::TchTensor},
    controller::registry::{ExpertId, ExpertIdRef, ShmqWorkerResp},
    proto::ek::worker::v1,
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
    result: Vec<Vec<Option<Tensor>>>,
}

unsafe impl Sync for IngressMeta {}

#[derive(Clone)]
struct EgressMeta {
    req_id: ReqId,
    seq_gid: GlobalSeqId,
    expert_idx: usize,
}

pub(crate) struct ForwardResponse {
    inner: ForwardResponseInner,
    duration: std::time::Duration,
}

pub(crate) enum ForwardResponseInner {
    Grpc(v1::ForwardResp),
    Shm(ShmqWorkerResp),
    Rdma(ShmqWorkerResp),
}

impl ForwardResponse {
    pub fn grpc(resp: v1::ForwardResp, duration: std::time::Duration) -> Self {
        Self {
            inner: ForwardResponseInner::Grpc(resp),
            duration,
        }
    }

    pub fn shm(resp: ShmqWorkerResp, duration: std::time::Duration) -> Self {
        Self {
            inner: ForwardResponseInner::Shm(resp),
            duration,
        }
    }

    pub fn rdma(resp: ShmqWorkerResp, duration: std::time::Duration) -> Self {
        Self {
            inner: ForwardResponseInner::Rdma(resp),
            duration,
        }
    }

    fn output_tensor(&self) -> &[u8] {
        match &self.inner {
            ForwardResponseInner::Grpc(resp) => &resp.output_tensor,
            ForwardResponseInner::Shm(resp) => resp.output_tensor(),
            ForwardResponseInner::Rdma(resp) => resp.output_tensor(),
        }
    }

    fn duration(&self) -> std::time::Duration {
        self.duration
    }
}

#[derive(Debug, Clone)]
pub(crate) enum PendingResponse {
    Shm(ShmqWorkerResp),
    Rdma(ShmqWorkerResp),
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
                experts.push(None);
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

    #[instrument]
    pub async fn inner_execute(&mut self) -> EKResult<()> {
        let mut tit = PerfTimer::new("inner_execute");
        let mut handles = vec![];
        let mut chips: Vec<(ExpertId, Vec<EgressMeta>)> = vec![];

        while let Some((expert_id, egress_meta)) = self.pending_egress.pop_first() {
            let expert_id: ExpertIdRef = expert_id.as_ref();
            let Ok(clients) = self.registry.lock().await.select_all(expert_id).await else {
                log::warn!("failed to select client for expert {expert_id}");
                continue;
            };
            chips.push((expert_id.to_owned(), egress_meta.to_owned()));

            let seq_gids = egress_meta
                .iter()
                .map(|e| e.seq_gid)
                .collect::<Vec<GlobalSeqId>>();

            let egress_tensor = self.assemble_seq_tensors(seq_gids)?;
            let egress_tensor_bat = egress_tensor.size()[0];
            log::debug!("egress tensor shape={:?}", egress_tensor.size());
            let seqs = egress_meta
                .iter()
                .map(|_e| v1::forward_req::SequenceInfo {
                    experts: vec![expert_id.to_owned()],
                })
                .collect::<Vec<_>>();

            let mut client_priorities = vec![];
            let mut client_priorities_sum = 0.0;
            for cli in &clients {
                let p = cli.get_priority().await;
                client_priorities.push(p);
                client_priorities_sum += p;
            }
            for p in &mut client_priorities {
                *p /= client_priorities_sum;
            }
            let mut splits = Vec::with_capacity(client_priorities.len());
            let mut acc = 0;
            for (i, &r) in client_priorities.iter().enumerate() {
                if i == client_priorities.len() - 1 {
                    splits.push(egress_tensor_bat - acc);
                } else {
                    let size = (r * egress_tensor_bat as f64).round() as i64;
                    splits.push(size);
                    acc += size;
                }
            }

            log::debug!("split egress_tensor with sizes={splits:?}");
            let tensors = egress_tensor.split_with_sizes(splits, 0);

            let mut handle_lst = vec![];
            let clients_count = clients.len();
            for (client, tensor) in clients.into_iter().zip(tensors) {
                if tensor.size()[0] == 0 {
                    continue;
                }
                let pending_resp = self.pending_resp.clone();
                let expert_id = expert_id.to_owned();
                log::debug!("forward tensor shape={:?}", tensor.size());
                let handle = client.clone().forward(
                    expert_id,
                    TchTensor::from(tensor).serialize(),
                    seqs.clone(),
                    pending_resp,
                );
                handle_lst.push((handle, client));
            }
            handles.push((clients_count, handle_lst));
        }

        tit.stop("egress_req_sent");

        'outer: for (egress_idx, handles) in handles.into_iter().enumerate() {
            let egress = &chips[egress_idx];
            let (clients_count, handles) = handles;
            let handles_count = handles.len();
            let mut clients = vec![];
            let mut res_duration = vec![];
            let mut res_tensors = vec![];
            for (handle, client) in handles {
                let Ok(res) = handle.await? else {
                    log::error!("failed to receive response for expert {}", egress.0);
                    continue 'outer;
                };
                let res_safetensor = SafeTensors::deserialize(res.output_tensor())?;
                // TODO: hardcode safe tensor name
                let view = res_safetensor.tensor("data")?;
                let res_tensor = TchTensor::from(&view).inner();
                if clients_count == handles_count {
                    let batch = res_tensor.size()[0];
                    clients.push(client);
                    res_duration.push((res.duration(), batch));
                }
                res_tensors.push(res_tensor);
            }
            if clients_count == handles_count {
                let per_batch_duration = res_duration
                    .iter()
                    .map(|&(d, b)| d.as_micros() / b as u128)
                    .collect::<Vec<_>>();
                let sum_of_duration = per_batch_duration.iter().sum::<u128>() as f64;
                let cli_priority_delta = per_batch_duration
                    .into_iter()
                    .map(|d| 1.0 - d as f64 / sum_of_duration);
                for (cli, delta) in clients.iter().zip(cli_priority_delta) {
                    log::debug!("expert client priority update: delta={delta}");
                    cli.update_priority(delta).await;
                }
            }

            let res_tensor = Tensor::cat(&res_tensors, 0);

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
                seq_completion[egress_meta.expert_idx] = Some(res_tensor.i(seq_idx as i64));
            }
        }

        tit.stop("remote resp joined");
        self.output().await;
        tit.stop("output generated");

        Ok(())
    }

    async fn output(&mut self) {
        let mut removed = vec![];
        for (req_id, meta) in self.pending_ingress.iter() {
            let completed = meta.result.iter().all(|x| x.iter().all(|v| v.is_some()));
            if !completed {
                continue;
            }
            let res_tensors = meta
                .result
                .iter()
                .map(|x| {
                    let must_tensor = x.iter().map(|x| x.as_ref().unwrap()).collect::<Vec<_>>();
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
    }

    #[instrument(skip(self, req))]
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
