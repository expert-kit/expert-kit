use std::{collections::BTreeMap, sync::Arc};

use ek_base::{
    error::{EKError, EKResult},
    utils::PerfTimer,
};
use once_cell::sync::OnceCell;
use safetensors::SafeTensors;
use tch::{IndexOp, Tensor};
use tokio::sync::{Mutex, mpsc};

use crate::{
    ffn::{EkTensor, expert_torch::TchTensor},
    proto::ek::worker::v1,
};

use super::registry::{ExpertRegistry, get_registry};

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

type ExpertId = String;

struct IngressMeta {
    tensor: Tensor,
    sender: mpsc::Sender<Arc<v1::ForwardResp>>,
    // tensor shape: [expert,hidden]
    result: Vec<Vec<Option<Tensor>>>,
}

unsafe impl Sync for IngressMeta {}

#[derive(Clone, Copy)]
struct EgressMeta {
    req_id: ReqId,
    seq_gid: GlobalSeqId,
    expert_idx: usize,
}

pub struct NaiveExecutor {
    pending_egress: BTreeMap<ExpertId, Vec<EgressMeta>>,
    pending_ingress: BTreeMap<ReqId, IngressMeta>,

    seq_mapping: BTreeMap<GlobalSeqId, (ReqId, LocalSeqIdx)>,
    seq_gid_cursor: u64,
    req_id_cursor: u64,
    registry: Arc<Mutex<dyn ExpertRegistry + Send + Sync>>,
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

    pub async fn inner_execute(&mut self) -> EKResult<()> {
        let mut tit = PerfTimer::new("inner_execute");
        let mut handles = vec![];
        let mut chips = vec![];
        {
            for egress_req in self.pending_egress.iter() {
                chips.push((egress_req.0.clone(), (egress_req.1.to_owned())));
                let exp_id = egress_req.0.clone();
                let channel = self.registry.lock().await.select(exp_id).await?;

                let mut cli =
                    v1::computation_service_client::ComputationServiceClient::new(channel)
                        .max_decoding_message_size(1024 * 1024 * 1024)
                        .max_encoding_message_size(1024 * 1024 * 1024);

                let seq_gids = egress_req
                    .1
                    .iter()
                    .map(|e| e.seq_gid)
                    .collect::<Vec<GlobalSeqId>>();

                let egress_tensor = self.assemble_seq_tensors(seq_gids)?;
                log::debug!("egress tensor shape={:?}", egress_tensor.size());
                let serialized_tensor = TchTensor::from(egress_tensor).serialize();
                let seqs = egress_req
                    .1
                    .iter()
                    .map(|_e| v1::forward_req::SequenceInfo {
                        experts: vec![egress_req.0.clone()],
                    })
                    .collect::<Vec<_>>();

                let f = tokio::spawn(async move {
                    let req = v1::ForwardReq {
                        instance_id: "0".into(),
                        tensor: serialized_tensor,
                        sequences: seqs,
                    };
                    cli.forward(req)
                        .await
                        .map(|resp| resp.into_inner())
                        .map_err(|e| {
                            log::error!("forward error: {}", e);
                            e
                        })
                });
                handles.push(f);
            }
        }

        for (egress_idx, _) in &chips {
            self.pending_egress.remove(egress_idx);
        }
        tit.stop("egress_req_sent");

        for (egress_idx, handle) in handles.into_iter().enumerate() {
            let egress = &chips[egress_idx];
            let res = handle.await??;
            let res_safetensor = SafeTensors::deserialize(&res.output_tensor)?;
            // TODO: hardcode safe tensor name
            let view = res_safetensor.tensor("data")?;
            let res_tensor = TchTensor::from(&view).inner();

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
                log::error!("send forward response  error: {}", e);
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
        }
    }
}

pub fn get_executor() -> Arc<Mutex<dyn Executor + Send>> {
    static INSTANCE: OnceCell<Arc<Mutex<dyn Executor + Send>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = NaiveExecutor::new();
        Arc::new(Mutex::new(inner))
    });
    (res.clone()) as _
}
