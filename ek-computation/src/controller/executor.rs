use std::{collections::BTreeMap, sync::Arc};

use ek_base::error::{EKError, EKResult};
use safetensors::SafeTensors;
use tch::{IndexOp, Tensor};
use tokio::sync::{Mutex, mpsc};

use crate::{
    ffn::{EkTensor, expert_torch::TchTensor},
    proto::ek::worker::v1,
};

use super::registry::ExpertRegistry;

#[async_trait::async_trait]
pub trait Executor {
    async fn submit(
        &mut self,
        req: &v1::ForwardReq,
    ) -> EKResult<mpsc::Receiver<Arc<v1::ForwardResp>>>;
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
        self.submit_inner(req)
    }
}

impl NaiveExecutor {
    fn submit_inner(
        &mut self,
        req: &v1::ForwardReq,
    ) -> EKResult<mpsc::Receiver<Arc<v1::ForwardResp>>> {
        let (sender, receiver) = mpsc::channel(1);

        let inp_safetensor = SafeTensors::deserialize(&req.tensor)?;
        let inp_view = inp_safetensor.tensor("data")?;
        let inp_tensor = TchTensor::from(&inp_view);
        let mut result = vec![];

        for i in &req.sequences {
            let mut experts = Vec::new();
            for j in &i.experts {
                experts.push(None);
            }
            result.push(experts);
        }

        let meta = IngressMeta {
            tensor: inp_tensor.inner(),
            sender: sender,
            result: result,
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
        let out = Tensor::cat(&tensors, 0);
        Ok(out)
    }

    async fn execute(&mut self) -> EKResult<()> {
        let mut handles = vec![];
        let mut chips = vec![];
        for egress_req in self.pending_egress.iter() {
            chips.push((egress_req.0.clone(), egress_req.1.clone()));
            let exp_id = egress_req.0.clone();
            let channel = self.registry.lock().await.select(exp_id).await?;

            let mut cli = v1::computation_service_client::ComputationServiceClient::new(channel);

            let seq_gids = egress_req
                .1
                .iter()
                .map(|e| e.seq_gid)
                .collect::<Vec<GlobalSeqId>>();

            let egress_tensor = self.assemble_seq_tensors(seq_gids)?;
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

        for (egress_idx, handle) in handles.into_iter().enumerate() {
            let egress = &chips[egress_idx];
            let res = handle.await??;
            let res_safetensor = SafeTensors::deserialize(&res.output_tensor)?;
            // TODO: hardcode safe tensor name
            let view = res_safetensor.tensor("output")?;
            let res_tensor = TchTensor::from(&view).inner();
            for (seq_idx, egress_meta) in egress.1.iter().enumerate() {
                let id_mapping = self
                    .seq_mapping
                    .get(&egress_meta.req_id)
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

        Ok(())
    }

    async fn output(&mut self, req_id: ReqId) {
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
                    Tensor::cat(&must_tensor, 0)
                })
                .collect::<Vec<_>>();

            let output_tensor = Tensor::cat(&res_tensors, 0);
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
                .map(|x| x.1.0)
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
                    .or_insert_with(Vec::new)
                    .push(EgressMeta {
                        req_id: req_id,
                        seq_gid: seq_gid,
                        expert_idx: idx,
                    });
            }
        }
    }
    fn add_seq(&mut self, rid: ReqId, seq_lid: LocalSeqIdx) -> GlobalSeqId {
        self.seq_gid_cursor += 1;
        self.seq_mapping.insert(self.seq_gid_cursor, (rid, seq_lid));
        return self.seq_gid_cursor;
    }
}
