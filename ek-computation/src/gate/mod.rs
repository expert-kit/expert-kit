use crate::{
    ffn::{EkTensor, ExpertBackend, expert_torch::TchTensor},
    proto::ek,
};
use ek_base::error::EKResult;
use std::collections::BTreeMap;

use crate::x;
use tokio::sync::RwLock;

pub struct EKInstanceGate {
    experts: RwLock<BTreeMap<String, Box<ExpertBackend>>>,
    tensor_db: ek_db::safetensor::SafeTensorDB,
    instance: x::EKInstance,
}

impl Default for EKInstanceGate {
    fn default() -> Self {
        Self::new()
    }
}

impl EKInstanceGate {
    pub fn new() -> Self {
        EKInstanceGate {
            experts: RwLock::new(BTreeMap::new()),
            tensor_db: ek_db::safetensor::SafeTensorDB::new(),
            instance: x::EKInstance::default(),
        }
    }
    pub async fn create_expert(&mut self, meta: ek::object::v1::Metadata) -> EKResult<()> {
        let safe_tensor = self.tensor_db.load(&meta.id).await?;
        let backend = ExpertBackend::build(self.instance, &safe_tensor).await?;
        let mut exps = self.experts.write().await;
        exps.insert(meta.id, Box::new(backend));
        Ok(())
    }

    pub async fn forward(
        &self,
        req: ek::worker::v1::ForwardReq,
    ) -> EKResult<ek::worker::v1::ForwardResp> {
        let lg = self.experts.read().await;
        let dim = 7168;

        let input_tensor = req.tensor;
        let mut output = vec![];
        for (seq_idx, inp) in req.sequences.iter().enumerate() {
            let seq_input = &input_tensor.as_slice()[seq_idx * dim..(seq_idx + 1) + dim];
            for exp_id in &inp.experts {
                let exp = lg
                    .get(exp_id)
                    .ok_or_else(|| ek_base::error::EKError::ExpertNotFound(exp_id.clone()))?;
                // TODO: batching here
                let res = exp.forward(seq_input)?;
                output.push((seq_idx, exp_id, res));
            }
        }
        output.sort_by(|a, b| a.0.cmp(&b.0));
        let tensors = output.into_iter().map(|x| x.2).collect::<Vec<TchTensor>>();
        let output_tensor = tch::Tensor::cat(&tensors, 0);
        let output_bytes = TchTensor::from(output_tensor).serialize();
        let resp = ek::worker::v1::ForwardResp {
            output_tensor: output_bytes,
        };
        Ok(resp)
    }
}
