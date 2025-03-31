mod x;
use super::ffn::Expert;
use crate::{ffn::EkTensor, proto::ek};
use ek_base::error::EKResult;
use std::collections::BTreeMap;

use tokio::sync::RwLock;

pub struct EKInstanceGate<T>
where
    T: for<'a> From<&'a [u8]>,
{
    experts: RwLock<BTreeMap<String, Box<dyn Expert<T> + 'static>>>,
}

impl<T> EKInstanceGate<T>
where
    T: for<'a> From<&'a [u8]> + EkTensor,
{
    pub fn new() -> Self {
        EKInstanceGate {
            experts: RwLock::new(BTreeMap::new()),
        }
    }
    async fn create_expert(&mut self, meta: ek::object::v1::Metadata) {}

    pub async fn add_expert(&mut self, name: String, expert: Box<dyn Expert<T> + 'static>) {
        let mut exps = self.experts.write().await;
        exps.insert(name, expert);
    }

    pub async fn forward(
        &self,
        req: ek::worker::v1::ForwardReq,
    ) -> EKResult<ek::worker::v1::ForwardResp> {
        let lg = self.experts.read().await;
        let dim = 7168;

        let seq_idx = 0;
        let input_tensor = req.tensor;
        let mut output = vec![];
        for inp in req.sequences.iter() {
            let seq_data = &input_tensor.as_slice()[seq_idx * dim..(seq_idx + 1) + dim];
            let tensor = T::from(seq_data);
            for exp_id in &inp.experts {
                let exp = lg
                    .get(exp_id)
                    .ok_or_else(|| ek_base::error::EKError::ExpertNotFound(exp_id.clone()))?;
                // TODO: batching here
                let result = exp.forward(&tensor);
                output.push(result);
            }
        }
        let output_tensor = T::cat(&output, 0);
        let output_bytes = output_tensor.serialize();
        let resp = ek::worker::v1::ForwardResp {
            output_tensor: output_bytes,
        };
        Ok(resp)
    }
}
