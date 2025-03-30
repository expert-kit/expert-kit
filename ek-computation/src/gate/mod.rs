use super::ffn::Expert;
use crate::proto::ek;
use ek_base::error::EKResult;
use std::collections::BTreeMap;
use tokio::sync::RwLock;

pub struct EKGate<T>
where
    T: From<Vec<u8>>,
{
    experts: RwLock<BTreeMap<String, Box<dyn Expert<T> + 'static>>>,
}

impl<T> EKGate<T>
where
    T: From<Vec<u8>>,
{
    pub fn new() -> Self {
        EKGate {
            experts: RwLock::new(BTreeMap::new()),
        }
    }
    async fn create_expert(&mut self, name: String) {
        

    }

    pub async fn add_expert(&mut self, name: String, expert: Box<dyn Expert<T> + 'static>) {
        let mut exps = self.experts.write().await;
        exps.insert(name, expert);
    }

    pub async fn forward(
        &self,
        req: ek::worker::v1::ForwardReq,
    ) -> EKResult<ek::worker::v1::ForwardResp> {
        let lg = self.experts.read().await;
        let exp = lg
            .get(&req.expert_id)
            .ok_or_else(|| ek_base::error::EKError::ExpertNotFound(req.expert_id))?;
        exp.forward(req.tensor.into());
        let resp = ek::worker::v1::ForwardResp {
            output_tensor: vec![],
        };
        Ok(resp)
    }
}
