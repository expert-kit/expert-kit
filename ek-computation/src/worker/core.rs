use super::manager::{ExpertDB, get_expert_db};
use crate::{
    ffn::{EkTensor, expert_torch::TchTensor},
    proto::ek,
};
use core::fmt;
use ek_base::error::EKResult;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

pub struct EKInstanceGate {
    experts: Arc<RwLock<dyn ExpertDB + Send + Sync>>,
}

impl fmt::Debug for EKInstanceGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EKInstanceGate").finish()
    }
}

impl Default for EKInstanceGate {
    fn default() -> Self {
        let edb = get_expert_db();
        EKInstanceGate { experts: edb }
    }
}

pub type GlobalEKInstanceGate = Arc<Mutex<EKInstanceGate>>;

pub fn get_instance_gate() -> GlobalEKInstanceGate {
    static INSTANCE: OnceCell<GlobalEKInstanceGate> = OnceCell::new();
    let inst = INSTANCE.get_or_init(|| {
        let inner = EKInstanceGate::new();
        Arc::new(Mutex::new(inner))
    });
    inst.clone()
}

impl EKInstanceGate {
    pub fn new() -> Self {
        let edb = get_expert_db();
        EKInstanceGate { experts: edb }
    }
    pub async fn current_experts(&self) -> EKResult<Vec<String>> {
        self.experts.read().await.keys().await
    }
    pub async fn forward(
        &self,
        req: ek::worker::v1::ForwardReq,
    ) -> EKResult<ek::worker::v1::ForwardResp> {
        // let lg = self.experts.load().await;
        let dim = 7168;

        let input_tensor = req.tensor;
        let mut output = vec![];
        for (seq_idx, inp) in req.sequences.iter().enumerate() {
            let seq_input = &input_tensor.as_slice()[seq_idx * dim..(seq_idx + 1) + dim];
            for exp_id in &inp.experts {
                let exp = self.experts.read().await.load(exp_id).await?;
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
