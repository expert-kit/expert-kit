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

pub type GlobalEKInstanceGate = Arc<RwLock<EKInstanceGate>>;

pub fn get_instance_gate() -> GlobalEKInstanceGate {
    static INSTANCE: OnceCell<GlobalEKInstanceGate> = OnceCell::new();
    let inst = INSTANCE.get_or_init(|| {
        let inner = EKInstanceGate::new();
        Arc::new(RwLock::new(inner))
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
        let input_tensor = req.tensor;
        let st = safetensors::SafeTensors::deserialize(&input_tensor).unwrap();
        let tv = st.tensor("data")?;
        log::debug!("receive forward request, seq_len={}", req.sequences.len());
        assert!(!req.sequences.is_empty());
        assert!(req.sequences[0].experts.len() == 1);
        let exp_id = &req.sequences[0].experts[0];
        let exp = self.experts.read().await.load(exp_id).await?;
        let res = exp.forward(&tv)?;
        let output_tensor = res.inner();
        let size = output_tensor.size();
        let kind = output_tensor.kind();
        let output_bytes = TchTensor::from(output_tensor).serialize();
        log::debug!(
            "output shape={:?} dtype={:?} bytes_len={}",
            size,
            kind,
            output_bytes.len()
        );
        let resp = ek::worker::v1::ForwardResp {
            output_tensor: output_bytes,
        };
        Ok(resp)
    }
}
