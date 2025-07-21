use super::manager::{ExpertDB, ExpertDBSync, get_expert_db, get_expert_db_sync};
use crate::{
    backend::{EkTensor, torch::TchTensor},
    proto::ek,
};
use core::fmt;
use ek_base::error::EKResult;
use once_cell::sync::OnceCell;
use std::sync::Arc;
use tokio;
use tracing::instrument;

/// Async version of EKInstanceGate for non-compute operations
pub struct EKInstanceGateAsync {
    experts: Arc<tokio::sync::RwLock<dyn ExpertDB + Send + Sync>>,
}

impl fmt::Debug for EKInstanceGateAsync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EKInstanceGateAsync").finish()
    }
}

impl Default for EKInstanceGateAsync {
    fn default() -> Self {
        let edb = get_expert_db();
        EKInstanceGateAsync { experts: edb }
    }
}

/// Sync version of EKInstanceGate
pub struct EKInstanceGateSync {
    experts: Arc<std::sync::RwLock<dyn ExpertDBSync + Send + Sync>>,
}

impl fmt::Debug for EKInstanceGateSync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EKInstanceGateSync").finish()
    }
}

impl Default for EKInstanceGateSync {
    fn default() -> Self {
        let edb = get_expert_db_sync();
        EKInstanceGateSync { experts: edb }
    }
}

pub type GlobalEKInstanceGateAsync = Arc<tokio::sync::RwLock<EKInstanceGateAsync>>;
pub type GlobalEKInstanceGateSync = Arc<std::sync::RwLock<EKInstanceGateSync>>;

/// Get the global async instance gate (for state management, etc.)
pub fn get_instance_gate() -> GlobalEKInstanceGateAsync {
    static INSTANCE: OnceCell<GlobalEKInstanceGateAsync> = OnceCell::new();
    let inst = INSTANCE.get_or_init(|| {
        let inner = EKInstanceGateAsync::new();
        Arc::new(tokio::sync::RwLock::new(inner))
    });
    inst.clone()
}

/// Get the global sync instance gate (for compute operations)
pub fn get_instance_gate_sync() -> GlobalEKInstanceGateSync {
    static INSTANCE: OnceCell<GlobalEKInstanceGateSync> = OnceCell::new();
    let inst = INSTANCE.get_or_init(|| {
        let inner = EKInstanceGateSync::new();
        Arc::new(std::sync::RwLock::new(inner))
    });
    inst.clone()
}

impl EKInstanceGateAsync {
    pub fn new() -> Self {
        let edb = get_expert_db();
        EKInstanceGateAsync { experts: edb }
    }
    
    /// Get list of currently loaded experts (async version for state management)
    pub async fn current_experts(&self) -> EKResult<Vec<String>> {
        self.experts.read().await.keys().await
    }
}

impl EKInstanceGateSync {
    pub fn new() -> Self {
        let edb = get_expert_db_sync();
        EKInstanceGateSync { experts: edb }
    }

    /// Synchronous forward computation - optimized for compute-intensive tasks
    #[instrument(skip(self, req))]
    pub fn forward_sync(
        &self,
        req: ek::worker::v1::ForwardReq,
    ) -> EKResult<ek::worker::v1::ForwardResp> {
        tracing::debug!(
            "[L3 {:?}] gate.forward_sync() start: seq={}",
            req.sequences[0].experts[0],
            req.sequences.len(),
        );
        let start = std::time::Instant::now();

        let input_tensor = req.tensor;

        // Validate request structure
        assert!(!req.sequences.is_empty());
        assert!(req.sequences[0].experts.len() == 1);
        let exp_id = &req.sequences[0].experts[0];
        
        // Load expert synchronously from shared database
        let exp = {
            let experts_guard = self.experts.read().unwrap();
            experts_guard.load(exp_id)?
        };

        let now = std::time::Instant::now();
        tracing::debug!(
            "[L3 {:?}] exp_backend.forward_sync() started",
            exp_id,
        );

        // Perform synchronous computation
        let st = safetensors::SafeTensors::deserialize(&input_tensor).unwrap();
        let tv = st.tensor("data")?;
        let res = exp.forward(&tv)?;
        
        tracing::debug!(
            "[L3 {:?}] exp_backend.forward_sync() completed in {:?}",
            exp_id,
            now.elapsed()
        );

        // Serialize output
        let output_tensor = res.inner();
        let size = output_tensor.size();
        let kind = output_tensor.kind();
        let output_bytes = TchTensor::from(output_tensor).serialize();
        
        tracing::debug!(
            "output shape={:?} dtype={:?} bytes_len={}",
            size,
            kind,
            output_bytes.len()
        );
        
        let resp = ek::worker::v1::ForwardResp {
            output_tensor: output_bytes,
        };

        tracing::debug!(
            "[L3 {:?}] gate.forward_sync() end with {:?}",
            &exp_id,
            start.elapsed(),
        );

        Ok(resp)
    }

    /// Get list of currently loaded experts (sync version)
    #[expect(unused)]
    pub fn current_experts_sync(&self) -> EKResult<Vec<String>> {
        let experts_guard = self.experts.read().unwrap();
        experts_guard.keys()
    }
}