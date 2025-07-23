use tonic::transport::Endpoint;

use std::{str::FromStr, sync::Arc};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::{ExpertKey, SafeTensorDB};
use tokio::sync::RwLock;

use crate::{ffn::ExpertBackend, x};

use super::manager::ExpertDB;

/// Load expert task - handles expert loading and database insertion
/// This function updates the shared database that both async and sync gates use
pub async fn load_expert_task(
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    instance: x::EKInstance,
    expert_key: &ExpertKey,
) -> EKResult<()> {
    let expert_str_key = expert_key.as_object_key();

    // Mark expert as loading in shared database
    {
        let mut wg = expert_db.write().await;
        wg.mark_loading(&expert_str_key)?;
    }

    // Load tensor and build backend
    {
        let rg = tensor_db.read().await;
        let st = rg.load(expert_key).await?;
        let backend = ExpertBackend::build(instance, &st).await?;

        // Insert loaded expert into shared database (accessible by both async and sync gates)
        let mut edb_wg = expert_db.write().await;
        edb_wg.insert(&expert_str_key, backend).await?;
    }

    Ok(())
}

/// Get worker ID from settings
pub fn get_worker_id() -> String {
    let settings = get_ek_settings();
    settings.worker.id.clone()
}

/// Get controller endpoint from settings
pub fn get_controller_addr() -> Endpoint {
    let settings = get_ek_settings();
    let addr = format!(
        "http://{}:{}",
        settings.controller.broadcast, settings.controller.ports.intra
    );
    Endpoint::from_str(addr.as_str()).unwrap()
}
