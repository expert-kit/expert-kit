use tonic::transport::Endpoint;

use std::{str::FromStr, sync::Arc};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::{safetensor::ExpertKey, weight_manager::LocalWeightManager};
use tokio::sync::RwLock;

use crate::ffn::ExpertBackend;

use super::manager::ExpertDB;

/// Load expert task - fetches weight bytes from the LocalWeightManager,
/// builds the ExpertBackend, and inserts it into the shared ExpertDB.
///
/// Zero-copy: `wm.get_expert()` returns an `Arc<Bytes>` whose refcount is
/// incremented atomically. The bytes are borrowed by SafeTensors for the
/// duration of `ExpertBackend::build`, then the Arc is dropped here while
/// the WM retains its own reference for future callers.
pub async fn load_expert_task(
    weight_manager: Arc<LocalWeightManager>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    instance: crate::x::EKInstance,
    expert_key: &ExpertKey,
) -> EKResult<()> {
    let expert_str_key = expert_key.as_object_key();

    // Mark expert as loading in shared database
    {
        let mut wg = expert_db.write().await;
        wg.mark_loading(&expert_str_key)?;
    }

    // Fetch bytes and build backend within a scoped block so that
    // `bytes` (and thus `st`) are dropped before we take the expert_db write lock.
    // On any error, unmark_loading so the expert can be retried on the next update.
    let backend = match async {
        let bytes = weight_manager.get_expert(expert_key).await?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        ExpertBackend::build(instance, &st).await
        // `bytes` and `st` are dropped here
    }
    .await
    {
        Ok(b) => b,
        Err(e) => {
            let mut wg = expert_db.write().await;
            wg.unmark_loading(&expert_str_key);
            return Err(e);
        }
    };

    // Insert loaded expert into shared database
    let mut edb_wg = expert_db.write().await;
    edb_wg.insert(&expert_str_key, backend).await?;

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
