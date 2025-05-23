use tonic::transport::Endpoint;

use std::{str::FromStr, sync::Arc};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::{ExpertKey, SafeTensorDB};
use tokio::sync::RwLock;

use crate::{ffn::ExpertBackend, x};

use super::manager::ExpertDB;

pub async fn load_expert_task(
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    instance: x::EKInstance,
    expert_key: &ExpertKey,
) -> EKResult<()> {
    let expert_str_key = expert_key.as_object_key();
    {
        let mut wg = expert_db.write().await;
        wg.mark_loading(&expert_str_key)?;
    }
    {
        let rg = tensor_db.read().await;
        let st = rg.load(expert_key).await?;
        let backend = ExpertBackend::build(instance, &st).await?;
        let mut edb_wg = expert_db.write().await;

        edb_wg.insert(&expert_str_key, backend).await?;
    }

    Ok(())
}

pub fn get_worker_id() -> String {
    let settings = get_ek_settings();

    settings.worker.id.clone()
}

pub fn get_controller_addr() -> Endpoint {
    let settings = get_ek_settings();
    let addr = format!(
        "http://{}:{}",
        settings.controller.broadcast, settings.controller.ports.intra
    );
    Endpoint::from_str(addr.as_str()).unwrap()
}
