use gethostname::gethostname;
use opendal::Buffer;
use tonic::transport::Endpoint;

use std::{str::FromStr, sync::Arc};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::SafeTensorDB;
use tokio::sync::RwLock;

use crate::{ffn::ExpertBackend, proto::ek, x};

use super::manager::ExpertDB;

pub async fn load_expert_task(
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    instance: x::EKInstance,
    meta: ek::object::v1::Metadata,
) -> EKResult<()> {
    {
        let read_guard = expert_db.read().await;
        if read_guard.has(&meta.id) {
            log::info!("expert {} already loaded or is loading", meta.id);
            return Ok(());
        }
    }
    {
        let mut wg = expert_db.write().await;
        wg.lock(&meta.id)?;
    }

    let buf: Buffer;
    {
        let rg = tensor_db.read().await;
        buf = rg.load(&meta.id).await?;
    }

    {
        let mut tdb_wg = tensor_db.write().await;
        tdb_wg.save(&meta.id, buf)?;
    }

    {
        let rg = tensor_db.read().await;
        let st = rg.as_safetensor(&meta.id)?;
        let backend = ExpertBackend::build(instance, &st).await?;
        let mut edb_wg = expert_db.write().await;
        edb_wg.insert(&meta.id, backend).await?;
        edb_wg.unlock(&meta.id);
    }

    Ok(())
}

pub fn get_hostname() -> String {
    let settings = get_ek_settings();
    let ek_worker_id = settings.worker.id.clone();
    if let Some(wid) = ek_worker_id {
        return wid;
    }
    let hn = gethostname();
    hn.into_string().unwrap()
}

pub fn get_controller_addr() -> Endpoint {
    let settings = get_ek_settings();
    let addr = format!(
        "http://{}:{}",
        settings.controller.broadcast.host, settings.controller.broadcast.port
    );
    Endpoint::from_str(addr.as_str()).unwrap()
}
