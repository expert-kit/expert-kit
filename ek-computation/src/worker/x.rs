use gethostname::gethostname;
use opendal::Buffer;
use tonic::transport::Endpoint;

use std::sync::Arc;

use ek_base::error::EKResult;
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
    let ek_hostname = std::option_env!("EK_HOSTNAME");
    if let Some(e) = ek_hostname {
        return e.to_owned();
    }
    let hn = gethostname();
    hn.into_string().unwrap()
}

pub fn get_control_plan_addr() -> Endpoint {
    let addr = "http://[::1]:5001";
    Endpoint::from_static(addr)
}
