use gethostname::gethostname;
use opendal::{
    Buffer, Operator,
    services::{Fs, S3},
};
use tonic::transport::Endpoint;

use std::{str::FromStr, sync::Arc};

use ek_base::{config::get_config_key, error::EKResult};
use ek_db::safetensor::SafeTensorDB;
use once_cell::sync::OnceCell;
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

pub fn get_s3_dal_operator() -> opendal::Operator {
    static INSTANCE: OnceCell<opendal::Operator> = OnceCell::new();

    let res = INSTANCE.get_or_init(|| {
        let provider = get_config_key("storage_provider");

        match provider {
            "s3" => {
                log::info!("using s3 as weight store");
                let builder = S3::default()
                    .access_key_id(get_config_key("storage_s3_access_key_id"))
                    .secret_access_key(get_config_key("storage_s3_access_key_secret"))
                    .endpoint(get_config_key("storage_s3_endpoint"))
                    .region(get_config_key("storage_s3_region"));
                Operator::new(builder).unwrap().finish()
            }
            "fs" => {
                log::info!("using local file system as weight store");
                let path = get_config_key("storage_fs_path");
                let builder = Fs::default().root(path);
                Operator::new(builder).unwrap().finish()
            }
            _ => {
                panic!("unsupported storage provider");
            }
        }
    });
    res.clone()
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
    let ek_control_plan_addr = std::env::var("EK_CONTROL_PLAN_ADDR").ok();
    if let Some(e) = ek_control_plan_addr {
        let static_addr = Box::leak(e.into_boxed_str());
        return Endpoint::from_static(static_addr);
    }
    // Default to localhost
    let addr = "http://[::1]:5001";
    Endpoint::from_static(addr)
}
