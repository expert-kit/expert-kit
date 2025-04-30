use gethostname::gethostname;
use opendal::{Operator, services::S3};
use tonic::transport::Endpoint;

use std::sync::Arc;

use ek_base::error::EKResult;
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
    let mut guard = tensor_db.write().await;
    let safe_tensor = guard.load(&meta.id).await?;
    let backend = ExpertBackend::build(instance, &safe_tensor).await?;
    let mut guard = expert_db.write().await;
    guard.insert(&meta.id, backend).await?;
    Ok(())
}

pub fn get_s3_dal_operator() -> opendal::Operator {
    static INSTANCE: OnceCell<opendal::Operator> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let builder = S3::default()
            .access_key_id("my_access_key")
            .secret_access_key("my_secret_key")
            .endpoint("my_endpoint")
            .region("my_region");

        Operator::new(builder).unwrap().finish()
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
    let addr = "[::1]:50051";
    Endpoint::from_static(addr)
}
