use std::sync::Arc;

use ek_base::error::EKResult;
use ek_db::safetensor::SafeTensorDB;
use tokio::sync::{Mutex, RwLock};

use crate::{ffn::ExpertBackend, proto::ek, x};

use super::manager::ExpertDB;

async fn load_expert(
    tensor_db: Arc<Mutex<SafeTensorDB>>,
    expert_db: Arc<RwLock<ExpertDB>>,
    instance: x::EKInstance,
    meta: ek::object::v1::Metadata,
) -> EKResult<()> {
    let mut guard = tensor_db.lock().await;
    let safe_tensor = guard.load(&meta.id).await?;
    let backend = ExpertBackend::build(instance, &safe_tensor).await?;
    let mut guard = expert_db.write().await;
    guard.insert(&meta.id, backend).await?;
    Ok(())
}
