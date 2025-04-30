use std::{collections::BTreeMap, sync::Arc};

use ek_base::error::{EKError, EKResult};
use once_cell::sync::OnceCell;
use tokio::sync::RwLock;
use tonic::async_trait;

use crate::ffn::ExpertBackend;

#[async_trait]
pub trait ExpertDB {
    async fn remove(&mut self, id: &str) -> EKResult<()>;
    async fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()>;
    async fn keys(&self) -> EKResult<Vec<String>>;
    async fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>>;
}

#[derive(Default)]
pub struct ExpertDBImpl {
    tree: BTreeMap<String, Arc<ExpertBackend>>,
}

pub fn get_expert_db() -> Arc<RwLock<dyn ExpertDB + Send + Sync>> {
    static INSTANCE: OnceCell<Arc<RwLock<ExpertDBImpl>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = ExpertDBImpl {
            tree: BTreeMap::new(),
        };
        Arc::new(RwLock::new(inner))
    });
    
    (res.clone()) as _
}

#[async_trait]
impl ExpertDB for ExpertDBImpl {
    async fn remove(&mut self, id: &str) -> EKResult<()> {
        self.tree.remove(id);
        Ok(())
    }
    async fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()> {
        self.tree.insert(id.to_owned(), Arc::new(backend));
        Ok(())
    }

    async fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>> {
        self.tree
            .get(id)
            .ok_or(EKError::ExpertNotFound(id.to_owned()))
            .cloned()
    }
    async fn keys(&self) -> EKResult<Vec<String>> {
        Ok(self.tree.keys().cloned().collect())
    }
}
