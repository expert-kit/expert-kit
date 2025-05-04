use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

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
    fn lock(&mut self, id: &str) -> EKResult<bool>;
    fn unlock(&mut self, id: &str);
    fn loaded(&self) -> usize;
    fn loading(&self) -> usize;
    fn has(&self, id: &str) -> bool;
}

#[derive(Default)]
pub struct ExpertDBImpl {
    tree: BTreeMap<String, Arc<ExpertBackend>>,
    loading: HashMap<String, bool>,
}

pub fn get_expert_db() -> Arc<RwLock<dyn ExpertDB + Send + Sync>> {
    static INSTANCE: OnceCell<Arc<RwLock<ExpertDBImpl>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = ExpertDBImpl {
            tree: BTreeMap::new(),
            loading: HashMap::new(),
        };
        Arc::new(RwLock::new(inner))
    });

    (res.clone()) as _
}

#[async_trait]
impl ExpertDB for ExpertDBImpl {
    fn loading(&self) -> usize {
        self.loading.len()
    }
    fn loaded(&self) -> usize {
        self.tree.len()
    }
    fn has(&self, id: &str) -> bool {
        let is_loading = self.loading.contains_key(id);
        let is_loaded = self.tree.contains_key(id);
        is_loading || is_loaded
    }

    fn unlock(&mut self, id: &str) {
        self.loading.remove(id);
    }
    fn lock(&mut self, id: &str) -> EKResult<bool> {
        let locked = self.loading.get(id);
        if let Some(locked) = locked {
            if *locked {
                return Ok(false);
            }
        }
        let entry = self.loading.entry(id.into()).or_insert(true);
        *entry = true;
        Ok(true)
    }
    async fn remove(&mut self, id: &str) -> EKResult<()> {
        self.tree.remove(id);
        Ok(())
    }
    async fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()> {
        self.loading.remove(id);
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
