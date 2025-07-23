use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use ek_base::error::{EKError, EKResult};
use once_cell::sync::OnceCell;
use tokio::sync::RwLock;
use tonic::async_trait;

use crate::ffn::ExpertBackend;

/// Async trait for expert database operations (used for state management)
#[async_trait]
pub trait ExpertDB {
    async fn remove(&mut self, id: &str) -> EKResult<()>;
    async fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()>;
    async fn keys(&self) -> EKResult<Vec<String>>;
    async fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>>;
    fn mark_loading(&mut self, id: &str) -> EKResult<bool>;
    fn loaded(&self) -> usize;
    fn loading(&self) -> usize;
    fn has(&self, id: &str) -> bool;
}

/// Sync trait for expert database operations (used for compute operations)
#[expect(unused)]
pub trait ExpertDBSync {
    fn remove(&mut self, id: &str) -> EKResult<()>;
    fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()>;
    fn keys(&self) -> EKResult<Vec<String>>;
    fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>>;
    fn mark_loading(&mut self, id: &str) -> EKResult<bool>;
    fn loaded(&self) -> usize;
    fn loading(&self) -> usize;
    fn has(&self, id: &str) -> bool;
}

/// Core database implementation - stores the actual data
#[derive(Default)]
pub struct ExpertDBCore {
    tree: BTreeMap<String, Arc<ExpertBackend>>,
    loading: HashMap<String, bool>,
}

/// Shared database instance using Arc<RwLock<Core>>
pub type SharedExpertDB = Arc<RwLock<ExpertDBCore>>;

/// Async wrapper that implements ExpertDB trait
pub struct ExpertDBImplAsync {
    core: SharedExpertDB,
}

/// Sync wrapper that implements ExpertDBSync trait  
pub struct ExpertDBImplSync {
    core: SharedExpertDB,
}

/// Get the shared database core instance
fn get_shared_db_core() -> SharedExpertDB {
    static INSTANCE: OnceCell<SharedExpertDB> = OnceCell::new();
    INSTANCE
        .get_or_init(|| {
            Arc::new(RwLock::new(ExpertDBCore {
                tree: BTreeMap::new(),
                loading: HashMap::new(),
            }))
        })
        .clone()
}

/// Get the async expert database instance
pub fn get_expert_db() -> Arc<RwLock<dyn ExpertDB + Send + Sync>> {
    static INSTANCE: OnceCell<Arc<RwLock<ExpertDBImplAsync>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let core = get_shared_db_core();
        Arc::new(RwLock::new(ExpertDBImplAsync { core }))
    });
    (res.clone()) as _
}

/// Get the sync expert database instance
pub fn get_expert_db_sync() -> Arc<std::sync::RwLock<dyn ExpertDBSync + Send + Sync>> {
    static INSTANCE: OnceCell<Arc<std::sync::RwLock<ExpertDBImplSync>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let core = get_shared_db_core();
        Arc::new(std::sync::RwLock::new(ExpertDBImplSync { core }))
    });
    (res.clone()) as _
}

impl ExpertDBCore {
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

    fn mark_loading(&mut self, id: &str) -> EKResult<bool> {
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

    fn remove(&mut self, id: &str) -> EKResult<()> {
        self.tree.remove(id);
        Ok(())
    }

    fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()> {
        self.loading.remove(id);
        self.tree.insert(id.to_owned(), Arc::new(backend));
        Ok(())
    }

    fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>> {
        self.tree
            .get(id)
            .ok_or(EKError::ExpertNotFound(id.to_owned()))
            .cloned()
    }

    fn keys(&self) -> EKResult<Vec<String>> {
        Ok(self.tree.keys().cloned().collect())
    }
}

#[async_trait]
impl ExpertDB for ExpertDBImplAsync {
    fn loading(&self) -> usize {
        // Need to access core synchronously for these simple operations
        // This is safe because we're just reading counters
        tokio::task::block_in_place(|| {
            let core = self.core.blocking_read();
            core.loading()
        })
    }

    fn loaded(&self) -> usize {
        tokio::task::block_in_place(|| {
            let core = self.core.blocking_read();
            core.loaded()
        })
    }

    fn has(&self, id: &str) -> bool {
        tokio::task::block_in_place(|| {
            let core = self.core.blocking_read();
            core.has(id)
        })
    }

    fn mark_loading(&mut self, id: &str) -> EKResult<bool> {
        tokio::task::block_in_place(|| {
            let mut core = self.core.blocking_write();
            core.mark_loading(id)
        })
    }

    async fn remove(&mut self, id: &str) -> EKResult<()> {
        let mut core = self.core.write().await;
        core.remove(id)
    }

    async fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()> {
        let mut core = self.core.write().await;
        core.insert(id, backend)
    }

    async fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>> {
        let core = self.core.read().await;
        core.load(id)
    }

    async fn keys(&self) -> EKResult<Vec<String>> {
        let core = self.core.read().await;
        core.keys()
    }
}

impl ExpertDBSync for ExpertDBImplSync {
    fn loading(&self) -> usize {
        let core = self.core.blocking_read();
        core.loading()
    }

    fn loaded(&self) -> usize {
        let core = self.core.blocking_read();
        core.loaded()
    }

    fn has(&self, id: &str) -> bool {
        let core = self.core.blocking_read();
        core.has(id)
    }

    fn mark_loading(&mut self, id: &str) -> EKResult<bool> {
        let mut core = self.core.blocking_write();
        core.mark_loading(id)
    }

    fn remove(&mut self, id: &str) -> EKResult<()> {
        let mut core = self.core.blocking_write();
        core.remove(id)
    }

    fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()> {
        let mut core = self.core.blocking_write();
        core.insert(id, backend)
    }

    fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>> {
        let core = self.core.blocking_read();
        core.load(id)
    }

    fn keys(&self) -> EKResult<Vec<String>> {
        let core = self.core.blocking_read();
        core.keys()
    }
}
