use std::{collections::BTreeMap, sync::Arc};

use ek_base::error::{EKError, EKResult};

use crate::ffn::ExpertBackend;

#[derive(Default)]
pub struct ExpertDB {
    tree: BTreeMap<String, Arc<ExpertBackend>>,
}

impl ExpertDB {
    pub async fn remove(&mut self, id: &str) -> EKResult<()> {
        self.tree.remove(id);
        Ok(())
    }
    pub async fn insert(&mut self, id: &str, backend: ExpertBackend) -> EKResult<()> {
        self.tree.insert(id.to_owned(), Arc::new(backend));
        Ok(())
    }

    pub async fn load(&self, id: &str) -> EKResult<Arc<ExpertBackend>> {
        self.tree
            .get(id)
            .ok_or(EKError::ExpertNotFound(id.to_owned()))
            .cloned()
    }

    pub async fn keys(&self) -> EKResult<Vec<String>> {
        Ok(self.tree.keys().cloned().collect())
    }
}
