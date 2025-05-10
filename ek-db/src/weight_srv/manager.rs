use std::{collections::HashMap, mem::transmute, path::PathBuf, sync::Arc};

use ek_base::error::{EKError, EKResult};
use tokio::sync::RwLock;

use crate::safetensor::transformer::{TransformerModelDesc, TransformerPretrained};

pub struct WeightManager<'a> {
    weights: HashMap<String, Arc<RwLock<TransformerPretrained<'a>>>>,
}

impl WeightManager<'_> {
    pub async fn new(roots: &'static [PathBuf]) -> EKResult<Self> {
        let mut wm = WeightManager {
            weights: HashMap::new(),
        };
        log::info!("loading model weights from {} path", roots.len());
        for root in roots {
            let desc = TransformerModelDesc {
                root: root.clone(),
                ..TransformerModelDesc::default()
            };
            let tp = Arc::new(RwLock::new(TransformerPretrained::try_from_desc(&desc)?));
            wm.weights
                .insert(root.file_name().unwrap().to_str().unwrap().to_owned(), tp);
        }
        Ok(unsafe { transmute::<WeightManager<'_>, WeightManager<'_>>(wm) })
    }
    pub async fn load_pretrained<'b>(
        &'b self,
        model: String,
    ) -> EKResult<Arc<RwLock<TransformerPretrained<'static>>>>
    where
        'b: 'static,
    {
        let pretrained = self
            .weights
            .get(&model)
            .ok_or(EKError::NotFound(model.clone()))?;
        Ok(pretrained.clone())
    }
}
