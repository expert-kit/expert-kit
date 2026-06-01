use std::{collections::HashMap, mem::transmute, path::PathBuf, sync::Arc};

use ek_base::error::{EKError, EKResult};
use tokio::sync::RwLock;

use crate::expert_index::ExpertIndex;
use crate::safetensor::transformer::{TransformerModelDesc, TransformerPretrained};

pub struct WeightManager<'a> {
    weights: HashMap<String, Arc<RwLock<TransformerPretrained<'a>>>>,
    /// Expert index per model, loaded at startup when `ek-expert-index.json` is present.
    indices: HashMap<String, ExpertIndex>,
    /// Root of the OpenDAL Fs cache — used for the fast-path blob reads.
    cache_dir: Option<PathBuf>,
}

impl WeightManager<'_> {
    pub async fn new(roots: &'static [PathBuf], cache_dir: Option<PathBuf>) -> EKResult<Self> {
        let mut wm = WeightManager {
            weights: HashMap::new(),
            indices: HashMap::new(),
            cache_dir,
        };
        log::info!("loading model weights from {} path(s)", roots.len());
        for root in roots {
            let model_name = root.file_name().unwrap().to_str().unwrap().to_owned();
            let desc = TransformerModelDesc {
                root: root.clone(),
                ..TransformerModelDesc::default()
            };
            let tp = Arc::new(RwLock::new(TransformerPretrained::try_from_desc(&desc)?));
            wm.weights.insert(model_name.clone(), tp);

            // Attempt to load the expert index from the cache dir (written there
            // by `ek-cli weight build` since model_root may be read-only).
            let index_dir = wm
                .cache_dir
                .as_deref()
                .map(|d| d.join(&model_name))
                .unwrap_or_else(|| root.to_path_buf());
            match ExpertIndex::load(&index_dir) {
                Ok(Some(idx)) => {
                    log::info!(
                        "loaded expert index for model={}: {} entries",
                        model_name,
                        idx.entries.len()
                    );
                    wm.indices.insert(model_name, idx);
                }
                Ok(None) => {
                    log::debug!(
                        "no expert index found for model={}, using live serialization",
                        model_name
                    );
                }
                Err(e) => {
                    log::warn!("failed to load expert index for model={}: {}", model_name, e);
                }
            }
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

    /// Serve expert bytes, using the pre-extracted cache blob when available.
    ///
    /// Fast path: index present + `cached: true` → one `fs::read` from the cache dir.
    /// Slow path: mmap shard + `safetensors::tensor::serialize` (existing behaviour).
    pub async fn get_expert_bytes(
        &'static self,
        model: &str,
        layer: usize,
        eid: usize,
    ) -> EKResult<Vec<u8>> {
        let key = format!("{}/l{}-e{}", model, layer, eid);

        if let (Some(idx), Some(cache_dir)) = (self.indices.get(model), &self.cache_dir) {
            if idx.entries.get(&key).map(|e| e.cached).unwrap_or(false) {
                let blob_path = cache_dir.join(model).join(format!("l{}-e{}", layer, eid));
                match tokio::fs::read(&blob_path).await {
                    Ok(bytes) => return Ok(bytes),
                    Err(e) => {
                        log::warn!(
                            "index says {} is cached at {} but read failed: {}, falling back",
                            key,
                            blob_path.display(),
                            e
                        );
                    }
                }
            }
        }

        // Slow path: existing mmap + serialize.
        let pretrained = self.load_pretrained(model.to_owned()).await?;
        pretrained.read().await.get_expert(layer, eid).await
    }
}
