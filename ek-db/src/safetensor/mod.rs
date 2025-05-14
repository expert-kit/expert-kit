pub mod memcache;
pub mod transformer;

use std::sync::Arc;

use ek_base::config::get_ek_settings;
use ek_base::error::{EKError, EKResult};
use memcache::MemCache;
use opendal::{self};
use opendal::{Buffer, Operator};
use safetensors::tensor::SafeTensors;
use tokio::sync::RwLock;

use crate::dal::op_from_settings;
use crate::weight_srv::client::WeightSrvClient;

pub struct SafeTensorDB {
    dal: Operator,
    data: MemCache,
    weight_srv: Option<WeightSrvClient>,
}

type SharedSafeTensorDB = Arc<RwLock<SafeTensorDB>>;

impl SafeTensorDB {
    pub fn new_shared() -> SharedSafeTensorDB {
        let settings = get_ek_settings();
        let settings = &settings.weight;
        let weight_srv_cli = if let Some(srv) = &settings.server {
            log::info!("weight server configured: {}", srv.addr);
            Some(WeightSrvClient::new(srv.addr.clone()))
        } else {
            log::warn!("weight server not configured");
            None
        };

        let dal = op_from_settings(&settings.cache);

        let inner = SafeTensorDB {
            data: MemCache::new(),
            dal,
            weight_srv: weight_srv_cli,
        };

        Arc::new(RwLock::new(inner))
    }
}

#[derive(Debug, Clone)]
pub struct ExpertKey {
    model: String,
    layer: usize,
    idx: usize,
}

impl ExpertKey {
    pub fn from_expert_id(model: &str, id: &str) -> EKResult<Self> {
        let inner = id.split("/l").collect::<Vec<_>>();
        if inner.len() != 2 {
            return Err(EKError::InvalidInput(format!("invalid expert id: {}", id)));
        }
        let inner = inner[1].split("-e").collect::<Vec<_>>();
        if inner.len() != 2 {
            return Err(EKError::InvalidInput(format!("invalid expert id: {}", id)));
        }
        let layer = inner[0].parse::<usize>()?;
        let expert = inner[1].parse::<usize>()?;
        Ok(Self {
            model: model.to_owned(),
            layer,
            idx: expert,
        })
    }

    pub fn new(model: String, layer: usize, idx: usize) -> Self {
        Self { model, layer, idx }
    }

    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn idx(&self) -> usize {
        self.idx
    }
    pub fn as_object_key(&self) -> String {
        format!("{}/l{}-e{}", self.model, self.layer, self.idx)
    }
}

// TODO: abstract to trait
impl SafeTensorDB {
    async fn load_from_fs_cache(&self, desc: &ExpertKey) -> EKResult<Buffer> {
        let raw = self.dal.read(&desc.as_object_key()).await?;
        Ok(raw)
    }
    async fn load_from_weight_srv(&self, desc: &ExpertKey) -> EKResult<Buffer> {
        if let Some(ref client) = self.weight_srv {
            let raw = client
                .load_expert(&desc.model, desc.layer, desc.idx)
                .await?;
            Ok(raw.into())
        } else {
            Err(EKError::NotFound(
                "cache miss and weight server not configured, unable load the weight".into(),
            ))
        }
    }

    fn as_safetensor<'a>(&'a self, key: &str) -> EKResult<SafeTensors<'a>> {
        let r = self.data.get_ref(key).unwrap();
        let st = safetensors::SafeTensors::deserialize(r)?;
        Ok(st)
    }

    pub async fn load<'a>(&'a self, desc: &ExpertKey) -> EKResult<SafeTensors<'a>> {
        let key = desc.as_object_key();
        let mem_hit = self.data.contains_key(&key);
        if mem_hit {
            let st = self.as_safetensor(&key)?;
            return Ok(st);
        }
        let cache_hit = self.dal.exists(&desc.as_object_key()).await?;

        // cache hit -> load from cache
        if cache_hit {
            let cached = self.load_from_fs_cache(desc).await;
            match cached {
                Ok(buf) => {
                    self.data.insert(&key, buf.to_bytes());
                }
                Err(e) => {
                    log::warn!(
                        "failed to load from cache: {:} reason={}",
                        desc.as_object_key(),
                        e
                    );
                }
            }
        }

        // cache miss: load from weight server
        let start = std::time::Instant::now();

        let res: Buffer = self.load_from_weight_srv(desc).await?;

        log::debug!(
            "loaded from weight server: {}, elapsed_ms={}",
            desc.as_object_key(),
            start.elapsed().as_millis()
        );
        let to_cache = res.clone();
        let cache_backend = self.dal.clone();
        let moved_key = key.clone();
        tokio::spawn(async move {
            let err = cache_backend.write(moved_key.as_str(), to_cache).await;
            if let Err(err) = err {
                log::error!("failed to cache to local cache {}: {}", moved_key, err);
            }
        });
        self.data.insert(&key, res.to_bytes());
        let st = self.as_safetensor(&key)?;
        Ok(st)
    }
}

#[cfg(test)]
mod test {
    #[tokio::test]
    async fn test_safetensor_load() {}
}
