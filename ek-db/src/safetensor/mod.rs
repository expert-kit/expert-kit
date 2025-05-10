pub mod transformer;

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use ek_base::config::get_ek_settings;
use ek_base::error::{EKError, EKResult};
use opendal::{self};
use opendal::{Buffer, Operator};
use safetensors::tensor::SafeTensors;
use tokio::sync::RwLock;

use crate::dal::op_from_settings;
use crate::weight_srv::client::WeightSrvClient;

pub struct SafeTensorDB {
    dal: Operator,

    data: BTreeMap<String, Bytes>,
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
            data: BTreeMap::new(),
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
        let inner = id
            .strip_prefix("model-layer")
            .ok_or(EKError::InvalidInput(format!(
                "no prefix 'model-layer' found in: {}",
                id
            )))?
            .strip_suffix(".safetensors")
            .ok_or(EKError::InvalidInput(format!(
                "no suffix '.safetensors' found in: {}",
                id
            )))?;

        let (layer_str, expert_part) =
            inner
                .split_once("-expert")
                .ok_or(EKError::InvalidInput(format!(
                    "no '-expert' found in two ids: {}",
                    id
                )))?;

        if layer_str.is_empty() || !layer_str.chars().all(char::is_numeric) {
            return Err(EKError::InvalidInput(format!(
                "layer part is empty or not numeric: {}",
                layer_str
            )));
        }
        if expert_part.is_empty() || !expert_part.chars().all(char::is_numeric) {
            return Err(EKError::InvalidInput(format!(
                "expert part is empty or not numeric: {}",
                expert_part
            )));
        }

        let layer = layer_str.parse::<usize>()?;
        let expert = expert_part.parse::<usize>()?;
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
        format!(
            "{}/model-layer{}-expert{}.safetensors",
            self.model, self.layer, self.idx
        )
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

    pub async fn load(&self, desc: &ExpertKey) -> EKResult<Buffer> {
        let cache_hit = self.dal.exists(&desc.as_object_key()).await?;

        // cache hit -> load from cache
        if cache_hit {
            let cached = self.load_from_fs_cache(desc).await;
            if let Ok(ref buf) = cached {
                return Ok(buf.clone());
            } else {
                log::warn!("failed to load from cache: {:}", desc.as_object_key());
            }
        }

        // cache miss: load from weight server
        let start = std::time::Instant::now();

        let res: Buffer = self.load_from_weight_srv(desc).await?;

        log::info!(
            "loaded from weight server: {}, elapsed_ms={}",
            desc.as_object_key(),
            start.elapsed().as_millis()
        );
        let obj_key = desc.as_object_key().to_owned();
        let to_cache = res.clone();
        let cache_backend = self.dal.clone();
        tokio::spawn(async move { cache_backend.write(obj_key.as_str(), to_cache).await });

        Ok(res)
    }

    pub fn save(&mut self, key: &str, buf: Buffer) -> EKResult<()> {
        self.data.insert(key.into(), buf.to_bytes());
        Ok(())
    }

    pub fn as_safetensor<'a>(&'a self, key: &str) -> EKResult<SafeTensors<'a>> {
        let d = self
            .data
            .get(key)
            .ok_or(EKError::NotFound("tensor not found".into()))?;
        let st = safetensors::SafeTensors::deserialize(d)?;
        Ok(st)
    }
}

#[cfg(test)]
mod test {
    #[tokio::test]
    async fn test_safetensor_load() {}
}
