use std::{collections::HashMap, mem::transmute, path::PathBuf, sync::Arc};

use actix_web::{HttpResponse, Responder, body::BoxBody, http::header::ContentType};
use ek_base::error::{EKError, EKResult};
use memmap2::{Mmap, MmapOptions};
use moka::future::Cache;
use once_cell::sync::OnceCell;
use safetensors::{SafeTensors, tensor::TensorView};
use serde::{Deserialize, Serialize};
use tokio::fs::File;
#[derive(Debug, Clone)]
pub struct ModelConfig {
    map: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize)]
pub struct VitalMeta {
    pub moe_layers: (usize, usize),
    pub routed_experts: usize,
    pub hidden_dim: usize,
    pub inter_dim: usize,
}

impl ModelConfig {
    fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        let path = desc.root.join(&desc.config_name);
        let file = std::fs::File::open(path.clone()).map_err(move |e| {
            log::error!("can not found model_config at {}", &path.to_string_lossy());
            EKError::IoError(e)
        })?;
        let map: HashMap<_, _> = serde_json::from_reader(file)?;
        Ok(Self { map })
    }
    pub fn model_type(&self) -> &str {
        self.map.get("model_type").unwrap().as_str().unwrap()
    }

    pub fn moe_layers(&self) -> Option<(usize, usize)> {
        match self.model_type() {
            "deepseek_v3" => {
                let start = self
                    .map
                    .get("first_k_dense_replace")
                    .unwrap()
                    .as_u64()
                    .unwrap() as usize;
                let end = self.map.get("num_hidden_layers")?.as_u64()? as usize;
                Some((start, end))
            }
            _ => {
                let end = self.map.get("num_hidden_layers")?.as_u64()? as usize;
                Some((0, end))
            }
        }
    }

    pub fn routed_experts(&self) -> Option<usize> {
        match self.model_type() {
            "deepseek_v3" => Some(self.map.get("n_routed_experts")?.as_u64()? as usize),
            "qwen3_moe" => Some(self.map.get("num_experts")?.as_u64()? as usize),
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn dim(&self) -> Option<(usize, usize)> {
        let hidden = self.map.get("hidden_size")?.as_u64()? as usize;
        let intermediate = self.map.get("moe_intermediate_size")?.as_u64()? as usize;
        Some((hidden, intermediate))
    }
    pub fn normalized_vital(&self) -> EKResult<VitalMeta> {
        let dim = self.dim().ok_or(EKError::InvalidInput(
            "can not determine hidden_dim and inter_dim".to_string(),
        ))?;
        Ok(VitalMeta {
            moe_layers: self.moe_layers().ok_or(EKError::InvalidInput(
                "can not determine moe layers".to_string(),
            ))?,
            routed_experts: self.routed_experts().ok_or(EKError::InvalidInput(
                "can not determine routed_experts".to_string(),
            ))?,
            hidden_dim: dim.0,
            inter_dim: dim.1,
        })
    }
}

struct WeightMap {
    map: std::collections::HashMap<String, String>,
}

impl WeightMap {
    fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        let mut map = HashMap::new();
        let path = desc.root.join(&desc.weight_map_name);
        let file = std::fs::File::open(path.clone()).map_err(move |e| {
            log::error!(
                "can not found weight_map_file at {}",
                &path.to_string_lossy()
            );
            EKError::IoError(e)
        })?;
        let res: serde_json::Value = serde_json::from_reader(file)?;
        res.get("weight_map")
            .ok_or(EKError::NotFound("weight_map key not found".to_string()))?
            .as_object()
            .ok_or(EKError::NotFound(
                "weight_map is not a valid object".to_string(),
            ))?
            .iter()
            .for_each(|(k, v)| {
                let v = v.as_str().unwrap();
                map.insert(k.to_string(), v.to_string());
            });
        Ok(Self { map })
    }
    fn map_layer(&self, key: &String) -> String {
        self.map.get(key).unwrap().to_owned()
    }
}

#[derive(Debug, Clone)]
pub struct TransformerModelDesc {
    pub root: PathBuf,
    pub weight_map_name: String,
    pub config_name: String,
}

impl Default for TransformerModelDesc {
    fn default() -> Self {
        Self {
            root: PathBuf::new(),
            weight_map_name: "model.safetensors.index.json".to_string(),
            config_name: "config.json".to_string(),
        }
    }
}

#[derive(Debug)]
struct SafeTensorWithData<'data> {
    st: OnceCell<Arc<SafeTensors<'data>>>,
    mmap: Mmap,
}

impl<'data> SafeTensorWithData<'data> {
    fn new(mmap: Mmap) -> Self {
        Self {
            st: OnceCell::new(),
            mmap,
        }
    }
    fn safetensors(&'data self) -> Arc<SafeTensors<'data>> {
        let st = self.st.get_or_init(|| {
            let st = safetensors::SafeTensors::deserialize(&self.mmap).unwrap();
            Arc::new(st)
        });
        st.clone()
    }
}

pub struct WrappedTensorView<'data> {
    data: &'data SafeTensorWithData<'data>,
    key: String,
}

impl Responder for WrappedTensorView<'_> {
    type Body = BoxBody;

    fn respond_to(self, _: &actix_web::HttpRequest) -> HttpResponse<BoxBody> {
        let body = self.inner().unwrap().data().to_vec();
        let res = HttpResponse::Ok()
            .content_type(ContentType::octet_stream())
            .body(body);
        res
    }
}

impl<'a> WrappedTensorView<'a> {
    pub fn inner(&self) -> EKResult<TensorView<'a>> {
        let st = self.data.safetensors();
        let tv = st.tensor(self.key.as_str())?;
        Ok(tv.clone())
    }
}

pub struct TransformerPretrained<'data> {
    desc: TransformerModelDesc,
    weight_map: WeightMap,
    model_config: ModelConfig,
    safetensors_cache: Cache<PathBuf, Arc<SafeTensorWithData<'data>>>,
}

impl<'data> TransformerPretrained<'data>
where
    'data: 'static,
{
    pub fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        // return Sel
        let weight_map = WeightMap::try_from_desc(desc)?;
        let model_config = ModelConfig::try_from_desc(desc)?;
        Ok(Self {
            desc: desc.clone(),
            weight_map,
            model_config,
            safetensors_cache: Cache::new(100000),
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.model_config
    }

    async fn get_safetensor(&self, key: String) -> EKResult<Arc<SafeTensorWithData<'data>>> {
        let fp = self.weight_map.map_layer(&key.to_string());
        let fp = self.desc.root.join(fp);
        let hit = self.safetensors_cache.contains_key(&fp.clone());
        if !hit {
            let file = File::open(fp.clone()).await.unwrap();
            let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
            let st = SafeTensorWithData::new(buffer);
            let st = Arc::new(st);
            self.safetensors_cache.insert(fp.clone(), st.clone()).await;
        }
        let res = self
            .safetensors_cache
            .get(&fp)
            .await
            .ok_or(EKError::NotFound("safetensor not found".to_string()))?;

        Ok(res)
    }

    pub async fn get_layer(&self, key: String) -> EKResult<Vec<u8>> {
        let st = self.get_safetensor(key.clone()).await?;
        let serialized = {
            let st: &SafeTensorWithData = unsafe { transmute(st.as_ref()) };
            let tv = &st.safetensors().tensor(key.as_str()).unwrap();
            safetensors::tensor::serialize([("data", tv)].to_vec(), &None)?
        };
        Ok(serialized)
    }

    async fn construct_expert_key(
        &self,
        layer_id: usize,
        expert_id: usize,
    ) -> EKResult<Vec<String>> {
        let key_up = format!(
            "model.layers.{}.mlp.experts.{}.up_proj.weight",
            layer_id, expert_id
        );
        let key_gate = format!(
            "model.layers.{}.mlp.experts.{}.down_proj.weight",
            layer_id, expert_id
        );
        let key_down = format!(
            "model.layers.{}.mlp.experts.{}.gate_proj.weight",
            layer_id, expert_id
        );

        Ok(vec![key_down, key_gate, key_up])
    }

    pub async fn get_expert(&self, layer_id: usize, eid: usize) -> EKResult<Vec<u8>> {
        let keys = self.construct_expert_key(layer_id, eid).await?;
        let mut tensors = vec![];

        for key in keys.iter() {
            let st = self.get_safetensor(key.clone()).await?;
            let st: &SafeTensorWithData = unsafe { transmute(st.as_ref()) };
            let tensor = st.safetensors().tensor(key)?;
            tensors.push((key.clone(), tensor));
        }
        let serialized = safetensors::tensor::serialize(tensors, &None)?;
        Ok(serialized)
    }
}

#[cfg(test)]
mod test {
    use ek_base::utils::workspace_root;

    use crate::safetensor::transformer::{TransformerModelDesc, TransformerPretrained};

    #[tokio::test]
    async fn test_get_layer() {
        let root = workspace_root();
        let test_model = root.join("ek-db").join("resources").join("ds-tiny");
        let desc = TransformerModelDesc {
            root: test_model.clone(),
            ..TransformerModelDesc::default()
        };
        let pretrained: TransformerPretrained =
            TransformerPretrained::try_from_desc(&desc).unwrap();
        let tensor = pretrained
            .get_layer("model.layers.21.mlp.experts.94.down_proj.weight".to_owned())
            .await
            .unwrap();
        let tv = safetensors::SafeTensors::deserialize(&tensor)
            .unwrap()
            .tensor("data")
            .unwrap();

        assert_eq!(tv.shape(), &[16, 8]);
    }

    #[tokio::test]
    async fn test_get_expert() {
        let root = workspace_root();
        let test_model = root.join("ek-db").join("resources").join("ds-tiny");
        let desc = TransformerModelDesc {
            root: test_model.clone(),
            ..TransformerModelDesc::default()
        };
        let pretrained: TransformerPretrained =
            TransformerPretrained::try_from_desc(&desc).unwrap();
        let tensor = pretrained.get_expert(21, 97).await.unwrap();
        let st = safetensors::SafeTensors::deserialize(&tensor).unwrap();
        let names = st.names();
        assert_eq!(names.len(), 3);
        let expected = vec![
            "model.layers.21.mlp.experts.97.gate_proj.weight",
            "model.layers.21.mlp.experts.97.down_proj.weight",
            "model.layers.21.mlp.experts.97.up_proj.weight",
        ];

        for name in expected {
            assert!(names.contains(&&name.to_string()));
        }
        let tensor = st
            .tensor("model.layers.21.mlp.experts.97.down_proj.weight")
            .unwrap();
        assert_eq!(tensor.shape(), &[16, 8]);
    }
}
