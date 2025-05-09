use std::{collections::HashMap, path::PathBuf};

use ek_base::error::{EKError, EKResult};
use memmap2::{Mmap, MmapOptions};
use safetensors::{SafeTensors, tensor::TensorView};
use tokio::fs::File;

struct WeightMap {
    map: std::collections::HashMap<String, String>,
}

impl WeightMap {
    fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        let mut map = HashMap::new();
        let path = desc.root.join(&desc.wight_map_name);
        let file = std::fs::File::open(path).unwrap();
        let res: serde_json::Value = serde_json::from_reader(file).unwrap();
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
    pub wight_map_name: String,
}

impl Default for TransformerModelDesc {
    fn default() -> Self {
        Self {
            root: PathBuf::new(),
            wight_map_name: "model.safetensors.index.json".to_string(),
        }
    }
}

pub struct TransformerPretrained<'a> {
    desc: TransformerModelDesc,
    weight_map: WeightMap,
    safetensors: HashMap<PathBuf, SafeTensors<'a>>,
    mmap: HashMap<PathBuf, Mmap>,
}

impl<'a> TransformerPretrained<'a> {
    pub fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        // return Sel
        let weight_map = WeightMap::try_from_desc(desc).unwrap();
        Ok(Self {
            desc: desc.clone(),
            weight_map,
            mmap: HashMap::new(),
            safetensors: HashMap::new(),
        })
    }

    pub async fn get(&'a mut self, key: &str) -> EKResult<TensorView<'a>> {
        let fp = self.weight_map.map_layer(&key.to_string());
        let fp = self.desc.root.join(fp);
        if let std::collections::hash_map::Entry::Vacant(e) = self.safetensors.entry(fp.clone()) {
            let file = File::open(fp.clone()).await?;
            let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
            let buffer = self.mmap.entry(fp.clone()).or_insert(buffer);
            let st = safetensors::SafeTensors::deserialize(buffer).unwrap();
            e.insert(st);
        }
        let st = self.safetensors.get(&fp.clone()).unwrap();
        let tv = st.tensor(key)?;
        Ok(tv)
    }
}

#[cfg(test)]
mod test {
    use ek_base::utils::workspace_root;

    use crate::safetensor::transformer::{TransformerModelDesc, TransformerPretrained};

    #[tokio::test]
    async fn test_basic() {
        let root = workspace_root();
        let test_model = root.join("ek-db").join("resources").join("ds-tiny");
        let desc = TransformerModelDesc {
            root: test_model.clone(),
            ..TransformerModelDesc::default()
        };
        let mut pretrained = TransformerPretrained::try_from_desc(&desc).unwrap();
        let tensor = pretrained
            .get("model.layers.21.mlp.experts.94.down_proj.weight")
            .await
            .unwrap();
        assert_eq!(tensor.shape(), &[16, 8]);
    }
}
