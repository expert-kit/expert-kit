use std::{collections::HashMap, mem::transmute, path::PathBuf, sync::Arc};

use actix_web::{HttpResponse, Responder, body::BoxBody, http::header::ContentType};
use ek_base::error::{EKError, EKResult};
use memmap2::{Mmap, MmapOptions};
use once_cell::sync::OnceCell;
use safetensors::{SafeTensors, tensor::TensorView};
use tokio::fs::File;

#[derive(Debug, Clone)]
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
}

impl Default for TransformerModelDesc {
    fn default() -> Self {
        Self {
            root: PathBuf::new(),
            weight_map_name: "model.safetensors.index.json".to_string(),
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

    safetensors_cache: HashMap<PathBuf, SafeTensorWithData<'data>>,
}

impl TransformerPretrained<'_> {
    pub fn try_from_desc(desc: &TransformerModelDesc) -> EKResult<Self> {
        // return Sel
        let weight_map = WeightMap::try_from_desc(desc)?;
        Ok(Self {
            desc: desc.clone(),
            weight_map,
            safetensors_cache: HashMap::new(),
        })
    }

    pub async fn get_raw(&mut self, key: String) -> EKResult<Vec<u8>> {
        let fp = self.weight_map.map_layer(&key.to_string());
        let fp = self.desc.root.join(fp);
        let hit = self.safetensors_cache.contains_key(&fp.clone());
        if !hit {
            let file = File::open(fp.clone()).await.unwrap();
            let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
            // self.mmap_cache.insert(fp.clone(), buffer);
            // let mmap = self.mmap_cache.get(&fp).unwrap();
            let st = SafeTensorWithData::new(buffer);
            self.safetensors_cache.insert(fp.clone(), st);
        }

        let serialized = {
            let st_data = self.safetensors_cache.get(&fp).unwrap();
            let st_data: &SafeTensorWithData = unsafe { transmute(st_data) };
            let tv = &st_data.safetensors().tensor(key.as_str()).unwrap();
            safetensors::tensor::serialize([("data", tv)].to_vec(), &None)?
        };

        Ok(serialized)
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
        let mut pretrained: TransformerPretrained =
            TransformerPretrained::try_from_desc(&desc).unwrap();
        let tensor = pretrained
            .get_raw("model.layers.21.mlp.experts.94.down_proj.weight".to_owned())
            .await
            .unwrap();
        let tv = safetensors::SafeTensors::deserialize(&tensor)
            .unwrap()
            .tensor("data")
            .unwrap();

        assert_eq!(tv.shape(), &[16, 8]);
    }
}
