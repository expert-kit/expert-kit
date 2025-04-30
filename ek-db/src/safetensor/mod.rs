use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use ek_base::error::EKResult;
use opendal::Operator;
use opendal::{self};
use safetensors::tensor::SafeTensors;
use tokio::sync::RwLock;

pub struct SafeTensorDB {
    dal: Operator,
    data: BTreeMap<String, Bytes>,
}

type SharedSafeTensorDB = Arc<RwLock<SafeTensorDB>>;

impl SafeTensorDB {
    pub fn new_shared(dal: Operator) -> SharedSafeTensorDB {
        let inner = SafeTensorDB {
            data: BTreeMap::new(),
            dal,
        };
        Arc::new(RwLock::new(inner))
    }
}

// TODO: abstract to trait
impl SafeTensorDB {
    pub async fn load<'a>(&'a mut self, key: &str) -> EKResult<SafeTensors<'a>> {
        let raw = self.dal.read(key).await?;
        self.data.insert(key.into(), raw.to_bytes());
        let buf = self.data.get(key).unwrap();
        let st = safetensors::SafeTensors::deserialize(buf)?;
        Ok(st)
    }
}

#[cfg(test)]
mod test {
    #[tokio::test]
    async fn test_safetensor_load() {}
}
