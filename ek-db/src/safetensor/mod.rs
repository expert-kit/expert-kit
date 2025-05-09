pub mod transformer;

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use ek_base::error::{EKError, EKResult};
use opendal::{self};
use opendal::{Buffer, Operator};
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
    pub async fn load(&self, key: &str) -> EKResult<Buffer> {
        let raw = self.dal.read(key).await?;
        Ok(raw)
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
