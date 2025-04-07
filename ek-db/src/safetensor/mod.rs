use std::collections::BTreeMap;

use bytes::Bytes;
use ek_base::error::EKResult;
use opendal::Operator;
use opendal::{self};
use safetensors::tensor::SafeTensors;

pub struct SafeTensorDB {
    dal: Operator,
    data: BTreeMap<String, Bytes>,
}

impl Default for SafeTensorDB {
    fn default() -> Self {
        Self::new()
    }
}

impl SafeTensorDB {
    pub fn new() -> Self {
        SafeTensorDB {
            data: BTreeMap::new(),
            dal: todo!(),
        }
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
