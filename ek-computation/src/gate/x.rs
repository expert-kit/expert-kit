use std::collections::BTreeMap;

use bytes::Bytes;
use ek_base::error::EKResult;
use opendal::Operator;
use opendal::{self, Buffer};
use safetensors::tensor::{Dtype, SafeTensors, View};

pub struct SafeTensorDB {
    dal: Operator,
    data: BTreeMap<String, Bytes>,
}

impl SafeTensorDB {
    async fn load<'a>(&'a mut self, key: &str) -> EKResult<SafeTensors<'a>> {
        let raw = self.dal.read(key).await?;
        self.data.insert(key.into(), raw.to_bytes());
        let buf = self.data.get(key).unwrap();
        let st = safetensors::SafeTensors::deserialize(&buf)?;
        Ok(st)
    }
}

#[cfg(test)]
mod test {
    #[tokio::test]
    async fn test_safetensor_load() {}
}
