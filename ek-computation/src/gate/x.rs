use ek_base::error::EKResult;
use opendal;
use opendal::Operator;
use safetensors::tensor::{Dtype, SafeTensors, View};

use crate::tch_safetensors::read_safetensors;
pub struct SafeTensorLoader {
    dal: Operator,
}

impl SafeTensorLoader {
    async fn load(&self, key: &str) -> EKResult<()> {
        let raw = self.dal.read(key).await?;
        let tensor = read_safetensors(raw.to_bytes().as_ref())?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    #[tokio::test]
    async fn test_safetensor_load() {}
}
