use ek_base::error::{EKError, EKResult};

use crate::safetensor::transformer::VitalMeta;

pub struct WeightSrvClient {
    pub client: reqwest::Client,
    pub addr: String,
    token: tokio::sync::Semaphore,
}

impl WeightSrvClient {
    pub fn new(addr: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        Self {
            client,
            addr,
            token: tokio::sync::Semaphore::new(50),
        }
    }

    pub async fn load_expert(&self, model: &str, layer: usize, expert: usize) -> EKResult<Vec<u8>> {
        let _g = self.token.acquire().await.unwrap();

        let url = format!("{}/expert/{}/{}/{}", self.addr, model, layer, expert);
        let res = self.client.get(&url).send().await?;
        if res.status().is_success() {
            Ok(res.bytes().await?.to_vec())
        } else {
            Err(EKError::NotFound(format!(
                "failed to load expert from {}",
                url
            )))
        }
    }

    pub async fn load_layer(&self, model: &str, key: &str) -> EKResult<Vec<u8>> {
        let _g = self.token.acquire().await.unwrap();
        let url = format!("{}/weight/{}/{}", self.addr, model, key);
        let res = self.client.get(&url).send().await?;
        if res.status().is_success() {
            Ok(res.bytes().await?.to_vec())
        } else {
            Err(EKError::NotFound(format!(
                "failed to load layer from {}",
                url
            )))
        }
    }
    pub async fn load_meta_vital(&self, model: &str) -> EKResult<VitalMeta> {
        let _g = self.token.acquire().await.unwrap();
        let url = format!("{}/meta/vital/{}", self.addr, model);
        let res = self.client.get(&url).send().await?;
        if res.status().is_success() {
            let res = serde_json::from_str::<VitalMeta>(&res.text().await?)?;
            Ok(res)
        } else {
            Err(EKError::NotFound(format!(
                "failed to load meta vital from {}",
                url
            )))
        }
    }
}
