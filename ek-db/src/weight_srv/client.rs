use ek_base::error::{EKError, EKResult};

pub struct WeightSrvClient {
    pub client: reqwest::Client,
    pub addr: String,
}

impl WeightSrvClient {
    pub fn new(addr: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap();
        Self { client, addr }
    }

    pub async fn load_expert(&self, model: &str, layer: usize, expert: usize) -> EKResult<Vec<u8>> {
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
}
