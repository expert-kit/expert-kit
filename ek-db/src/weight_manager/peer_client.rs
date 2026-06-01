use std::sync::Arc;

use bytes::Bytes;
use ek_base::error::{EKError, EKResult};

use crate::safetensor::ExpertKey;

/// HTTP client for fetching expert weights from a peer LocalWeightManager.
/// Uses a shared reqwest::Client with connection pooling for parallel transfers.
pub struct WeightManagerClient {
    client: reqwest::Client,
    addr: String,
}

impl WeightManagerClient {
    pub fn new(addr: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .pool_max_idle_per_host(16)
            .build()
            .unwrap();
        Self { addr, client }
    }

    /// Fetch a single expert's weight bytes from this peer via HTTP.
    pub async fn get_expert(&self, key: &ExpertKey) -> EKResult<Arc<Bytes>> {
        let url = format!(
            "{}/expert/{}/{}/{}",
            self.addr,
            key.model(),
            key.layer(),
            key.idx()
        );
        let res = self.client.get(&url).send().await?;
        if res.status().is_success() {
            Ok(Arc::new(res.bytes().await?))
        } else {
            Err(EKError::NotFound(format!(
                "peer fetch from {url} returned {}",
                res.status()
            )))
        }
    }
}
