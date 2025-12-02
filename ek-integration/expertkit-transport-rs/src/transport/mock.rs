use super::*;

/// Mock transport for testing - echoes back requests
pub struct MockTransport;

impl MockTransport {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Transport for MockTransport {
    async fn send_batch(
        &self,
        _endpoint: &str,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Echo back the same data
        Ok(requests
            .into_iter()
            .map(|req| ExpertResponse {
                expert_id: req.expert_id,
                tensor_data: req.tensor_data,
            })
            .collect())
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Grpc
    }

    async fn is_available(&self, _endpoint: &str) -> bool {
        true
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}
