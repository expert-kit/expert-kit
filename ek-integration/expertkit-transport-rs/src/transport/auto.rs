use super::*;
use crate::transport::{grpc::GrpcTransport, shm::ShmTransport};

/// Automatic transport selection with fallback
///
/// Tries shared memory first for local workers, falls back to gRPC for remote workers
/// or if shared memory is unavailable.
pub struct AutoTransport {
    shm: ShmTransport,
    grpc: GrpcTransport,
}

impl AutoTransport {
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            shm: ShmTransport::new(timeout_sec),
            grpc: GrpcTransport::new(timeout_sec),
        }
    }
}

#[async_trait]
impl Transport for AutoTransport {
    async fn send_batch(
        &self,
        endpoint: &str,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Try shared memory first (for local workers)
        if self.shm.is_available(endpoint).await {
            match self.shm.send_batch(endpoint, requests.clone()).await {
                Ok(responses) => return Ok(responses),
                Err(e) => {
                    // SHM failed, fall back to gRPC
                    eprintln!(
                        "Shared memory transport failed for {}: {}. Falling back to gRPC",
                        endpoint, e
                    );
                }
            }
        }

        // Fall back to gRPC (for remote workers or if SHM failed)
        self.grpc.send_batch(endpoint, requests).await
    }

    fn transport_type(&self) -> TransportType {
        // Report as Auto to indicate automatic selection
        TransportType::Grpc // Use Grpc as the base type for now
    }

    async fn is_available(&self, endpoint: &str) -> bool {
        // Available if either SHM or gRPC is available
        self.shm.is_available(endpoint).await || self.grpc.is_available(endpoint).await
    }
}
