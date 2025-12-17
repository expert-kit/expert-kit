use super::*;
use crate::transport::{grpc::GrpcTransport, shm::ShmTransport};
use log::info;

/// Transport selector based on WorkerEndpoint channel type
pub struct AutoTransport {
    grpc: GrpcTransport,
    shm: ShmTransport,
    // TODO: Add RDMA transport when implemented
    // rdma: RdmaTransport,
}

impl AutoTransport {
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            grpc: GrpcTransport::new(timeout_sec),
            shm: ShmTransport::new(timeout_sec),
        }
    }
}

#[async_trait]
impl Transport for AutoTransport {
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Select transport based on worker's advertised channel type
        match endpoint.channel.as_str() {
            "grpc" => {
                // Worker has gRPC server - use gRPC transport
                info!(
                    "[AutoTransport] Using gRPC for worker {}",
                    endpoint.grpc_addr
                );
                self.grpc.send_batch(endpoint, requests).await
            }
            "rdma" => {
                // Worker has RDMA queues - client needs RDMA transport
                Err(anyhow::anyhow!(
                    "RDMA transport not yet implemented in client. \
                     Worker {} advertises channel='rdma' (TCP port {}), but client only supports gRPC.",
                    endpoint.grpc_addr,
                    endpoint.rdma_tcp_port
                ))
            }
            "shm" => {
                // Worker has shared memory queues - workers CREATE /dev/shm files!
                info!(
                    "[AutoTransport] Using SHM for worker {} (queue: {})",
                    endpoint.grpc_addr, endpoint.shm_queue_prefix
                );
                self.shm.send_batch(endpoint, requests).await
            }
            unknown => Err(anyhow::anyhow!(
                "Unknown channel type '{}' for worker {}. Supported: grpc, rdma, shm",
                unknown,
                endpoint.grpc_addr
            )),
        }
    }

    fn transport_type(&self) -> TransportType {
        // Return Grpc as default (most common)
        TransportType::Grpc
    }

    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool {
        match endpoint.channel.as_str() {
            "grpc" => self.grpc.is_available(endpoint).await,
            "shm" => self.shm.is_available(endpoint).await,
            _ => false,
        }
    }
}
