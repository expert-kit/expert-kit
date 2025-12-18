use super::*;
use crate::transport::{grpc::GrpcTransport, shm::ShmTransport};
use log::info;

#[cfg(feature = "rdma")]
use crate::transport::rdma::RdmaTransport;

/// Transport selector based on WorkerEndpoint channel type
pub struct AutoTransport {
    grpc: GrpcTransport,
    shm: ShmTransport,
    #[cfg(feature = "rdma")]
    rdma: RdmaTransport,
}

impl AutoTransport {
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            grpc: GrpcTransport::new(timeout_sec),
            shm: ShmTransport::new(timeout_sec),
            #[cfg(feature = "rdma")]
            rdma: RdmaTransport::new(timeout_sec),
        }
    }

    /// Warm up connections to all workers in the routing table
    pub async fn warmup_connections(&self, endpoints: &[WorkerEndpoint]) -> Result<()> {
        info!(
            "[AutoTransport] Warming up connections to {} workers",
            endpoints.len()
        );

        let start = std::time::Instant::now();
        let mut tasks: Vec<tokio::task::JoinHandle<Result<()>>> = Vec::new();

        for endpoint in endpoints {
            match endpoint.channel.as_str() {
                "rdma" => {
                    #[cfg(feature = "rdma")]
                    {
                        info!(
                            "[AutoTransport] Pre-connecting RDMA to worker {} (TCP port {})",
                            endpoint.grpc_addr, endpoint.rdma_tcp_port
                        );
                        let endpoint_clone = endpoint.clone();
                        let rdma = self.rdma.clone();
                        tasks.push(tokio::spawn(async move {
                            rdma.warmup_connection(&endpoint_clone).await
                        }));
                    }
                    #[cfg(not(feature = "rdma"))]
                    {
                        log::warn!(
                            "[AutoTransport] Skipping RDMA warmup for {} (feature not enabled)",
                            endpoint.grpc_addr
                        );
                    }
                }
                "shm" => {
                    // SHM connections are lazy (just open existing files), so no warmup needed
                    info!(
                        "[AutoTransport] Skipping warmup for SHM worker {} (lazy connection)",
                        endpoint.grpc_addr
                    );
                }
                "grpc" => {
                    // gRPC connections are also lazy, no warmup needed
                    info!(
                        "[AutoTransport] Skipping warmup for gRPC worker {} (lazy connection)",
                        endpoint.grpc_addr
                    );
                }
                _ => {}
            }
        }

        // Wait for all warmup tasks to complete
        let results = futures::future::join_all(tasks).await;
        let mut success_count = 0;
        let mut error_count = 0;

        for result in results {
            match result {
                Ok(Ok(_)) => success_count += 1,
                Ok(Err(e)) => {
                    log::error!("[AutoTransport] Warmup connection failed: {}", e);
                    error_count += 1;
                }
                Err(e) => {
                    log::error!("[AutoTransport] Warmup task panicked: {}", e);
                    error_count += 1;
                }
            }
        }

        info!(
            "[AutoTransport] Connection warmup completed in {:?} - {} succeeded, {} failed",
            start.elapsed(),
            success_count,
            error_count
        );

        Ok(())
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
                #[cfg(feature = "rdma")]
                {
                    // Worker has RDMA queues - use RDMA transport
                    info!(
                        "[AutoTransport] Using RDMA for worker {} (TCP port {})",
                        endpoint.grpc_addr, endpoint.rdma_tcp_port
                    );
                    self.rdma.send_batch(endpoint, requests).await
                }
                #[cfg(not(feature = "rdma"))]
                {
                    // RDMA feature not enabled
                    Err(anyhow::anyhow!(
                        "RDMA transport not available. Worker {} advertises channel='rdma' (TCP port {}), \
                         but this client was not built with RDMA support. \
                         Rebuild with: cargo build --features rdma",
                        endpoint.grpc_addr,
                        endpoint.rdma_tcp_port
                    ))
                }
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
            "rdma" => {
                #[cfg(feature = "rdma")]
                {
                    self.rdma.is_available(endpoint).await
                }
                #[cfg(not(feature = "rdma"))]
                {
                    false
                }
            }
            _ => false,
        }
    }
}
