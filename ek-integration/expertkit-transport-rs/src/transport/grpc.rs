use super::*;
use log::{debug, warn};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Channel;

// Include generated proto code
#[allow(unused)]
pub mod proto {
    pub mod ek {
        pub mod object {
            pub mod v1 {
                tonic::include_proto!("ek.object.v1");
            }
        }
        pub mod worker {
            pub mod v1 {
                tonic::include_proto!("ek.worker.v1");
            }
        }
        pub mod control {
            pub mod v1 {
                tonic::include_proto!("ek.control.v1");
            }
        }
    }
}

use proto::ek::worker::v1::{ForwardReq, computation_service_client::ComputationServiceClient};

/// Connect timeout for establishing new gRPC connections (3 seconds)
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// gRPC transport with connection pooling
pub struct GrpcTransport {
    channels: Arc<RwLock<HashMap<String, Channel>>>,
    timeout: std::time::Duration,
    max_message_size: usize,
}

impl GrpcTransport {
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            channels: Arc::new(RwLock::new(HashMap::new())),
            timeout: std::time::Duration::from_secs_f64(timeout_sec),
            max_message_size: 1024 * 1024 * 1024, // 1 GB
        }
    }

    /// Create a new channel with proper timeout configuration
    async fn create_channel(&self, endpoint: &str) -> Result<Channel> {
        let uri = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("http://{}", endpoint)
        };

        // Build channel with connection timeout
        let channel = Channel::from_shared(uri)?
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(self.timeout)
            .connect()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect to {}: {}", endpoint, e))?;

        Ok(channel)
    }

    async fn get_channel(&self, endpoint: &str) -> Result<Channel> {
        // Check cache first
        {
            let channels = self.channels.read().await;
            if let Some(channel) = channels.get(endpoint) {
                return Ok(channel.clone());
            }
        }

        // Create new channel with timeout
        let channel = self.create_channel(endpoint).await?;

        // Cache it
        self.channels
            .write()
            .await
            .insert(endpoint.to_string(), channel.clone());

        Ok(channel)
    }

    /// Invalidate a cached channel (e.g., when a worker becomes unavailable)
    async fn invalidate_channel(&self, endpoint: &str) {
        let mut channels = self.channels.write().await;
        if channels.remove(endpoint).is_some() {
            warn!("[GrpcTransport] Invalidated cached channel for {}", endpoint);
        }
    }
}

#[async_trait]
impl Transport for GrpcTransport {
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Try to get cached channel first, with retry on failure
        let channel = match self.get_channel(&endpoint.grpc_addr).await {
            Ok(ch) => ch,
            Err(e) => {
                // Connection failed, try once more with fresh connection
                warn!(
                    "[GrpcTransport] Initial connection failed to {}: {}, retrying...",
                    endpoint.grpc_addr, e
                );
                self.invalidate_channel(&endpoint.grpc_addr).await;
                self.create_channel(&endpoint.grpc_addr).await?
            }
        };

        let mut client = ComputationServiceClient::new(channel)
            .max_decoding_message_size(self.max_message_size)
            .max_encoding_message_size(self.max_message_size);

        // Send ONE gRPC request PER ExpertRequest
        let mut responses = Vec::new();

        for req in requests {
            // Each ExpertRequest already has a batched tensor for multiple sequences
            // Create SequenceInfo for each sequence (all with the same expert)
            let sequences: Vec<_> = (0..req.num_sequences)
                .map(|_| proto::ek::worker::v1::forward_req::SequenceInfo {
                    experts: vec![req.expert_id.clone()],
                })
                .collect();

            let grpc_req = ForwardReq {
                instance_id: "0".to_string(),
                tensor: req.tensor_data, // Use as-is, don't concatenate!
                sequences,
            };

            debug!(
                "[GrpcTransport] Sending request for expert {} with {} sequences, tensor size {} bytes",
                req.expert_id,
                req.num_sequences,
                grpc_req.tensor.len()
            );

            // Send with timeout
            let result = tokio::time::timeout(self.timeout, client.forward(grpc_req)).await;

            match result {
                Ok(Ok(response)) => {
                    // Success - process response
                    let output_tensor = response.into_inner().output_tensor;
                    debug!(
                        "[GrpcTransport] Received response for expert {}, size {} bytes",
                        req.expert_id,
                        output_tensor.len()
                    );
                    responses.push(ExpertResponse {
                        expert_id: req.expert_id.clone(),
                        tensor_data: output_tensor,
                    });
                }
                Ok(Err(e)) => {
                    // gRPC error - invalidate channel and return error
                    warn!(
                        "[GrpcTransport] gRPC error for expert {} on {}: {}",
                        req.expert_id, endpoint.grpc_addr, e
                    );
                    self.invalidate_channel(&endpoint.grpc_addr).await;
                    return Err(anyhow::anyhow!(
                        "gRPC error for expert {}: {}",
                        req.expert_id,
                        e
                    ));
                }
                Err(_) => {
                    // Timeout - invalidate channel and return error
                    warn!(
                        "[GrpcTransport] Request timeout for expert {} on {} after {:?}",
                        req.expert_id, endpoint.grpc_addr, self.timeout
                    );
                    self.invalidate_channel(&endpoint.grpc_addr).await;
                    return Err(anyhow::anyhow!(
                        "Request timeout for expert {} after {:?}",
                        req.expert_id,
                        self.timeout
                    ));
                }
            }
        }

        Ok(responses)
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Grpc
    }

    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool {
        // Try to create a fresh connection to check availability
        self.create_channel(&endpoint.grpc_addr).await.is_ok()
    }
}
