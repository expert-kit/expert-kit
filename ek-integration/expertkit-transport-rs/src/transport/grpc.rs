use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Channel;

// Include generated proto code
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

    async fn get_channel(&self, endpoint: &str) -> Result<Channel> {
        // Check cache first
        {
            let channels = self.channels.read().await;
            if let Some(channel) = channels.get(endpoint) {
                return Ok(channel.clone());
            }
        }

        // Create new channel
        let uri = if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            endpoint.to_string()
        } else {
            format!("http://{}", endpoint)
        };

        let channel = Channel::from_shared(uri)?
            .connect()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to connect to {}: {}", endpoint, e))?;

        // Cache it
        self.channels
            .write()
            .await
            .insert(endpoint.to_string(), channel.clone());

        Ok(channel)
    }
}

#[async_trait]
impl Transport for GrpcTransport {
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Extract gRPC address from endpoint
        let channel = self.get_channel(&endpoint.grpc_addr).await?;
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

            eprintln!(
                "[GrpcTransport] Sending request for expert {} with {} sequences, tensor size {} bytes",
                req.expert_id,
                req.num_sequences,
                grpc_req.tensor.len()
            );

            // Send with timeout
            let response = tokio::time::timeout(self.timeout, client.forward(grpc_req))
                .await
                .map_err(|_| anyhow::anyhow!("Request timeout after {:?}", self.timeout))?
                .map_err(|e| anyhow::anyhow!("gRPC error for expert {}: {}", req.expert_id, e))?;

            // The response is a complete safetensors blob
            let output_tensor = response.into_inner().output_tensor;

            eprintln!(
                "[GrpcTransport] Received response for expert {}, size {} bytes",
                req.expert_id,
                output_tensor.len()
            );

            responses.push(ExpertResponse {
                expert_id: req.expert_id.clone(),
                tensor_data: output_tensor,
            });
        }

        Ok(responses)
    }

    fn transport_type(&self) -> TransportType {
        TransportType::Grpc
    }

    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool {
        self.get_channel(&endpoint.grpc_addr).await.is_ok()
    }
}
