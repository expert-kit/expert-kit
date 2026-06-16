use anyhow::Result;
use async_trait::async_trait;

pub mod auto;
pub mod grpc;
pub mod shm;

#[cfg(feature = "rdma")]
pub mod rdma;

// Re-export WorkerEndpoint from grpc proto
pub use grpc::proto::ek::control::v1::WorkerEndpoint;

#[derive(Debug, Clone)]
#[allow(unused)]
pub enum TransportType {
    Grpc,
    SharedMemory,
    Nvshmem,
}

/// Request for a single expert computation
#[derive(Debug, Clone)]
pub struct ExpertRequest {
    pub expert_id: String,
    pub tensor_data: Vec<u8>, // Safetensors blob containing batched sequences
    pub num_sequences: usize, // Number of sequences in this batch
}

impl ExpertRequest {
    /// Create a new expert request with a batched tensor
    pub fn new(expert_id: String, tensor_data: Vec<u8>, num_sequences: usize) -> Self {
        Self {
            expert_id,
            tensor_data,
            num_sequences,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertResponse {
    #[allow(unused)]
    pub expert_id: String,
    pub tensor_data: Vec<u8>, // Safetensors blob with output sequences
}

/// Transport abstraction for expert communication
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send batch of expert requests to a worker
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>>;

    /// Get transport type
    #[allow(unused)]
    fn transport_type(&self) -> TransportType;

    /// Check if transport is available for endpoint
    #[allow(unused)]
    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool;
}
