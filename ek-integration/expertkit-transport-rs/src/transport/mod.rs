use anyhow::Result;
use async_trait::async_trait;

pub mod auto;
pub mod grpc;
pub mod mock;
pub mod shm;

#[derive(Debug, Clone)]
pub enum TransportType {
    Grpc,
    SharedMemory,
    Nvshmem,
}

/// Request for a single expert computation
/// Each request represents ONE expert, but can contain MULTIPLE sequences
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

    /// Create a single-sequence request (for compatibility)
    pub fn single(expert_id: String, tensor_data: Vec<u8>) -> Self {
        Self {
            expert_id,
            tensor_data,
            num_sequences: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertResponse {
    pub expert_id: String,
    pub tensor_data: Vec<u8>, // Safetensors blob with output sequences
}

/// Transport abstraction for expert communication
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send batch of expert requests to a worker
    /// Note: Each ExpertRequest already contains batched sequences for that expert
    /// The transport should send one computation call per ExpertRequest (not concatenate)
    async fn send_batch(
        &self,
        endpoint: &str,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>>;

    /// Get transport type
    fn transport_type(&self) -> TransportType;

    /// Check if transport is available for endpoint
    async fn is_available(&self, endpoint: &str) -> bool;
}
