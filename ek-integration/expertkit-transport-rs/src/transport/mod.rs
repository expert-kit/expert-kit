use anyhow::Result;
use async_trait::async_trait;

pub mod auto;
pub mod grpc;
pub mod shm;

// Re-export WorkerEndpoint from grpc proto
pub use grpc::proto::ek::control::v1::WorkerEndpoint;

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
    ///
    /// # Arguments
    /// * `endpoint` - Worker endpoint information (includes channel type, addresses, etc.)
    /// * `requests` - Batch of expert requests to send
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>>;

    /// Get transport type
    #[allow(dead_code)]
    fn transport_type(&self) -> TransportType;

    /// Check if transport is available for endpoint
    #[allow(dead_code)]
    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool;
}
