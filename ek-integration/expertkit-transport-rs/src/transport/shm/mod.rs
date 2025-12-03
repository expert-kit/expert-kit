mod queue;

pub use queue::{ShmQueue, ShmQueueError, ShmqWorkerReq, ShmqWorkerResp};

use super::*;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

const MAX_TENSOR_SIZE: usize = 64 * 1024 * 1024; // 64 MB
const REQ_CAPACITY: usize = 8 + 64 + 8 + MAX_TENSOR_SIZE;
const RESP_CAPACITY: usize = 8 + 8 + MAX_TENSOR_SIZE;

/// Shared memory transport for local workers
pub struct ShmTransport {
    /// Cache of opened request/response queue pairs
    connections: Arc<Mutex<HashMap<String, (ShmQueue, ShmQueue)>>>,
    /// Timeout for operations
    timeout: std::time::Duration,
}

impl ShmTransport {
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            connections: Arc::new(Mutex::new(HashMap::new())),
            timeout: std::time::Duration::from_secs_f64(timeout_sec),
        }
    }

    /// Discover worker queue by scanning /dev/shm
    async fn discover_worker(&self, endpoint: &str) -> Result<(ShmQueue, ShmQueue)> {
        // Parse endpoint to extract worker identifier
        let worker_id = endpoint;

        // Worker creates queues with format: ek-shmq-{req,resp}-{worker_id}
        let req_name = format!("ek-shmq-req-{}", worker_id);
        let resp_name = format!("ek-shmq-resp-{}", worker_id);

        eprintln!(
            "[ShmTransport] Looking for worker queues: {}, {}",
            req_name, resp_name
        );

        // Try to find existing queues in /dev/shm
        let shm_dir = Path::new("/dev/shm");
        let req_path = shm_dir.join(&req_name);
        let resp_path = shm_dir.join(&resp_name);

        if !req_path.exists() || !resp_path.exists() {
            return Err(anyhow::anyhow!(
                "Worker queues not found in /dev/shm: {} (exists: {}), {} (exists: {}). \
                 Worker may not have started yet, or channel type may be wrong.",
                req_name,
                req_path.exists(),
                resp_name,
                resp_path.exists()
            ));
        }

        eprintln!("[ShmTransport] Found worker queues, opening...");

        // Open queues (don't create - worker creates them!)
        let req_queue = ShmQueue::open(&req_name, 16, REQ_CAPACITY)
            .ok_or_else(|| anyhow::anyhow!("Failed to open request queue: {}", req_name))?;

        let resp_queue = ShmQueue::open(&resp_name, 16, RESP_CAPACITY)
            .ok_or_else(|| anyhow::anyhow!("Failed to open response queue: {}", resp_name))?;

        eprintln!("[ShmTransport] Successfully opened worker queues");

        Ok((req_queue, resp_queue))
    }

    /// Get or create connection to worker
    async fn get_connection(&self, endpoint: &str) -> Result<()> {
        let mut connections = self.connections.lock().await;

        if !connections.contains_key(endpoint) {
            let (req_queue, resp_queue) = self.discover_worker(endpoint).await?;
            connections.insert(endpoint.to_string(), (req_queue, resp_queue));
        }

        Ok(())
    }
}

#[async_trait]
impl Transport for ShmTransport {
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Extract queue prefix from endpoint
        let queue_prefix = &endpoint.shm_queue_prefix;

        // Ensure connection exists
        self.get_connection(queue_prefix).await?;

        let mut connections = self.connections.lock().await;
        let (req_queue, resp_queue) = connections
            .get_mut(queue_prefix)
            .ok_or_else(|| anyhow::anyhow!("Connection not found for {}", queue_prefix))?;

        // Send all requests
        let mut request_ids = Vec::new();
        for req in &requests {
            let shm_req = ShmqWorkerReq::new(&req.expert_id, &req.tensor_data);
            let req_id = shm_req.id;

            // Send request
            req_queue
                .send(&shm_req)
                .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

            request_ids.push(req_id);
        }

        // Collect responses (may arrive out of order)
        let mut responses_map: HashMap<usize, ShmqWorkerResp> = HashMap::new();
        let start = std::time::Instant::now();

        while responses_map.len() < requests.len() {
            if start.elapsed() > self.timeout {
                return Err(anyhow::anyhow!(
                    "Timeout waiting for responses: got {}/{} responses",
                    responses_map.len(),
                    requests.len()
                ));
            }

            // Try to receive response
            match resp_queue.recv::<ShmqWorkerResp>() {
                Ok(resp) => {
                    // Check if this is one of our responses
                    if request_ids.contains(&resp.id) {
                        responses_map.insert(resp.id, resp);
                    }
                }
                Err(ShmQueueError::Empty) => {
                    // tokio::time::sleep(tokio::time::Duration::from_micros(1)).await;
                    continue;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Failed to receive response: {}", e));
                }
            }
        }

        // Build response vector in request order
        let responses: Vec<ExpertResponse> = request_ids
            .into_iter()
            .zip(requests.iter())
            .map(|(req_id, req)| {
                let resp = responses_map
                    .get(&req_id)
                    .expect("Response should exist for all requests");

                ExpertResponse {
                    expert_id: req.expert_id.clone(),
                    tensor_data: resp.output_tensor.clone(),
                }
            })
            .collect();

        Ok(responses)
    }

    fn transport_type(&self) -> TransportType {
        TransportType::SharedMemory
    }

    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool {
        self.discover_worker(&endpoint.shm_queue_prefix)
            .await
            .is_ok()
    }
}
