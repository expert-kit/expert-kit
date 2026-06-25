mod queue;

pub use queue::{ShmQueue, ShmQueueError, ShmqWorkerReq, ShmqWorkerResp};

use super::*;
use dashmap::DashMap;
use log::debug;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use parking_lot::Mutex;

const MAX_TENSOR_SIZE: usize = 64 * 1024 * 1024; // 64 MB
const REQ_CAPACITY: usize = 8 + 64 + 8 + MAX_TENSOR_SIZE;
const RESP_CAPACITY: usize = 8 + 8 + MAX_TENSOR_SIZE;

type PendingResponses = HashMap<usize, ShmqWorkerResp>;
type SharedPendingResponses = Arc<Mutex<PendingResponses>>;
type PendingResponseCache = Arc<DashMap<String, SharedPendingResponses>>;

/// Shared memory transport for local workers
pub struct ShmTransport {
    /// Cache of request queues
    req_connections: Arc<DashMap<String, Arc<Mutex<ShmQueue>>>>,
    /// Cache of response queues
    resp_connections: Arc<DashMap<String, Arc<Mutex<ShmQueue>>>>,
    /// Pending responses cache
    /// Maps worker endpoint -> (response_id -> response)
    pending_responses: PendingResponseCache,
    /// Timeout for operations
    timeout: std::time::Duration,
}

impl ShmTransport {
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            req_connections: Arc::new(DashMap::new()),
            resp_connections: Arc::new(DashMap::new()),
            pending_responses: Arc::new(DashMap::new()),
            timeout: std::time::Duration::from_secs_f64(timeout_sec),
        }
    }

    /// Get or create pending response map for a worker
    fn get_pending_map(&self, endpoint: &str) -> SharedPendingResponses {
        self.pending_responses
            .entry(endpoint.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(HashMap::new())))
            .value()
            .clone()
    }

    /// Discover worker queue by scanning /dev/shm
    async fn discover_worker(&self, endpoint: &str) -> Result<(ShmQueue, ShmQueue)> {
        // Parse endpoint to extract worker identifier
        let worker_id = endpoint;

        // Worker creates queues with format: ek-shmq-{req,resp}-{worker_id}
        let req_name = format!("ek-shmq-req-{}", worker_id);
        let resp_name = format!("ek-shmq-resp-{}", worker_id);

        debug!(
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

        debug!("[ShmTransport] Found worker queues, opening...");

        // Open queues
        let req_queue = ShmQueue::open(&req_name, 16, REQ_CAPACITY)
            .ok_or_else(|| anyhow::anyhow!("Failed to open request queue: {}", req_name))?;

        let resp_queue = ShmQueue::open(&resp_name, 16, RESP_CAPACITY)
            .ok_or_else(|| anyhow::anyhow!("Failed to open response queue: {}", resp_name))?;

        debug!("[ShmTransport] Successfully opened worker queues");

        Ok((req_queue, resp_queue))
    }

    /// Get or create request queue for sending
    async fn get_req_queue(&self, endpoint: &str) -> Result<Arc<Mutex<ShmQueue>>> {
        // Fast path: check if request queue exists
        if let Some(queue) = self.req_connections.get(endpoint) {
            return Ok(queue.value().clone());
        }

        // Slow path: discover worker and create both queues
        let (req_queue, resp_queue) = self.discover_worker(endpoint).await?;

        let req_arc = Arc::new(Mutex::new(req_queue));
        let resp_arc = Arc::new(Mutex::new(resp_queue));

        // Insert both queues
        self.req_connections
            .entry(endpoint.to_string())
            .or_insert_with(|| req_arc.clone());
        self.resp_connections
            .entry(endpoint.to_string())
            .or_insert(resp_arc);

        // Return the inserted value (in case another thread won)
        Ok(self
            .req_connections
            .get(endpoint)
            .expect("Just inserted")
            .value()
            .clone())
    }

    /// Get or create response queue for receiving
    async fn get_resp_queue(&self, endpoint: &str) -> Result<Arc<Mutex<ShmQueue>>> {
        // Fast path: check if response queue exists
        if let Some(queue) = self.resp_connections.get(endpoint) {
            return Ok(queue.value().clone());
        }

        // Slow path: discover worker and create both queues
        let (req_queue, resp_queue) = self.discover_worker(endpoint).await?;

        let req_arc = Arc::new(Mutex::new(req_queue));
        let resp_arc = Arc::new(Mutex::new(resp_queue));

        // Insert both queues
        self.req_connections
            .entry(endpoint.to_string())
            .or_insert(req_arc);
        self.resp_connections
            .entry(endpoint.to_string())
            .or_insert_with(|| resp_arc.clone());

        // Return the inserted value
        Ok(self
            .resp_connections
            .get(endpoint)
            .expect("Just inserted")
            .value()
            .clone())
    }
}

#[async_trait]
impl Transport for ShmTransport {
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        let queue_prefix = &endpoint.shm_queue_prefix;

        // Get queue handles (separate locks!)
        let req_queue_arc = self.get_req_queue(queue_prefix).await?;
        let resp_queue_arc = self.get_resp_queue(queue_prefix).await?;
        let pending_map = self.get_pending_map(queue_prefix);

        // Send all requests - only locks req_queue
        let mut request_ids = Vec::new();
        {
            let mut req_queue = req_queue_arc.lock();
            for req in &requests {
                let shm_req = ShmqWorkerReq::new(&req.expert_id, &req.tensor_data);
                let req_id = shm_req.id;

                // Send request
                req_queue
                    .send(&shm_req)
                    .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

                request_ids.push(req_id);
            }
        }

        // Collect responses
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

            // Check if responses are in the pending map
            {
                let mut pending = pending_map.lock();
                for &req_id in &request_ids {
                    if let Some(resp) = pending.remove(&req_id) {
                        responses_map.insert(req_id, resp);
                    }
                }
            }

            // If we have all responses, break early
            if responses_map.len() == requests.len() {
                break;
            }

            // Try to receive response (parking_lot mutex is fast enough for direct use)
            let recv_result = {
                let mut resp_queue = resp_queue_arc.lock();
                resp_queue.recv::<ShmqWorkerResp>()
            };

            match recv_result {
                Ok(resp) => {
                    let resp_id = resp.id;
                    // Check if this is one of our responses
                    if request_ids.contains(&resp_id) {
                        responses_map.insert(resp_id, resp);
                    } else {
                        let mut pending = pending_map.lock();
                        pending.insert(resp_id, resp);
                    }
                }
                Err(ShmQueueError::Empty) => {
                    // Yield to allow other tasks to run
                    tokio::task::yield_now().await;
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
