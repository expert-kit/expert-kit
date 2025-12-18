mod queue;
mod tcp_exchange;

pub use queue::{RdmaQueue, RdmaQueueError, ShmqWorkerReq, ShmqWorkerResp};

use super::*;
use dashmap::DashMap;
use log::{debug, info};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;

/// RDMA transport for high-performance remote workers
#[derive(Clone)]
pub struct RdmaTransport {
    /// Cache of request queues (endpoint -> queue)
    req_connections: Arc<DashMap<String, Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>>>,
    /// Cache of response queues (endpoint -> queue)
    resp_connections: Arc<DashMap<String, Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>>>,
    /// Pending responses cache (endpoint -> (response_id -> response))
    pending_responses: Arc<DashMap<String, Arc<Mutex<HashMap<usize, ShmqWorkerResp>>>>>,
    /// Maps worker endpoint -> mutex that guards connection establishment
    connection_locks: Arc<DashMap<String, Arc<TokioMutex<()>>>>,
    /// Timeout for operations
    timeout: std::time::Duration,
}

impl RdmaTransport {
    /// Create a new RDMA transport
    pub fn new(timeout_sec: f64) -> Self {
        Self {
            req_connections: Arc::new(DashMap::new()),
            resp_connections: Arc::new(DashMap::new()),
            pending_responses: Arc::new(DashMap::new()),
            connection_locks: Arc::new(DashMap::new()),
            timeout: std::time::Duration::from_secs_f64(timeout_sec),
        }
    }

    /// Get or create pending response map for a worker
    fn get_pending_map(&self, endpoint: &str) -> Arc<Mutex<HashMap<usize, ShmqWorkerResp>>> {
        self.pending_responses
            .entry(endpoint.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(HashMap::new())))
            .value()
            .clone()
    }

    /// Extract host from gRPC address (format: "http://host:port" or "host:port")
    fn extract_host(grpc_addr: &str) -> Result<String> {
        let addr = grpc_addr
            .strip_prefix("http://")
            .or_else(|| grpc_addr.strip_prefix("https://"))
            .unwrap_or(grpc_addr);

        let host = addr
            .split(':')
            .next()
            .ok_or_else(|| anyhow::anyhow!("Invalid gRPC address: {}", grpc_addr))?;

        Ok(host.to_string())
    }

    /// Connect to worker and establish RDMA connection
    async fn connect_to_worker(
        &self,
        worker_host: &str,
        rdma_tcp_port: u16,
    ) -> Result<(RdmaQueue<ShmqWorkerReq>, RdmaQueue<ShmqWorkerResp>)> {
        info!(
            "[RdmaTransport] Establishing RDMA connection to {}:{}",
            worker_host, rdma_tcp_port
        );

        // Create RDMA queues
        let mut req_queue = tokio::task::spawn_blocking(move || {
            RdmaQueue::<ShmqWorkerReq>::new(
                None, // Auto-select RDMA device
                16,   // Queue capacity
                true, // is_sender = true
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to spawn queue creation task: {}", e))?
        .map_err(|e| anyhow::anyhow!("Failed to create request queue: {}", e))?;

        let mut resp_queue = tokio::task::spawn_blocking(move || {
            RdmaQueue::<ShmqWorkerResp>::new(
                None,  // Auto-select RDMA device
                16,    // Queue capacity
                false, // is_sender = false
            )
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to spawn queue creation task: {}", e))?
        .map_err(|e| anyhow::anyhow!("Failed to create response queue: {}", e))?;

        // Perform TCP endpoint exchange and RDMA connection
        let worker_host = worker_host.to_string();
        tokio::task::spawn_blocking(move || {
            tcp_exchange::connect_and_exchange(
                &worker_host,
                rdma_tcp_port,
                &mut req_queue,
                &mut resp_queue,
            )
            .map(|_| (req_queue, resp_queue))
        })
        .await
        .map_err(|e| anyhow::anyhow!("Failed to spawn TCP exchange task: {}", e))?
        .map_err(|e| anyhow::anyhow!("RDMA connection failed: {}", e))
    }

    /// Get or create request queue for sending
    async fn get_req_queue(
        &self,
        worker_host: &str,
        rdma_tcp_port: u16,
    ) -> Result<Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>> {
        let cache_key = format!("{}:{}", worker_host, rdma_tcp_port);

        // Fast path: check if request queue exists
        if let Some(queue) = self.req_connections.get(&cache_key) {
            debug!("[RdmaTransport] Reusing existing RDMA connection");
            return Ok(queue.value().clone());
        }

        // Get or create connection lock for this worker
        let lock = self
            .connection_locks
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(TokioMutex::new(())))
            .value()
            .clone();

        // Acquire lock - only ONE task can establish connection
        let _guard = lock.lock().await;

        // Double-check cache
        if let Some(queue) = self.req_connections.get(&cache_key) {
            debug!("[RdmaTransport] Connection established by another task");
            return Ok(queue.value().clone());
        }

        // Establish new RDMA connection
        info!(
            "[RdmaTransport] Establishing new RDMA connection to {}",
            cache_key
        );
        let (req_queue, resp_queue) = self.connect_to_worker(worker_host, rdma_tcp_port).await?;

        let req_arc = Arc::new(Mutex::new(req_queue));
        let resp_arc = Arc::new(Mutex::new(resp_queue));

        // Insert both queues atomically
        self.req_connections
            .insert(cache_key.clone(), req_arc.clone());
        self.resp_connections.insert(cache_key.clone(), resp_arc);

        Ok(req_arc)
    }

    /// Get or create response queue for receiving
    async fn get_resp_queue(
        &self,
        worker_host: &str,
        rdma_tcp_port: u16,
    ) -> Result<Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>> {
        let cache_key = format!("{}:{}", worker_host, rdma_tcp_port);

        // Fast path: check if response queue exists
        if let Some(queue) = self.resp_connections.get(&cache_key) {
            return Ok(queue.value().clone());
        }

        // Get or create connection lock for this worker
        let lock = self
            .connection_locks
            .entry(cache_key.clone())
            .or_insert_with(|| Arc::new(TokioMutex::new(())))
            .value()
            .clone();

        // Acquire lock
        let _guard = lock.lock().await;

        // Double-check cache
        if let Some(queue) = self.resp_connections.get(&cache_key) {
            return Ok(queue.value().clone());
        }

        // Establish new RDMA connection
        info!(
            "[RdmaTransport] Establishing new RDMA connection to {}",
            cache_key
        );
        let (req_queue, resp_queue) = self.connect_to_worker(worker_host, rdma_tcp_port).await?;

        let req_arc = Arc::new(Mutex::new(req_queue));
        let resp_arc = Arc::new(Mutex::new(resp_queue));

        // Insert both queues atomically
        self.req_connections.insert(cache_key.clone(), req_arc);
        self.resp_connections
            .insert(cache_key.clone(), resp_arc.clone());

        Ok(resp_arc)
    }

    /// Warm up connection to a specific worker
    pub async fn warmup_connection(&self, endpoint: &WorkerEndpoint) -> Result<()> {
        let worker_host = Self::extract_host(&endpoint.grpc_addr)?;
        let rdma_tcp_port = endpoint.rdma_tcp_port as u16;

        info!(
            "[RdmaTransport] Warming up connection to {}:{}",
            worker_host, rdma_tcp_port
        );

        // Establish both request and response queues
        let _ = self.get_req_queue(&worker_host, rdma_tcp_port).await?;
        let _ = self.get_resp_queue(&worker_host, rdma_tcp_port).await?;

        info!(
            "[RdmaTransport] Connection warmed up for {}:{}",
            worker_host, rdma_tcp_port
        );

        Ok(())
    }
}

#[async_trait]
impl Transport for RdmaTransport {
    async fn send_batch(
        &self,
        endpoint: &WorkerEndpoint,
        requests: Vec<ExpertRequest>,
    ) -> Result<Vec<ExpertResponse>> {
        // Extract worker host and RDMA TCP port from endpoint
        let worker_host = Self::extract_host(&endpoint.grpc_addr)?;
        let rdma_tcp_port = endpoint.rdma_tcp_port as u16;
        let cache_key = format!("{}:{}", worker_host, rdma_tcp_port);

        // Get queue handles
        let req_queue_arc = self.get_req_queue(&worker_host, rdma_tcp_port).await?;
        let resp_queue_arc = self.get_resp_queue(&worker_host, rdma_tcp_port).await?;
        let pending_map = self.get_pending_map(&cache_key);

        // Send all requests - only locks req_queue
        let mut request_ids = Vec::new();
        for req in &requests {
            let shm_req = ShmqWorkerReq::new(&req.expert_id, &req.tensor_data);
            let req_id = shm_req.id();

            // Send request via RDMA
            let req_queue_clone = req_queue_arc.clone();
            let shm_req_clone = shm_req.clone();
            tokio::task::spawn_blocking(move || {
                let mut req_queue = req_queue_clone.lock();
                req_queue.send(&shm_req_clone)
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to spawn send task: {}", e))?
            .map_err(|e| anyhow::anyhow!("Failed to send request via RDMA: {}", e))?;

            request_ids.push(req_id);
        }

        // Collect responses
        let mut responses_map: HashMap<usize, ShmqWorkerResp> = HashMap::new();
        let start = std::time::Instant::now();

        while responses_map.len() < requests.len() {
            if start.elapsed() > self.timeout {
                return Err(anyhow::anyhow!(
                    "Timeout waiting for responses: got {}/{} responses after {:?}",
                    responses_map.len(),
                    requests.len(),
                    start.elapsed()
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

            // Try to receive response
            let resp_queue_clone = resp_queue_arc.clone();
            let recv_result = tokio::task::spawn_blocking(move || {
                let mut resp_queue = resp_queue_clone.lock();
                resp_queue.recv()
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to spawn recv task: {}", e))?;

            match recv_result {
                Ok(resp) => {
                    let resp_id = resp.id();
                    // Check if this is one of our responses
                    if request_ids.contains(&resp_id) {
                        responses_map.insert(resp_id, resp);
                    } else {
                        // Response for a different request, cache it
                        let mut pending = pending_map.lock();
                        pending.insert(resp_id, resp);
                    }
                }
                Err(RdmaQueueError::Empty) => {
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
                    tensor_data: resp.output_tensor().to_vec(),
                }
            })
            .collect();

        Ok(responses)
    }

    fn transport_type(&self) -> TransportType {
        TransportType::SharedMemory // RDMA is logically similar to shared memory
    }

    async fn is_available(&self, endpoint: &WorkerEndpoint) -> bool {
        let worker_host = match Self::extract_host(&endpoint.grpc_addr) {
            Ok(host) => host,
            Err(_) => return false,
        };

        let rdma_tcp_port = endpoint.rdma_tcp_port as u16;

        // Try to connect to worker's TCP port
        let addr = format!("{}:{}", worker_host, rdma_tcp_port);
        std::net::TcpStream::connect_timeout(
            &addr.parse().unwrap(),
            std::time::Duration::from_secs(2),
        )
        .is_ok()
    }
}
