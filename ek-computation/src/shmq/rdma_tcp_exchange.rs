use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use ibverbs::{QueuePairEndpoint, RemoteMemoryRegion};

use crate::controller::registry::{ShmqWorkerReq, ShmqWorkerResp};
use super::rdma_impl::RdmaQueue;

/// Connection information exchanged over TCP
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RdmaConnectionInfo {
    pub qp_endpoint: String,
    pub memory_region: String,
}

/// Pair of connection info for bidirectional communication
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RdmaConnectionPair {
    pub request_endpoint: RdmaConnectionInfo,
    pub response_endpoint: RdmaConnectionInfo,
}

/// TCP server for RDMA endpoint exchange (worker side)
pub struct RdmaEndpointServer {
    pub tcp_port: u16,
    req_queue: Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>,
    resp_queue: Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>,
}

impl RdmaEndpointServer {
    /// Create new RDMA endpoint server with queues
    pub fn new(
        req_queue: Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>,
        resp_queue: Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>,
    ) -> io::Result<Self> {
        // Bind to any available port
        let listener = TcpListener::bind("0.0.0.0:0")?;
        let tcp_port = listener.local_addr()?.port();
        drop(listener); // Release the port for actual use
        
        Ok(Self {
            tcp_port,
            req_queue,
            resp_queue,
        })
    }

    /// Get the TCP port for controller to connect
    pub fn port(&self) -> u16 {
        self.tcp_port
    }

    /// Start TCP server to handle RDMA endpoint exchange
    pub async fn start(&self) -> io::Result<()> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.tcp_port))?;
        log::info!("🌐 RDMA endpoint exchange server listening on port {}", self.tcp_port);

        loop {
            match listener.accept() {
                Ok((stream, addr)) => {
                    log::info!("📡 Controller connected from: {}", addr);
                    
                    let req_queue = self.req_queue.clone();
                    let resp_queue = self.resp_queue.clone();
                    
                    // Handle connection in background
                    tokio::task::spawn_blocking(move || {
                        if let Err(e) = Self::handle_connection(stream, req_queue, resp_queue) {
                            log::error!("Failed to handle RDMA endpoint exchange: {}", e);
                        }
                    });
                }
                Err(e) => {
                    log::error!("Failed to accept TCP connection: {}", e);
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    }

    /// Handle single TCP connection for endpoint exchange
    fn handle_connection(
        mut stream: TcpStream,
        req_queue: Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>,
        resp_queue: Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>,
    ) -> io::Result<()> {
        log::info!("🔄 Starting RDMA endpoint exchange");

        // Get worker's RDMA endpoints
        let worker_connection_pair = {
            let req_guard = req_queue.lock().unwrap();
            let resp_guard = resp_queue.lock().unwrap();
            
            let req_endpoint = req_guard.endpoint()?;
            let req_memory = req_guard.memory_region();
            let resp_endpoint = resp_guard.endpoint()?;
            let resp_memory = resp_guard.memory_region();

            RdmaConnectionPair {
                request_endpoint: RdmaConnectionInfo {
                    qp_endpoint: serde_json::to_string(&req_endpoint)
                        .map_err(|e| io::Error::other(format!("Failed to serialize req endpoint: {}", e)))?,
                    memory_region: serde_json::to_string(&req_memory)
                        .map_err(|e| io::Error::other(format!("Failed to serialize req memory: {}", e)))?,
                },
                response_endpoint: RdmaConnectionInfo {
                    qp_endpoint: serde_json::to_string(&resp_endpoint)
                        .map_err(|e| io::Error::other(format!("Failed to serialize resp endpoint: {}", e)))?,
                    memory_region: serde_json::to_string(&resp_memory)
                        .map_err(|e| io::Error::other(format!("Failed to serialize resp memory: {}", e)))?,
                },
            }
        };

        // Send worker's endpoints to controller
        let worker_info_json = serde_json::to_string(&worker_connection_pair)
            .map_err(|e| io::Error::other(format!("Failed to serialize worker info: {}", e)))?;
        writeln!(stream, "{}", worker_info_json)?;
        stream.flush()?;
        log::info!("📤 Sent worker RDMA endpoints to controller");

        // Receive controller's endpoints
        let mut reader = BufReader::new(&stream);
        let mut controller_info_line = String::new();
        reader.read_line(&mut controller_info_line)?;
        let controller_connection_pair: RdmaConnectionPair = serde_json::from_str(controller_info_line.trim())
            .map_err(|e| io::Error::other(format!("Failed to parse controller info: {}", e)))?;
        log::info!("📥 Received controller RDMA endpoints");

        // Parse controller endpoints
        let controller_req_endpoint: QueuePairEndpoint = serde_json::from_str(&controller_connection_pair.request_endpoint.qp_endpoint)
            .map_err(|e| io::Error::other(format!("Failed to parse controller req endpoint: {}", e)))?;
        let controller_req_memory: RemoteMemoryRegion = serde_json::from_str(&controller_connection_pair.request_endpoint.memory_region)
            .map_err(|e| io::Error::other(format!("Failed to parse controller req memory: {}", e)))?;
        let controller_resp_endpoint: QueuePairEndpoint = serde_json::from_str(&controller_connection_pair.response_endpoint.qp_endpoint)
            .map_err(|e| io::Error::other(format!("Failed to parse controller resp endpoint: {}", e)))?;
        let controller_resp_memory: RemoteMemoryRegion = serde_json::from_str(&controller_connection_pair.response_endpoint.memory_region)
            .map_err(|e| io::Error::other(format!("Failed to parse controller resp memory: {}", e)))?;

        // Establish RDMA connections
        log::info!("🔗 Establishing RDMA connections with controller");

        // Connect worker's request queue to controller's request queue (for receiving requests)
        {
            let mut req_queue_lock = req_queue.lock().unwrap();
            if !req_queue_lock.is_connected() {
                req_queue_lock.connect(controller_req_endpoint, controller_req_memory)?;
                log::info!("🚀 Worker request queue connected to controller");
            }
        }

        // Connect worker's response queue to controller's response queue (for sending responses)
        {
            let mut resp_queue_lock = resp_queue.lock().unwrap();
            if !resp_queue_lock.is_connected() {
                resp_queue_lock.connect(controller_resp_endpoint, controller_resp_memory)?;
                log::info!("🚀 Worker response queue connected to controller");
            }
        }

        // Signal successful connection
        let mut stream = reader.into_inner();
        writeln!(stream, "RDMA_READY")?;
        stream.flush()?;

        // Wait for controller ready signal
        let mut final_reader = BufReader::new(stream);
        let mut ready_line = String::new();
        final_reader.read_line(&mut ready_line)?;
        if ready_line.trim() != "RDMA_READY" {
            return Err(io::Error::other("Controller not ready"));
        }

        log::info!("✅ RDMA bidirectional connection established successfully");
        
        // Update global connection status
        crate::worker::update_rdma_connection_status(true);
        
        Ok(())
    }
}

/// TCP client for RDMA endpoint exchange (controller side)
pub struct RdmaEndpointClient;

impl RdmaEndpointClient {
    /// Connect to worker and exchange RDMA endpoints
    pub async fn connect_and_exchange(
        worker_host: &str,
        worker_tcp_port: u16,
        controller_req_queue: Arc<tokio::sync::Mutex<RdmaQueue<ShmqWorkerReq>>>,
        controller_resp_queue: Arc<tokio::sync::Mutex<RdmaQueue<ShmqWorkerResp>>>,
    ) -> io::Result<()> {
        let worker_addr = format!("{}:{}", worker_host, worker_tcp_port);
        log::info!("🔗 Connecting to worker at {} for RDMA endpoint exchange", worker_addr);

        let mut stream = TcpStream::connect(&worker_addr)?;
        log::info!("✅ Connected to worker TCP server");

        // Get controller's RDMA endpoints
        let controller_connection_pair = {
            let req_guard = controller_req_queue.lock().await;
            let resp_guard = controller_resp_queue.lock().await;
            
            let req_endpoint = req_guard.endpoint()?;
            let req_memory = req_guard.memory_region();
            let resp_endpoint = resp_guard.endpoint()?;
            let resp_memory = resp_guard.memory_region();

            RdmaConnectionPair {
                request_endpoint: RdmaConnectionInfo {
                    qp_endpoint: serde_json::to_string(&req_endpoint)
                        .map_err(|e| io::Error::other(format!("Failed to serialize req endpoint: {}", e)))?,
                    memory_region: serde_json::to_string(&req_memory)
                        .map_err(|e| io::Error::other(format!("Failed to serialize req memory: {}", e)))?,
                },
                response_endpoint: RdmaConnectionInfo {
                    qp_endpoint: serde_json::to_string(&resp_endpoint)
                        .map_err(|e| io::Error::other(format!("Failed to serialize resp endpoint: {}", e)))?,
                    memory_region: serde_json::to_string(&resp_memory)
                        .map_err(|e| io::Error::other(format!("Failed to serialize resp memory: {}", e)))?,
                },
            }
        };

        // Receive worker's endpoints first
        let mut reader = BufReader::new(&stream);
        let mut worker_info_line = String::new();
        reader.read_line(&mut worker_info_line)?;
        let worker_connection_pair: RdmaConnectionPair = serde_json::from_str(worker_info_line.trim())
            .map_err(|e| io::Error::other(format!("Failed to parse worker info: {}", e)))?;
        log::info!("📥 Received worker RDMA endpoints");

        // Send controller's endpoints to worker
        let mut stream = reader.into_inner();
        let controller_info_json = serde_json::to_string(&controller_connection_pair)
            .map_err(|e| io::Error::other(format!("Failed to serialize controller info: {}", e)))?;
        writeln!(stream, "{}", controller_info_json)?;
        stream.flush()?;
        log::info!("📤 Sent controller RDMA endpoints to worker");

        // Parse worker endpoints
        let worker_req_endpoint: QueuePairEndpoint = serde_json::from_str(&worker_connection_pair.request_endpoint.qp_endpoint)
            .map_err(|e| io::Error::other(format!("Failed to parse worker req endpoint: {}", e)))?;
        let worker_req_memory: RemoteMemoryRegion = serde_json::from_str(&worker_connection_pair.request_endpoint.memory_region)
            .map_err(|e| io::Error::other(format!("Failed to parse worker req memory: {}", e)))?;
        let worker_resp_endpoint: QueuePairEndpoint = serde_json::from_str(&worker_connection_pair.response_endpoint.qp_endpoint)
            .map_err(|e| io::Error::other(format!("Failed to parse worker resp endpoint: {}", e)))?;
        let worker_resp_memory: RemoteMemoryRegion = serde_json::from_str(&worker_connection_pair.response_endpoint.memory_region)
            .map_err(|e| io::Error::other(format!("Failed to parse worker resp memory: {}", e)))?;

        // Establish RDMA connections
        log::info!("🔗 Establishing RDMA connections with worker");

        // Connect controller's request queue to worker's request queue (for sending requests)
        {
            let mut req_queue_lock = controller_req_queue.lock().await;
            if !req_queue_lock.is_connected() {
                req_queue_lock.connect(worker_req_endpoint, worker_req_memory)?;
                log::info!("🚀 Controller request queue connected to worker");
            }
        }

        // Connect controller's response queue to worker's response queue (for receiving responses)
        {
            let mut resp_queue_lock = controller_resp_queue.lock().await;
            if !resp_queue_lock.is_connected() {
                resp_queue_lock.connect(worker_resp_endpoint, worker_resp_memory)?;
                log::info!("🚀 Controller response queue connected to worker");
            }
        }

        // Wait for worker ready signal
        let mut ready_reader = BufReader::new(stream);
        let mut ready_line = String::new();
        ready_reader.read_line(&mut ready_line)?;
        if ready_line.trim() != "RDMA_READY" {
            return Err(io::Error::other("Worker not ready"));
        }

        // Signal controller ready
        let mut stream = ready_reader.into_inner();
        writeln!(stream, "RDMA_READY")?;
        stream.flush()?;

        log::info!("✅ RDMA bidirectional connection established successfully");
        
        Ok(())
    }
}