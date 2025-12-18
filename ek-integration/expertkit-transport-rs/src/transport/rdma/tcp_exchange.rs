use ibverbs::{QueuePairEndpoint, RemoteMemoryRegion};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;

use super::queue::{RdmaQueue, ShmqWorkerReq, ShmqWorkerResp};

/// Connection information exchanged over TCP
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RdmaConnectionInfo {
    pub qp_endpoint: String,   // Serialized QueuePairEndpoint
    pub memory_region: String, // Serialized RemoteMemoryRegion
}

/// Pair of connection info for bidirectional communication
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RdmaConnectionPair {
    pub request_endpoint: RdmaConnectionInfo, // For request queue
    pub response_endpoint: RdmaConnectionInfo, // For response queue
}

/// Connect to worker and exchange RDMA endpoints (client-side)
pub fn connect_and_exchange(
    worker_host: &str,
    worker_tcp_port: u16,
    req_queue: &mut RdmaQueue<ShmqWorkerReq>,
    resp_queue: &mut RdmaQueue<ShmqWorkerResp>,
) -> io::Result<()> {
    let worker_addr = format!("{}:{}", worker_host, worker_tcp_port);
    log::info!(
        "Connecting to worker at {} for RDMA endpoint exchange",
        worker_addr
    );

    // Connect to worker's TCP server
    let stream = TcpStream::connect(&worker_addr)?;
    log::debug!("Connected to worker TCP server");

    // Prepare client's RDMA endpoints
    let client_connection_pair = RdmaConnectionPair {
        request_endpoint: RdmaConnectionInfo {
            qp_endpoint: serde_json::to_string(&req_queue.endpoint()?).map_err(|e| {
                io::Error::other(format!("Failed to serialize req endpoint: {}", e))
            })?,
            memory_region: serde_json::to_string(&req_queue.memory_region())
                .map_err(|e| io::Error::other(format!("Failed to serialize req memory: {}", e)))?,
        },
        response_endpoint: RdmaConnectionInfo {
            qp_endpoint: serde_json::to_string(&resp_queue.endpoint()?).map_err(|e| {
                io::Error::other(format!("Failed to serialize resp endpoint: {}", e))
            })?,
            memory_region: serde_json::to_string(&resp_queue.memory_region())
                .map_err(|e| io::Error::other(format!("Failed to serialize resp memory: {}", e)))?,
        },
    };

    // Receive worker's endpoints first
    let mut reader = BufReader::new(&stream);
    let mut worker_info_line = String::new();
    reader.read_line(&mut worker_info_line)?;
    let worker_connection_pair: RdmaConnectionPair = serde_json::from_str(worker_info_line.trim())
        .map_err(|e| io::Error::other(format!("Failed to parse worker info: {}", e)))?;
    log::debug!("Received worker RDMA endpoints");

    // Send client's endpoints to worker
    let mut stream = reader.into_inner();
    let client_info_json = serde_json::to_string(&client_connection_pair)
        .map_err(|e| io::Error::other(format!("Failed to serialize client info: {}", e)))?;
    writeln!(stream, "{}", client_info_json)?;
    stream.flush()?;
    log::debug!("Sent client RDMA endpoints to worker");

    // Parse worker endpoints
    let worker_req_endpoint: QueuePairEndpoint =
        serde_json::from_str(&worker_connection_pair.request_endpoint.qp_endpoint)
            .map_err(|e| io::Error::other(format!("Failed to parse worker req endpoint: {}", e)))?;
    let worker_req_memory: RemoteMemoryRegion =
        serde_json::from_str(&worker_connection_pair.request_endpoint.memory_region)
            .map_err(|e| io::Error::other(format!("Failed to parse worker req memory: {}", e)))?;
    let worker_resp_endpoint: QueuePairEndpoint = serde_json::from_str(
        &worker_connection_pair.response_endpoint.qp_endpoint,
    )
    .map_err(|e| io::Error::other(format!("Failed to parse worker resp endpoint: {}", e)))?;
    let worker_resp_memory: RemoteMemoryRegion =
        serde_json::from_str(&worker_connection_pair.response_endpoint.memory_region)
            .map_err(|e| io::Error::other(format!("Failed to parse worker resp memory: {}", e)))?;

    // Establish RDMA connections
    log::debug!("Establishing RDMA connections with worker");

    // Connect request queue (client sends requests to worker)
    if !req_queue.is_connected() {
        req_queue.connect(worker_req_endpoint, worker_req_memory)?;
        log::debug!("Request queue connected to worker");
    }

    // Connect response queue (client receives responses from worker)
    if !resp_queue.is_connected() {
        resp_queue.connect(worker_resp_endpoint, worker_resp_memory)?;
        log::debug!("Response queue connected to worker");
    }

    // Synchronize readiness
    // Wait for worker "RDMA_READY" signal
    let mut ready_reader = BufReader::new(stream);
    let mut ready_line = String::new();
    ready_reader.read_line(&mut ready_line)?;
    if ready_line.trim() != "RDMA_READY" {
        return Err(io::Error::other("Worker not ready"));
    }

    // Send client "RDMA_READY" signal
    let mut stream = ready_reader.into_inner();
    writeln!(stream, "RDMA_READY")?;
    stream.flush()?;

    log::info!("RDMA bidirectional connection established successfully");

    Ok(())
}
