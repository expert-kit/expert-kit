use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use crate::{
    shmq::{GeneralShmQueueBytes, RdmaEndpointClient, ShmQueue, rdma_impl::RdmaQueue},
    state::io::StateReaderImpl,
};
use ek_base::{
    error::{EKError, EKResult},
    tracing::grpc::OTelGrpcClientMiddleware,
};
use ndarray_rand::rand;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tonic::transport::Channel;
use tower::ServiceBuilder;
use url::Url;

const MAX_RETRIES: u32 = 3;
const BASE_DELAY_MS: u64 = 1000;

/// When true, force controller to use gRPC for all worker connections.
/// Set to false to respect each worker's configured channel (shm/rdma/grpc).
const FORCE_GRPC_CHANNEL: bool = false;

pub type ExpertId = String;
pub type ExpertIdRef<'a> = &'a str;

pub type LocalShmChannel = (
    Arc<Mutex<ShmQueue<'static, ShmqWorkerReq>>>,
    Arc<Mutex<ShmQueue<'static, ShmqWorkerResp>>>,
);

pub type RdmaChannel = (
    Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>,
    Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>,
);

#[derive(Clone)]
pub enum ExpertClient {
    Grpc(OTelGrpcClientMiddleware),
    Shm(LocalShmChannel),
    Rdma(RdmaChannel),
}

impl std::fmt::Debug for ExpertClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpertClient::Grpc(_) => write!(f, "ExpertClient::Grpc(..)"),
            ExpertClient::Shm(_) => write!(f, "ExpertClient::Shm(..)"),
            ExpertClient::Rdma(_) => write!(f, "ExpertClient::Rdma(..)"),
        }
    }
}

impl ExpertClient {
    pub fn into_grpc_client(self) -> Option<OTelGrpcClientMiddleware> {
        match self {
            ExpertClient::Grpc(client) => Some(client),
            ExpertClient::Shm(_) => None,
            ExpertClient::Rdma(_) => None,
        }
    }

    pub fn into_shm_channels(self) -> Option<LocalShmChannel> {
        match self {
            ExpertClient::Grpc(_) => None,
            ExpertClient::Shm(channels) => Some(channels),
            ExpertClient::Rdma(_) => None,
        }
    }

    pub fn into_rdma_channels(self) -> Option<RdmaChannel> {
        match self {
            ExpertClient::Grpc(_) => None,
            ExpertClient::Shm(_) => None,
            ExpertClient::Rdma(channels) => Some(channels),
        }
    }

    pub fn is_grpc(&self) -> bool {
        matches!(self, ExpertClient::Grpc(_))
    }

    pub fn is_shm(&self) -> bool {
        matches!(self, ExpertClient::Shm(_))
    }

    pub fn is_rdma(&self) -> bool {
        matches!(self, ExpertClient::Rdma(_))
    }
}

#[async_trait::async_trait]
pub trait ExpertRegistry {
    async fn select(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ExpertClient>;
    async fn reset(&mut self) -> EKResult<()>;
    async fn deregister(&mut self, host_id: &str);
    /// Clear the deregistered flag for a returning worker.
    fn reregister(&mut self, host_id: &str);
}

#[derive(Clone)]
struct GrpcChannelMeta {
    host_id: String,
    ch: Channel,
}

#[derive(Clone)]
struct ShmChannelMeta {
    host_id: String,
    ch: LocalShmChannel,
}

#[derive(Clone)]
struct RdmaChannelMeta {
    host_id: String,
    ch: RdmaChannel,
}

#[derive(Clone)]
enum ChannelMeta {
    Grpc(GrpcChannelMeta),
    Shm(ShmChannelMeta),
    Rdma(RdmaChannelMeta),
}

/// RDMA connection state for a worker node
#[derive(Clone)]
struct RdmaNodeConnection {
    req_queue: Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>,
    resp_queue: Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>,
    connected: bool,
}

pub struct ExpertRegistryImpl {
    eid2channels: HashMap<ExpertId, Vec<ChannelMeta>>,
    all_shm_channels: HashMap<String, LocalShmChannel>,
    all_rdma_connections: HashMap<String, RdmaNodeConnection>,
    /// Hostnames that have been explicitly deregistered.  Checked in
    /// `create_then_select_channel` to avoid reconnecting to a dead
    /// node before `deactivate_node` finishes its DB write.
    deregistered: std::collections::HashSet<String>,
}

#[async_trait::async_trait]
impl ExpertRegistry for ExpertRegistryImpl {
    async fn reset(&mut self) -> EKResult<()> {
        self.inner_reset().await
    }
    async fn select(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ExpertClient> {
        let ch = self.inner_select(eid).await?;
        match ch {
            ChannelMeta::Grpc(meta) => {
                let client = ServiceBuilder::new()
                    .layer_fn(OTelGrpcClientMiddleware::new)
                    .service(meta.ch.clone());
                Ok(ExpertClient::Grpc(client))
            }
            ChannelMeta::Shm(meta) => Ok(ExpertClient::Shm(meta.ch.clone())),
            ChannelMeta::Rdma(meta) => Ok(ExpertClient::Rdma(meta.ch.clone())),
        }
    }
    async fn deregister(&mut self, host_id: &str) {
        self.inner_deregister(host_id).await;
    }
    fn reregister(&mut self, host_id: &str) {
        self.deregistered.remove(host_id);
    }
}

impl ExpertRegistryImpl {
    async fn inner_reset(&mut self) -> EKResult<()> {
        self.eid2channels.clear();
        Ok(())
    }

    async fn inner_select(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ChannelMeta> {
        let channels = self.eid2channels.get(eid);
        if let Some(channels) = channels {
            if channels.is_empty() {
                return self.create_then_select_channel(eid).await;
            }
            self.select_random(eid).await
        } else {
            self.create_then_select_channel(eid).await
        }
    }

    async fn select_random(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ChannelMeta> {
        let channels = self.eid2channels.get(eid);
        if let Some(channels) = channels {
            if channels.is_empty() {
                return self.create_then_select_channel(eid).await;
            }
            let idx = rand::random::<usize>() % channels.len();
            Ok(channels[idx].clone())
        } else {
            self.create_then_select_channel(eid).await
        }
    }

    async fn create_then_select_channel(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ChannelMeta> {
        // Only consider nodes where the expert is actually "loaded" —
        // not "scheduled" or "pending" (still being loaded by progressive
        // assignment).  This prevents routing to workers that just
        // reconnected and haven't loaded the expert yet.
        let nodes = StateReaderImpl::new()
            .node_by_expert_loaded(eid)
            .await?;

        for node in nodes {
            // Skip nodes that have been explicitly deregistered — instant
            // in-memory check, no race with deactivate_node's DB write.
            if self.deregistered.contains(&node.hostname) {
                continue;
            }
            // Skip deactivated nodes (empty config after deactivate_node)
            let addr = match node.config.get("addr").and_then(|v| v.as_str()) {
                Some(a) => a.to_owned(),
                None => continue,
            };
            let configured_channel = match node.config.get("channel").and_then(|v| v.as_str()) {
                Some(c) => c.to_owned(),
                None => continue,
            };

            // Force gRPC if configured, otherwise use the node's channel type
            let channel = if FORCE_GRPC_CHANNEL {
                "grpc".to_string()
            } else {
                configured_channel
            };

            match channel.as_str() {
                "grpc" => {
                    let end = Channel::from_shared(addr)
                        .map_err(|e| EKError::InvalidInput(format!("invalid url for gRPC: {e}")))?;
                    let channel = end.connect().await?;
                    let meta = GrpcChannelMeta {
                        ch: channel,
                        host_id: node.hostname.clone(),
                    };
                    self.eid2channels
                        .entry(eid.to_owned())
                        .or_default()
                        .push(ChannelMeta::Grpc(meta));
                }
                "shm" => {
                    let shm_channel = self
                        .all_shm_channels
                        .entry(node.hostname.clone())
                        .or_insert_with(|| {
                            let req_queue = Arc::new(Mutex::new(ShmQueue::new(
                                &format!("ek-shmq-req-{}", node.hostname),
                                128,
                            )));
                            let resp_queue = Arc::new(Mutex::new(ShmQueue::new(
                                &format!("ek-shmq-resp-{}", node.hostname),
                                128,
                            )));
                            (req_queue, resp_queue)
                        });
                    let meta = ShmChannelMeta {
                        ch: shm_channel.clone(),
                        host_id: node.hostname.clone(),
                    };
                    self.eid2channels
                        .entry(eid.to_owned())
                        .or_default()
                        .push(ChannelMeta::Shm(meta));
                }
                "rdma" => {
                    // Handle RDMA channel creation
                    self.setup_rdma_channel(&node, eid).await?;
                }
                _ => {
                    return Err(EKError::NotFound(format!(
                        "unknown channel type {channel} for expert {eid}"
                    )));
                }
            }
        }
        let res = self.eid2channels.get(eid).ok_or(EKError::NotFound(format!(
            "no channel found for expert {eid}"
        )))?;
        if res.is_empty() {
            return Err(EKError::NotFound(format!(
                "no channel found for expert {eid}"
            )));
        }
        let idx = rand::random::<usize>() % res.len();
        Ok(res[idx].clone())
    }

    /// Setup RDMA channel for a specific node and expert
    async fn setup_rdma_channel(
        &mut self,
        node: &crate::state::models::Node,
        eid: &str,
    ) -> EKResult<()> {
        log::debug!(
            "registering RDMA channel for expert {eid} on node {}",
            node.hostname
        );

        // Get worker's TCP port for RDMA endpoint exchange
        let worker_tcp_port = node
            .config
            .get("rdma_tcp_port")
            .and_then(|v| v.as_u64())
            .map(|p| p as u16);

        if worker_tcp_port.is_none() {
            log::warn!("Missing worker RDMA TCP port for node {}", node.hostname);
            return Ok(());
        }

        let tcp_port = worker_tcp_port.unwrap();

        // Extract worker IP address from addr field
        let worker_addr = node.config["addr"].as_str().unwrap();
        let worker_ip = self.extract_ip_from_addr(worker_addr)?;

        // Check if we already have a connection for this node
        let needs_new_connection = !self.all_rdma_connections.contains_key(&node.hostname);

        if needs_new_connection {
            // Create controller-side RDMA queues
            // Controller sends requests (sender=true) and receives responses (sender=false)
            let req_queue = RdmaQueue::<ShmqWorkerReq>::new(None, 256, true).map_err(|e| {
                EKError::IoError(std::io::Error::other(format!(
                    "Failed to create controller RDMA request queue: {e}"
                )))
            })?;
            let resp_queue = RdmaQueue::<ShmqWorkerResp>::new(None, 256, false).map_err(|e| {
                EKError::IoError(std::io::Error::other(format!(
                    "Failed to create controller RDMA response queue: {e}"
                )))
            })?;

            let req_queue_arc = Arc::new(Mutex::new(req_queue));
            let resp_queue_arc = Arc::new(Mutex::new(resp_queue));

            // Create connection entry
            let connection = RdmaNodeConnection {
                req_queue: req_queue_arc.clone(),
                resp_queue: resp_queue_arc.clone(),
                connected: false,
            };

            self.all_rdma_connections
                .insert(node.hostname.clone(), connection);

            // Attempt to establish RDMA connection via TCP with retry
            self.connect_to_worker_via_tcp_with_retry(&node.hostname, &worker_ip, tcp_port)
                .await?;
        }

        let connection = self.all_rdma_connections.get(&node.hostname).unwrap();
        let rdma_channel = (connection.req_queue.clone(), connection.resp_queue.clone());

        let meta = RdmaChannelMeta {
            ch: rdma_channel,
            host_id: node.hostname.clone(),
        };

        self.eid2channels
            .entry(eid.to_owned())
            .or_default()
            .push(ChannelMeta::Rdma(meta));

        Ok(())
    }

    /// Establish RDMA connection to a worker node via TCP
    async fn connect_to_worker_via_tcp(
        &mut self,
        hostname: &str,
        worker_ip: &str,
        tcp_port: u16,
    ) -> EKResult<()> {
        if let Some(connection) = self.all_rdma_connections.get_mut(hostname) {
            if connection.connected {
                return Ok(()); // Already connected
            }

            log::info!(
                "Establishing RDMA connection to worker {} (IP: {}) via TCP port {}",
                hostname,
                worker_ip,
                tcp_port
            );

            // Use the TCP client to connect and exchange endpoints
            let req_queue = connection.req_queue.clone();
            let resp_queue = connection.resp_queue.clone();

            RdmaEndpointClient::connect_and_exchange(worker_ip, tcp_port, req_queue, resp_queue)
                .await
                .map_err(|e| EKError::IoError(e))?;

            // Mark connection as established
            connection.connected = true;

            log::info!(
                "🚀RDMA connection to worker {} established successfully",
                hostname
            );
        }

        Ok(())
    }

    /// Establish RDMA connection to a worker node via TCP with retry logic
    async fn connect_to_worker_via_tcp_with_retry(
        &mut self,
        hostname: &str,
        worker_ip: &str,
        tcp_port: u16,
    ) -> EKResult<()> {
        let mut last_error = None;

        for attempt in 0..MAX_RETRIES {
            match self
                .connect_to_worker_via_tcp(hostname, worker_ip, tcp_port)
                .await
            {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < MAX_RETRIES - 1 {
                        let delay_ms = BASE_DELAY_MS * 2_u64.pow(attempt);
                        log::warn!(
                            "RDMA connection attempt {} failed for worker {}, retrying in {}ms: {:?}",
                            attempt + 1,
                            hostname,
                            delay_ms,
                            last_error.as_ref().unwrap()
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        // Clean up failed connection entry before returning error
        self.all_rdma_connections.remove(hostname);

        Err(last_error
            .unwrap_or_else(|| EKError::IoError(std::io::Error::other("Unknown connection error"))))
    }

    /// Extract IP address from worker addr (e.g., "http://192.168.1.100:8080" -> "192.168.1.100")
    fn extract_ip_from_addr(&self, addr: &str) -> EKResult<String> {
        let url = Url::parse(addr)
            .map_err(|e| EKError::InvalidInput(format!("Invalid URL format: {e}")))?;

        let host = url
            .host_str()
            .ok_or_else(|| EKError::InvalidInput("No host found in URL".to_string()))?;

        Ok(host.to_string())
    }

    /// Reset connection state and force reconnection on next use
    pub async fn reset_connection(&mut self, hostname: &str) {
        if let Some(connection) = self.all_rdma_connections.get_mut(hostname) {
            connection.connected = false;
            log::info!("Reset RDMA connection state for worker {}", hostname);
        }
    }

    pub async fn inner_deregister(&mut self, host_id: &str) {
        log::info!("deregister host_id {host_id}");

        // Mark as deregistered so create_then_select_channel won't
        // reconnect before deactivate_node's DB write commits.
        self.deregistered.insert(host_id.to_string());

        // Remove from all channel types
        for (_, channels) in self.eid2channels.iter_mut() {
            channels.retain(|meta| match meta {
                ChannelMeta::Grpc(meta) => meta.host_id != host_id,
                ChannelMeta::Shm(meta) => meta.host_id != host_id,
                ChannelMeta::Rdma(meta) => meta.host_id != host_id,
            });
        }

        // Remove SHM channels
        self.all_shm_channels
            .retain(|hostname, _| hostname != host_id);

        // Reset and remove RDMA connections
        self.reset_connection(host_id).await;
        self.all_rdma_connections
            .retain(|hostname, _| hostname != host_id);

        log::info!("Deregistered worker: {}", host_id);
    }

}

impl Default for ExpertRegistryImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertRegistryImpl {
    pub fn new() -> Self {
        Self {
            eid2channels: HashMap::new(),
            all_shm_channels: HashMap::new(),
            all_rdma_connections: HashMap::new(),
            deregistered: std::collections::HashSet::new(),
        }
    }
}

const MAX_TENSOR_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ShmqWorkerReq {
    id: usize,
    #[serde(with = "serde_arrays")]
    expert_id: [u8; 64],
    input_tensor: Vec<u8>,
}

impl ShmqWorkerReq {
    pub fn new(expert_id: ExpertIdRef<'_>, input_tensor: &[u8]) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(1);

        assert!(expert_id.len() < 64);
        assert!(input_tensor.len() <= MAX_TENSOR_SIZE);

        let mut expert_id_array = [0u8; 64];
        let expert_id_bytes = expert_id.as_bytes();
        let copy_len = std::cmp::min(expert_id_bytes.len(), 63);
        expert_id_array[..copy_len].copy_from_slice(&expert_id_bytes[..copy_len]);

        Self {
            id: ID.fetch_add(1, Ordering::SeqCst),
            expert_id: expert_id_array,
            input_tensor: input_tensor.to_vec(),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn expert_id(&self) -> ExpertId {
        let end = self.expert_id.iter().position(|&b| b == 0).unwrap_or(64);
        String::from_utf8(self.expert_id[..end].to_vec()).unwrap()
    }

    pub fn input_tensor(&self) -> &[u8] {
        &self.input_tensor
    }
}

impl GeneralShmQueueBytes for ShmqWorkerReq {
    const CAPACITY: usize =
        std::mem::size_of::<usize>() + 64 + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn write_to_slice(&self, slice: &mut [u8]) {
        let mut offset = 0;

        // Add id (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.id.to_le_bytes());
        offset += 8;

        // Add expert_id (64 bytes)
        slice[offset..offset + 64].copy_from_slice(&self.expert_id);
        offset += 64;

        // Add input_tensor length (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.input_tensor.len().to_le_bytes());
        offset += 8;

        // Add input_tensor data
        let tensor_len = self.input_tensor.len();
        slice[offset..offset + tensor_len].copy_from_slice(&self.input_tensor);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let id = usize::from_le_bytes(bytes[..std::mem::size_of::<usize>()].try_into().unwrap());
        let expert_id = bytes[std::mem::size_of::<usize>()..std::mem::size_of::<usize>() + 64]
            .try_into()
            .unwrap();
        let input_tensor_len = usize::from_le_bytes(
            bytes[std::mem::size_of::<usize>() + 64
                ..std::mem::size_of::<usize>() + 64 + std::mem::size_of::<usize>()]
                .try_into()
                .unwrap(),
        );
        let input_tensor = bytes
            [std::mem::size_of::<usize>() + 64 + std::mem::size_of::<usize>()..]
            [..input_tensor_len]
            .to_vec();

        Self {
            id,
            expert_id,
            input_tensor,
        }
    }

    fn len(&self) -> usize {
        std::mem::size_of::<usize>() + 64 + std::mem::size_of::<usize>() + self.input_tensor.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShmqWorkerResp {
    id: usize,
    output_tensor: Vec<u8>,
}

impl ShmqWorkerResp {
    pub fn new(id: usize, output_tensor: Vec<u8>) -> Self {
        assert!(output_tensor.len() <= MAX_TENSOR_SIZE);
        Self { id, output_tensor }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn output_tensor(&self) -> &[u8] {
        &self.output_tensor
    }
}

impl GeneralShmQueueBytes for ShmqWorkerResp {
    const CAPACITY: usize =
        std::mem::size_of::<usize>() + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn write_to_slice(&self, slice: &mut [u8]) {
        let mut offset = 0;
        // Add id (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.id.to_le_bytes());
        offset += 8;

        // Add output_tensor length (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.output_tensor.len().to_le_bytes());
        offset += 8;

        // Add output_tensor data
        let tensor_len = self.output_tensor.len();
        slice[offset..offset + tensor_len].copy_from_slice(&self.output_tensor);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let id = usize::from_le_bytes(bytes[..std::mem::size_of::<usize>()].try_into().unwrap());
        let output_tensor_len = usize::from_le_bytes(
            bytes[std::mem::size_of::<usize>()
                ..std::mem::size_of::<usize>() + std::mem::size_of::<usize>()]
                .try_into()
                .unwrap(),
        );
        let output_tensor = bytes[std::mem::size_of::<usize>() + std::mem::size_of::<usize>()..]
            [..output_tensor_len]
            .to_vec();

        Self { id, output_tensor }
    }

    fn len(&self) -> usize {
        std::mem::size_of::<usize>() + std::mem::size_of::<usize>() + self.output_tensor.len()
    }
}

pub type GlobalWorkerRegistry = Arc<Mutex<dyn ExpertRegistry + Send + Sync>>;

pub fn get_registry() -> GlobalWorkerRegistry {
    static INSTANCE: OnceLock<Arc<Mutex<ExpertRegistryImpl>>> = OnceLock::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = ExpertRegistryImpl::new();
        Arc::new(Mutex::new(inner))
    });
    (res.clone()) as _
}
