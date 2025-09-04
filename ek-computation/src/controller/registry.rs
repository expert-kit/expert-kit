use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use ek_base::{
    error::{EKError, EKResult},
    tracing::grpc::OTelGrpcClientMiddleware,
};
use ndarray_rand::rand;
use tokio::sync::Mutex;
use tonic::transport::Channel;
use tower::ServiceBuilder;

use crate::{
    shmq::{
        ShmBytes, ShmQueue,
        rdma_impl::{RdmaBytes, RdmaQueue},
    },
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewNode,
        writer::StateWriterImpl,
    },
};
use serde_json;

pub type ExpertId = String;
pub type ExpertIdRef<'a> = &'a str;

pub type LocalShmChannel = (
    Arc<Mutex<ShmQueue<'static, LocalShmWorkerReq>>>,
    Arc<Mutex<ShmQueue<'static, LocalShmWorkerResp>>>,
);

pub type RdmaChannel<T, U> = (Arc<Mutex<RdmaQueue<T>>>, Arc<Mutex<RdmaQueue<U>>>);

#[derive(Clone)]
pub enum ExpertClient {
    Grpc(OTelGrpcClientMiddleware),
    Shm(LocalShmChannel),
    Rdma(RdmaChannel<RdmaWorkerReq, RdmaWorkerResp>),
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

    pub fn into_rdma_channels(self) -> Option<RdmaChannel<RdmaWorkerReq, RdmaWorkerResp>> {
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
    ch: RdmaChannel<RdmaWorkerReq, RdmaWorkerResp>,
}

#[derive(Clone)]
enum ChannelMeta {
    Grpc(GrpcChannelMeta),
    Shm(ShmChannelMeta),
    Rdma(RdmaChannelMeta),
}

pub struct ExpertRegistryImpl {
    eid2channels: HashMap<ExpertId, Vec<ChannelMeta>>,
    all_shm_channels: HashMap<String, LocalShmChannel>,
    reader: Box<dyn StateReader + Send + Sync>,
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
        let nodes = self.reader.node_by_expert(eid).await?;
        for node in nodes {
            let addr = node.config["addr"].as_str().unwrap().to_owned();
            let channel = node.config["channel"].as_str().unwrap().to_owned();

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

    pub async fn inner_deregister(&mut self, host_id: &str) {
        log::info!("deregister host_id {host_id}");
        for (_, channels) in self.eid2channels.iter_mut() {
            channels.retain(|meta| match meta {
                ChannelMeta::Grpc(meta) => meta.host_id != host_id,
                ChannelMeta::Shm(meta) => meta.host_id != host_id,
                ChannelMeta::Rdma(meta) => meta.host_id != host_id,
            });
        }
        self.all_shm_channels
            .retain(|hostname, _| hostname != host_id);
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
            reader: Box::new(StateReaderImpl::new()),
        }
    }
}

const MAX_TENSOR_SIZE: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalShmWorkerReq {
    id: usize,
    expert_id: [u8; 64],
    input_tensor: Vec<u8>,
}

impl LocalShmWorkerReq {
    pub fn new(expert_id: ExpertIdRef<'_>, input_tensor: &[u8]) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(1);

        assert!(expert_id.len() < 64);
        assert!(input_tensor.len() <= MAX_TENSOR_SIZE);

        // Safely convert expert_id to fixed-size array, padding with zeros if necessary
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
        // Find the first null byte to determine the actual string length
        let end = self.expert_id.iter().position(|&b| b == 0).unwrap_or(64);
        // Convert bytes to string, handling potential UTF-8 errors gracefully
        String::from_utf8(self.expert_id[..end].to_vec()).unwrap()
    }

    pub fn input_tensor(&self) -> &[u8] {
        &self.input_tensor
    }
}

impl ShmBytes for LocalShmWorkerReq {
    const SIZE: usize =
        std::mem::size_of::<usize>() + 64 + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn as_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.id
            .to_le_bytes()
            .into_iter()
            .chain(self.expert_id)
            .chain(self.input_tensor.len().to_le_bytes())
            .chain(self.input_tensor.clone())
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalShmWorkerResp {
    id: usize,
    output_tensor: Vec<u8>,
}

impl LocalShmWorkerResp {
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

impl ShmBytes for LocalShmWorkerResp {
    const SIZE: usize =
        std::mem::size_of::<usize>() + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn as_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        self.id
            .to_le_bytes()
            .into_iter()
            .chain(self.output_tensor.len().to_le_bytes())
            .chain(self.output_tensor.clone())
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaWorkerReq {
    id: usize,
    expert_id: [u8; 64],
    input_tensor: Vec<u8>,
}

impl RdmaWorkerReq {
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

impl RdmaBytes for RdmaWorkerReq {
    const SIZE: usize =
        std::mem::size_of::<usize>() + 64 + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn as_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        let mut result = Vec::with_capacity(Self::SIZE);

        // Add id (8 bytes)
        result.extend_from_slice(&self.id.to_le_bytes());

        // Add expert_id (64 bytes)
        result.extend_from_slice(&self.expert_id);

        // Add input_tensor length (8 bytes)
        result.extend_from_slice(&self.input_tensor.len().to_le_bytes());

        // Add input_tensor data
        result.extend_from_slice(&self.input_tensor);

        // Pad to exact SIZE with zeros
        result.resize(Self::SIZE, 0);

        result.into_iter()
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdmaWorkerResp {
    id: usize,
    output_tensor: Vec<u8>,
}

impl RdmaWorkerResp {
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

impl RdmaBytes for RdmaWorkerResp {
    const SIZE: usize =
        std::mem::size_of::<usize>() + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn as_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        let mut result = Vec::with_capacity(Self::SIZE);

        // Add id (8 bytes)
        result.extend_from_slice(&self.id.to_le_bytes());

        // Add output_tensor length (8 bytes)
        result.extend_from_slice(&self.output_tensor.len().to_le_bytes());

        // Add output_tensor data
        result.extend_from_slice(&self.output_tensor);

        // Pad to exact SIZE with zeros
        result.resize(Self::SIZE, 0);

        result.into_iter()
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
}

#[expect(clippy::type_complexity)]
pub struct LocalShmExpertRegistry {
    all_channels: HashMap<
        String,
        (
            Arc<Mutex<ShmQueue<'static, LocalShmWorkerReq>>>,
            Arc<Mutex<ShmQueue<'static, LocalShmWorkerResp>>>,
        ),
    >,
    experts2channels: HashMap<
        ExpertId,
        Vec<(
            String,
            Arc<Mutex<ShmQueue<'static, LocalShmWorkerReq>>>,
            Arc<Mutex<ShmQueue<'static, LocalShmWorkerResp>>>,
        )>,
    >,
    reader: Box<dyn StateReader + Send + Sync>,
}

impl Default for LocalShmExpertRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalShmExpertRegistry {
    pub fn new() -> Self {
        Self {
            all_channels: HashMap::default(),
            experts2channels: HashMap::default(),
            reader: Box::new(StateReaderImpl::new()),
        }
    }
}

#[async_trait::async_trait]
impl ExpertRegistry for LocalShmExpertRegistry {
    async fn select(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ExpertClient> {
        if !self.experts2channels.contains_key(eid) {
            let nodes = self.reader.node_by_expert(eid).await?;
            for node in nodes {
                log::debug!(
                    "registering channel for expert {eid} on node {}",
                    node.hostname
                );
                let (req_channel, resp_channel) = self
                    .all_channels
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
                self.experts2channels
                    .entry(eid.to_owned())
                    .or_default()
                    .push((
                        node.hostname.clone(),
                        req_channel.clone(),
                        resp_channel.clone(),
                    ));
            }
        }
        let channels = self
            .experts2channels
            .get(eid)
            .ok_or(EKError::NotFound(format!(
                "no channel found for expert {eid}"
            )))?;
        let idx = rand::random::<usize>() % channels.len();
        Ok(ExpertClient::Shm((
            channels[idx].1.clone(),
            channels[idx].2.clone(),
        )))
    }

    async fn reset(&mut self) -> EKResult<()> {
        self.experts2channels.clear();
        Ok(())
    }

    async fn deregister(&mut self, host_id: &str) {
        self.all_channels.retain(|hostname, _| hostname != host_id);

        for (_, channels) in self.experts2channels.iter_mut() {
            channels.retain(|(id, _, _)| id != host_id);
        }
    }
}

/// RDMA connection state for a worker node
struct RdmaNodeConnection {
    req_queue: Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>,
    resp_queue: Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>,
    controller_endpoint_data: Vec<u8>, // Serialized controller endpoint info
    connected: bool,
}

#[expect(clippy::type_complexity)]
pub struct RdmaExpertRegistry {
    all_connections: HashMap<String, RdmaNodeConnection>,
    experts2channels: HashMap<
        ExpertId,
        Vec<(
            String,
            Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>,
            Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>,
        )>,
    >,
    reader: Box<dyn StateReader + Send + Sync>,
    writer: StateWriterImpl,
}

impl Default for RdmaExpertRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl RdmaExpertRegistry {
    pub fn new() -> Self {
        Self {
            all_connections: HashMap::default(),
            experts2channels: HashMap::default(),
            reader: Box::new(StateReaderImpl::new()),
            writer: StateWriterImpl::new(),
        }
    }

    /// Get controller RDMA endpoint data for a worker node
    pub fn get_controller_endpoint(&self, hostname: &str) -> Option<Vec<u8>> {
        self.all_connections
            .get(hostname)
            .map(|conn| conn.controller_endpoint_data.clone())
    }

    /// Establish RDMA connection to a worker node
    async fn connect_to_worker(
        &mut self,
        hostname: &str,
        worker_endpoint_info: &serde_json::Value,
    ) -> EKResult<()> {
        if let Some(connection) = self.all_connections.get_mut(hostname) {
            if connection.connected {
                return Ok(()); // Already connected
            }

            // Extract worker endpoint JSON strings from database
            let worker_qp_json = worker_endpoint_info
                .get("qp_endpoint")
                .and_then(|v| v.as_str())
                .ok_or_else(|| EKError::InvalidInput("Missing worker qp_endpoint".into()))?;
            let worker_memory_json = worker_endpoint_info
                .get("memory_region")
                .and_then(|v| v.as_str())
                .ok_or_else(|| EKError::InvalidInput("Missing worker memory_region".into()))?;

            // Deserialize worker endpoint information
            let worker_qp_endpoint: ibverbs::QueuePairEndpoint =
                serde_json::from_str(worker_qp_json).map_err(|e| {
                    EKError::InvalidInput(format!("Failed to deserialize worker QP endpoint: {e}"))
                })?;
            let worker_memory_region: ibverbs::RemoteMemoryRegion =
                serde_json::from_str(worker_memory_json).map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to deserialize worker memory region: {e}"
                    ))
                })?;

            log::info!(
                "Establishing RDMA connection to worker {} using endpoint data",
                hostname
            );
            log::debug!("Worker QP endpoint: {:?}", worker_qp_endpoint);
            log::debug!("Worker memory region: {:?}", worker_memory_region);

            // Establish connections from controller to worker
            {
                let mut req_queue = connection.req_queue.lock().await;
                if let Err(e) =
                    req_queue.connect(worker_qp_endpoint.clone(), worker_memory_region.clone())
                {
                    log::error!(
                        "Failed to connect controller request queue to worker {}: {e}",
                        hostname
                    );
                    return Err(EKError::IoError(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Controller request queue connection failed: {e}"),
                    )));
                }
                log::info!(
                    "Controller request queue connected to worker {} successfully",
                    hostname
                );
            }

            // Mark connection as established
            connection.connected = true;
            log::info!(
                "RDMA connection to worker {} established successfully",
                hostname
            );
        }

        Ok(())
    }

    /// Reset connection state and force reconnection on next use
    pub async fn reset_connection(&mut self, hostname: &str) {
        if let Some(connection) = self.all_connections.get_mut(hostname) {
            connection.connected = false;
            log::info!("Reset RDMA connection state for worker {}", hostname);
        }
    }
}

#[async_trait::async_trait]
impl ExpertRegistry for RdmaExpertRegistry {
    async fn select(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ExpertClient> {
        if !self.experts2channels.contains_key(eid) {
            log::info!("🚀hello");

            let nodes = self.reader.node_by_expert(eid).await?;
            for node in nodes {
                log::debug!(
                    "registering RDMA channel for expert {eid} on node {}",
                    node.hostname
                );

                let worker_rdma_req_endpoint = node.config.get("worker_rdma_request_endpoint");
                let _worker_rdma_resp_endpoint = node.config.get("worker_rdma_response_endpoint");
                if worker_rdma_req_endpoint.is_none() {
                    log::warn!(
                        "No worker RDMA request endpoint found for node {}",
                        node.hostname
                    );
                    continue;
                }

                // Extract worker's request endpoint info from database (where controller sends requests)
                let worker_req_endpoint_info = worker_rdma_req_endpoint.unwrap();

                // Check if we already have a connection for this node
                let needs_new_connection = !self.all_connections.contains_key(&node.hostname);

                if needs_new_connection {
                    // Create controller-side RDMA queues (reverse roles from worker)
                    let req_queue = RdmaQueue::<RdmaWorkerReq>::new(None, 128, true)
                        .expect("Failed to create controller RDMA request queue");
                    let resp_queue = RdmaQueue::<RdmaWorkerResp>::new(None, 128, false)
                        .expect("Failed to create controller RDMA response queue");

                    // Get controller endpoints for handshake
                    let controller_req_endpoint = req_queue
                        .endpoint()
                        .expect("Failed to get controller request endpoint");
                    let controller_req_memory = req_queue.memory_region();
                    let controller_resp_endpoint = resp_queue
                        .endpoint()
                        .expect("Failed to get controller response endpoint");
                    let controller_resp_memory = resp_queue.memory_region();

                    // Serialize controller endpoint info for sending back to worker
                    let controller_endpoint_data = serde_json::to_vec(&(
                        format!("{:?}", controller_req_endpoint),
                        format!("{:?}", controller_resp_endpoint),
                    ))
                    .expect("Failed to serialize controller endpoints");

                    // Store controller RESPONSE endpoint in database (for worker to send responses to)
                    let controller_qp_json = serde_json::to_string(&controller_resp_endpoint)
                        .expect("Failed to serialize controller QP endpoint");
                    let controller_memory_json = serde_json::to_string(&controller_resp_memory)
                        .expect("Failed to serialize controller memory region");

                    let mut updated_config = node.config.clone();

                    // Store controller's request endpoint (for worker to connect its request queue to)
                    updated_config["controller_rdma_request_endpoint"] = serde_json::json!({
                        "qp_endpoint": serde_json::to_string(&controller_req_endpoint).unwrap(),
                        "memory_region": serde_json::to_string(&controller_req_memory).unwrap(),
                    });

                    // Store controller's response endpoint (for worker to connect its response queue to)
                    updated_config["controller_rdma_response_endpoint"] = serde_json::json!({
                        "qp_endpoint": controller_qp_json,
                        "memory_region": controller_memory_json,
                    });

                    // Update database with controller endpoint info
                    let node_update = NewNode {
                        hostname: node.hostname.clone(),
                        device: node.device.clone(),
                        config: updated_config,
                    };

                    // Store controller endpoint in database
                    if let Err(e) = self.writer.node_upsert(node_update).await {
                        log::error!("Failed to store controller RDMA endpoint in database: {e:?}");
                    }

                    // Create connection entry
                    let connection = RdmaNodeConnection {
                        req_queue: Arc::new(Mutex::new(req_queue)),
                        resp_queue: Arc::new(Mutex::new(resp_queue)),
                        controller_endpoint_data,
                        connected: false,
                    };

                    self.all_connections
                        .insert(node.hostname.clone(), connection);
                }

                // Attempt to establish RDMA connection if worker endpoint is available
                if let Err(e) = self
                    .connect_to_worker(&node.hostname, worker_req_endpoint_info)
                    .await
                {
                    log::error!(
                        "Failed to establish RDMA connection to worker {}: {e:?}",
                        node.hostname
                    );
                }

                let connection = self.all_connections.get(&node.hostname).unwrap();

                let (req_channel, resp_channel) =
                    (connection.req_queue.clone(), connection.resp_queue.clone());

                self.experts2channels
                    .entry(eid.to_owned())
                    .or_default()
                    .push((
                        node.hostname.clone(),
                        req_channel.clone(),
                        resp_channel.clone(),
                    ));
            }
        }
        let channels = self
            .experts2channels
            .get(eid)
            .ok_or(EKError::NotFound(format!(
                "no channel found for expert {eid}"
            )))?;
        let idx = rand::random::<usize>() % channels.len();
        Ok(ExpertClient::Rdma((
            channels[idx].1.clone(),
            channels[idx].2.clone(),
        )))
    }

    async fn reset(&mut self) -> EKResult<()> {
        self.experts2channels.clear();
        Ok(())
    }

    async fn deregister(&mut self, host_id: &str) {
        // Reset and remove RDMA connections
        self.reset_connection(host_id).await;
        self.all_connections
            .retain(|hostname, _| hostname != host_id);

        // Remove from expert channels mapping
        for (_, channels) in self.experts2channels.iter_mut() {
            channels.retain(|(id, _, _)| id != host_id);
        }

        log::info!("Deregistered RDMA worker: {}", host_id);
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
