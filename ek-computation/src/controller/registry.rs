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
    shmq::{ShmBytes, ShmQueue},
    state::io::{StateReader, StateReaderImpl},
};

pub type ExpertId = String;
pub type ExpertIdRef<'a> = &'a str;

pub type LocalShmChannel = (
    Arc<Mutex<ShmQueue<'static, LocalShmWorkerReq>>>,
    Arc<Mutex<ShmQueue<'static, LocalShmWorkerResp>>>,
);

#[derive(Clone)]
pub enum ExpertClient {
    Grpc(OTelGrpcClientMiddleware),
    Shm(LocalShmChannel),
}

impl std::fmt::Debug for ExpertClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpertClient::Grpc(_) => write!(f, "ExpertClient::Grpc(..)"),
            ExpertClient::Shm(_) => write!(f, "ExpertClient::Shm(..)"),
        }
    }
}

impl ExpertClient {
    pub fn into_grpc_client(self) -> Option<OTelGrpcClientMiddleware> {
        match self {
            ExpertClient::Grpc(client) => Some(client),
            ExpertClient::Shm(_) => None,
        }
    }

    pub fn into_shm_channels(self) -> Option<LocalShmChannel> {
        match self {
            ExpertClient::Grpc(_) => None,
            ExpertClient::Shm(channels) => Some(channels),
        }
    }

    pub fn is_grpc(&self) -> bool {
        matches!(self, ExpertClient::Grpc(_))
    }

    pub fn is_shm(&self) -> bool {
        matches!(self, ExpertClient::Shm(_))
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
enum ChannelMeta {
    Grpc(GrpcChannelMeta),
    Shm(ShmChannelMeta),
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

pub type GlobalWorkerRegistry = Arc<Mutex<dyn ExpertRegistry + Send + Sync>>;

pub fn get_registry() -> GlobalWorkerRegistry {
    static INSTANCE: OnceLock<Arc<Mutex<ExpertRegistryImpl>>> = OnceLock::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = ExpertRegistryImpl::new();
        Arc::new(Mutex::new(inner))
    });
    (res.clone()) as _
}
