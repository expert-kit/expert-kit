use std::{
    collections::HashMap,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
    tracing::grpc::OTelGrpcClientMiddleware,
    utils::Defers,
};
use ndarray_rand::rand;
use serde::{Deserialize, Serialize};
use tokio::{sync::Mutex, time};
use tonic::transport::Channel;
use tower::ServiceBuilder;
use tracing::Instrument;

use crate::{
    controller::executor::{ForwardResponse, PendingResponse},
    metrics::METRIC_CONTROLLER_INTRA_REQ,
    proto::ek::worker::v1::{self, forward_req::SequenceInfo},
    shmq::{GeneralShmQueueBytes, ShmQueue, rdma_impl::RdmaQueue},
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewNode,
        writer::StateWriterImpl,
    },
};
use serde_json;

pub type ExpertId = String;
pub type ExpertIdRef<'a> = &'a str;

pub type LocalShmReqQueue = Arc<Mutex<ShmQueue<'static, ShmqWorkerReq>>>;
pub type LocalShmRespQueue = Arc<Mutex<(ShmQueue<'static, ShmqWorkerResp>, ChannelMetrics)>>;
pub type LocalShmChannel = (LocalShmReqQueue, LocalShmRespQueue);

pub type RdmaReqQueue = Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>;
pub type RdmaRespQueue = Arc<Mutex<(RdmaQueue<ShmqWorkerResp>, ChannelMetrics)>>;
pub type RdmaChannel = (RdmaReqQueue, RdmaRespQueue);

pub struct ChannelMetrics {
    priority: f64,
}

impl ChannelMetrics {
    pub fn priority(&self) -> f64 {
        self.priority
    }

    pub fn accumulate(&mut self, delta: f64) {
        self.priority = (self.priority * 0.8 + delta) / 2.0;
    }
}

impl Default for ChannelMetrics {
    fn default() -> Self {
        Self { priority: 1.0 }
    }
}

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
    pub(crate) async fn get_priority(&self) -> f64 {
        match self {
            ExpertClient::Grpc(_) => 1.0, // gRPC client does not have priority tracking
            ExpertClient::Shm((_, resp_channel)) => resp_channel.lock().await.1.priority(),
            ExpertClient::Rdma((_, resp_channel)) => resp_channel.lock().await.1.priority(),
        }
    }

    pub(crate) async fn update_priority(&self, delta: f64) {
        match self {
            ExpertClient::Grpc(_) => {
                // gRPC client does not have priority tracking
            }
            ExpertClient::Shm((_, resp_channel)) => {
                resp_channel.lock().await.1.accumulate(delta);
            }
            ExpertClient::Rdma((_, resp_channel)) => {
                resp_channel.lock().await.1.accumulate(delta);
            }
        }
    }

    pub(crate) fn forward(
        self,
        expert_id: ExpertId,
        tensor: Vec<u8>,
        sequences: Vec<SequenceInfo>,
        pending_resp: Arc<Mutex<HashMap<usize, PendingResponse>>>,
    ) -> tokio::task::JoinHandle<EKResult<ForwardResponse>> {
        match self {
            ExpertClient::Grpc(grpc_channel) => {
                let mut cli =
                    v1::computation_service_client::ComputationServiceClient::new(grpc_channel)
                        .max_decoding_message_size(1024 * 1024 * 1024)
                        .max_encoding_message_size(1024 * 1024 * 1024);

                let fu = async move {
                    let req = v1::ForwardReq {
                        // TODO: hardcode instance id.
                        instance_id: "0".into(),
                        tensor,
                        sequences,
                    };

                    let _d = Self::create_metrics_defer();
                    let now = time::Instant::now();
                    cli.forward(req)
                        .await
                        .map(|resp| ForwardResponse::grpc(resp.into_inner(), now.elapsed()))
                        .map_err(|e| {
                            log::error!("forward error: {e}");
                            EKError::IoError(std::io::Error::other(format!("grpc error: {e}")))
                        })
                }
                .in_current_span();
                tokio::task::spawn(fu)
            }
            ExpertClient::Shm((send_channel, recv_channel)) => {
                let fu = async move {
                    let req = ShmqWorkerReq::new(expert_id.as_ref(), &tensor);
                    let _d = Self::create_metrics_defer();
                    let now = time::Instant::now();

                    // Send request with retry
                    if let Err(e) =
                        Self::send_shm_request_with_retry(&send_channel, &req, &expert_id).await
                    {
                        log::error!("Failed to send SHM request: {e:?}");
                        return Err(e);
                    }

                    log::debug!(
                        "request sent for expert {}, waiting for response",
                        expert_id
                    );

                    // Receive response with ID matching
                    match Self::receive_shm_response(
                        &recv_channel,
                        &pending_resp,
                        req.id(),
                        &expert_id,
                    )
                    .await
                    {
                        Ok(resp) => Ok(ForwardResponse::shm(resp, now.elapsed())),
                        Err(e) => {
                            log::error!("Failed to receive SHM response: {e:?}");
                            Err(e)
                        }
                    }
                }
                .in_current_span();
                tokio::task::spawn(fu)
            }
            ExpertClient::Rdma((send_channel, recv_channel)) => {
                let fu = async move {
                    let req = ShmqWorkerReq::new(expert_id.as_ref(), &tensor);
                    let _d = Self::create_metrics_defer();
                    let now = time::Instant::now();

                    // Send request with retry
                    if let Err(e) =
                        Self::send_rdma_request_with_retry(&send_channel, &req, &expert_id).await
                    {
                        log::error!("Failed to send RDMA request: {e:?}");
                        return Err(e);
                    }

                    log::debug!(
                        "RDMA request sent for expert {}, waiting for response",
                        expert_id
                    );

                    // Receive response with ID matching
                    match Self::receive_rdma_response(
                        &recv_channel,
                        &pending_resp,
                        req.id(),
                        &expert_id,
                    )
                    .await
                    {
                        Ok(resp) => Ok(ForwardResponse::rdma(resp, now.elapsed())),
                        Err(e) => {
                            log::error!("Failed to receive RDMA response: {e:?}");
                            Err(e)
                        }
                    }
                }
                .in_current_span();
                tokio::task::spawn(fu)
            }
        }
    }

    /// Create a performance timer deferred callback for metrics collection
    fn create_metrics_defer() -> Defers {
        let settings = get_ek_settings();
        let start = time::Instant::now();
        Defers::defer(Box::new(move || {
            let elapsed = start.elapsed();
            METRIC_CONTROLLER_INTRA_REQ
                .with_label_values(&[settings.inference.model_name.as_str()])
                .observe(elapsed.as_micros() as f64);
        }))
    }

    /// Handle shared memory request sending with retry logic
    async fn send_shm_request_with_retry(
        send_channel: &LocalShmReqQueue,
        req: &ShmqWorkerReq,
        expert_id: &str,
    ) -> EKResult<()> {
        while send_channel.lock().await.send(req).is_err() {
            log::warn!("failed to send request to expert {expert_id}");
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }
        Ok(())
    }

    /// Handle RDMA request sending with retry logic
    async fn send_rdma_request_with_retry(
        send_channel: &RdmaReqQueue,
        req: &ShmqWorkerReq,
        expert_id: &str,
    ) -> EKResult<()> {
        loop {
            match send_channel.lock().await.send(req) {
                Ok(_) => break,
                Err(e) => {
                    log::warn!("failed to send RDMA request to expert {expert_id}: {e}");
                    tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
                }
            }
        }
        Ok(())
    }

    /// Handle shared memory response reception with ID matching
    async fn receive_shm_response(
        recv_channel: &LocalShmRespQueue,
        pending_resp: &Arc<Mutex<HashMap<usize, PendingResponse>>>,
        req_id: usize,
        expert_id: &str,
    ) -> EKResult<ShmqWorkerResp> {
        loop {
            // Check for pending response first
            if let Some(pending_resp_item) = pending_resp.lock().await.remove(&req_id)
                && let PendingResponse::Shm(resp) = pending_resp_item
            {
                return Ok(resp);
            }

            // Receive new response
            match recv_channel.lock().await.0.recv() {
                Ok(resp) => {
                    if resp.id() != req_id {
                        log::debug!(
                            "received response for expert {} but id not correct",
                            expert_id
                        );
                        pending_resp
                            .lock()
                            .await
                            .insert(resp.id(), PendingResponse::Shm(resp));
                        tokio::task::yield_now().await;
                        continue;
                    }
                    return Ok(resp);
                }
                Err(_) => {
                    tokio::task::yield_now().await;
                }
            }
        }
    }

    /// Handle RDMA response reception with ID matching
    async fn receive_rdma_response(
        recv_channel: &RdmaRespQueue,
        pending_resp: &Arc<Mutex<HashMap<usize, PendingResponse>>>,
        req_id: usize,
        expert_id: &str,
    ) -> EKResult<ShmqWorkerResp> {
        loop {
            // Check for pending response first
            if let Some(pending_resp_item) = pending_resp.lock().await.remove(&req_id)
                && let PendingResponse::Rdma(resp) = pending_resp_item
            {
                return Ok(resp);
            }

            // Receive new response
            match recv_channel.lock().await.0.recv() {
                Ok(resp) => {
                    if resp.id() != req_id {
                        log::debug!(
                            "received RDMA response for expert {} but id not correct",
                            expert_id
                        );
                        pending_resp
                            .lock()
                            .await
                            .insert(resp.id(), PendingResponse::Rdma(resp));
                        tokio::task::yield_now().await;
                        continue;
                    }
                    return Ok(resp);
                }
                Err(_) => {
                    tokio::task::yield_now().await;
                }
            }
        }
    }
}

#[async_trait::async_trait]
pub trait ExpertRegistry {
    async fn select(&mut self, eid: ExpertIdRef<'_>) -> EKResult<ExpertClient>;
    async fn select_all(&mut self, eid: ExpertIdRef<'_>) -> EKResult<Vec<ExpertClient>>;
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
    req_queue: RdmaReqQueue,
    resp_queue: RdmaRespQueue,
    connected: bool,
}

pub struct ExpertRegistryImpl {
    eid2channels: HashMap<ExpertId, Vec<ChannelMeta>>,
    all_shm_channels: HashMap<String, LocalShmChannel>,
    all_rdma_connections: HashMap<String, RdmaNodeConnection>,
    reader: Box<dyn StateReader + Send + Sync>,
    writer: StateWriterImpl,
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

    async fn select_all(&mut self, eid: ExpertIdRef<'_>) -> EKResult<Vec<ExpertClient>> {
        let channels = self.inner_select_all(eid).await?;
        channels
            .into_iter()
            .map(|ch| match ch {
                ChannelMeta::Grpc(meta) => {
                    let client = ServiceBuilder::new()
                        .layer_fn(OTelGrpcClientMiddleware::new)
                        .service(meta.ch.clone());
                    Ok(ExpertClient::Grpc(client))
                }
                ChannelMeta::Shm(meta) => Ok(ExpertClient::Shm(meta.ch.clone())),
                ChannelMeta::Rdma(meta) => Ok(ExpertClient::Rdma(meta.ch.clone())),
            })
            .collect()
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
        if let Some(channels) = channels
            && !channels.is_empty()
        {
            let idx = rand::random::<usize>() % channels.len();
            return Ok(channels[idx].clone());
        }
        let channels = self.create_then_select_channels(eid).await?;
        let idx = rand::random::<usize>() % channels.len();
        Ok(channels[idx].clone())
    }

    async fn inner_select_all(&mut self, eid: ExpertIdRef<'_>) -> EKResult<Vec<ChannelMeta>> {
        if let Some(channels) = self.eid2channels.get(eid)
            && !channels.is_empty()
        {
            Ok(channels.clone())
        } else {
            Ok(self.create_then_select_channels(eid).await?.clone())
        }
    }

    async fn create_then_select_channels(
        &mut self,
        eid: ExpertIdRef<'_>,
    ) -> EKResult<&Vec<ChannelMeta>> {
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
                            let resp_queue = Arc::new(Mutex::new((
                                ShmQueue::new(&format!("ek-shmq-resp-{}", node.hostname), 128),
                                ChannelMetrics::default(),
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

        Ok(res)
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

        let worker_rdma_req_endpoint = node.config.get("worker_rdma_request_endpoint");
        let worker_rdma_resp_endpoint = node.config.get("worker_rdma_response_endpoint");
        if worker_rdma_req_endpoint.is_none() || worker_rdma_resp_endpoint.is_none() {
            log::warn!(
                "Missing worker RDMA endpoints for node {} (req: {}, resp: {})",
                node.hostname,
                worker_rdma_req_endpoint.is_some(),
                worker_rdma_resp_endpoint.is_some()
            );
            return Ok(());
        }

        // Extract worker's endpoint info from database
        let worker_req_endpoint_info = worker_rdma_req_endpoint.unwrap();
        let worker_resp_endpoint_info = worker_rdma_resp_endpoint.unwrap();

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

            // Get controller endpoints for handshake
            let controller_req_endpoint = req_queue.endpoint().map_err(|e| {
                EKError::IoError(std::io::Error::other(format!(
                    "Failed to get controller request endpoint: {e}"
                )))
            })?;
            let controller_req_memory = req_queue.memory_region();
            let controller_resp_endpoint = resp_queue.endpoint().map_err(|e| {
                EKError::IoError(std::io::Error::other(format!(
                    "Failed to get controller response endpoint: {e}"
                )))
            })?;
            let controller_resp_memory = resp_queue.memory_region();

            // Serialize controller endpoints as JSON strings
            let controller_req_qp_json =
                serde_json::to_string(&controller_req_endpoint).map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to serialize controller request QP endpoint: {e}"
                    ))
                })?;
            let controller_req_memory_json = serde_json::to_string(&controller_req_memory)
                .map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to serialize controller request memory region: {e}"
                    ))
                })?;
            let controller_resp_qp_json = serde_json::to_string(&controller_resp_endpoint)
                .map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to serialize controller response QP endpoint: {e}"
                    ))
                })?;
            let controller_resp_memory_json = serde_json::to_string(&controller_resp_memory)
                .map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to serialize controller response memory region: {e}"
                    ))
                })?;

            let mut updated_config = node.config.clone();

            // Store controller endpoints as JSON strings (consistent with worker endpoints)
            updated_config["controller_rdma_request_endpoint"] = serde_json::json!({
                "qp_endpoint": controller_req_qp_json,
                "memory_region": controller_req_memory_json,
            });

            updated_config["controller_rdma_response_endpoint"] = serde_json::json!({
                "qp_endpoint": controller_resp_qp_json,
                "memory_region": controller_resp_memory_json,
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
                resp_queue: Arc::new(Mutex::new((resp_queue, ChannelMetrics::default()))),
                connected: false,
            };

            self.all_rdma_connections
                .insert(node.hostname.clone(), connection);
        }

        // Attempt to establish RDMA connection if worker endpoints are available
        if let Err(e) = self
            .connect_to_worker(
                &node.hostname,
                worker_req_endpoint_info,
                worker_resp_endpoint_info,
            )
            .await
        {
            log::error!(
                "Failed to establish RDMA connection to worker {}: {e:?}",
                node.hostname
            );
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

    /// Establish RDMA connection to a worker node
    async fn connect_to_worker(
        &mut self,
        hostname: &str,
        worker_req_endpoint_info: &serde_json::Value,
        worker_resp_endpoint_info: &serde_json::Value,
    ) -> EKResult<()> {
        if let Some(connection) = self.all_rdma_connections.get_mut(hostname) {
            if connection.connected {
                return Ok(()); // Already connected
            }
            log::debug!(
                "Worker request endpoint info: {:?}",
                worker_req_endpoint_info
            );
            log::debug!(
                "Worker response endpoint info: {:?}",
                worker_resp_endpoint_info
            );

            // Extract worker request endpoint JSON strings from database
            let worker_req_qp_json = worker_req_endpoint_info
                .get("qp_endpoint")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    EKError::InvalidInput("Missing worker request qp_endpoint".into())
                })?;
            let worker_req_memory_json = worker_req_endpoint_info
                .get("memory_region")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    EKError::InvalidInput("Missing worker request memory_region".into())
                })?;

            // Extract worker response endpoint JSON strings from database
            let worker_resp_qp_json = worker_resp_endpoint_info
                .get("qp_endpoint")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    EKError::InvalidInput("Missing worker response qp_endpoint".into())
                })?;
            let worker_resp_memory_json = worker_resp_endpoint_info
                .get("memory_region")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    EKError::InvalidInput("Missing worker response memory_region".into())
                })?;

            // Deserialize worker request endpoint information
            let worker_req_qp_endpoint: ibverbs::QueuePairEndpoint =
                serde_json::from_str(worker_req_qp_json).map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to deserialize worker request QP endpoint: {e}"
                    ))
                })?;
            let worker_req_memory_region: ibverbs::RemoteMemoryRegion =
                serde_json::from_str(worker_req_memory_json).map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to deserialize worker request memory region: {e}"
                    ))
                })?;

            // Deserialize worker response endpoint information
            let worker_resp_qp_endpoint: ibverbs::QueuePairEndpoint =
                serde_json::from_str(worker_resp_qp_json).map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to deserialize worker response QP endpoint: {e}"
                    ))
                })?;
            let worker_resp_memory_region: ibverbs::RemoteMemoryRegion =
                serde_json::from_str(worker_resp_memory_json).map_err(|e| {
                    EKError::InvalidInput(format!(
                        "Failed to deserialize worker response memory region: {e}"
                    ))
                })?;

            log::info!(
                "Establishing RDMA connection to worker {} using endpoint data",
                hostname
            );
            log::debug!("Worker request QP endpoint: {:?}", worker_req_qp_endpoint);
            log::debug!(
                "Worker request memory region: {:?}",
                worker_req_memory_region
            );
            log::debug!("Worker response QP endpoint: {:?}", worker_resp_qp_endpoint);
            log::debug!(
                "Worker response memory region: {:?}",
                worker_resp_memory_region
            );

            // Connect controller's request queue to worker's request queue (for sending requests)
            {
                let mut req_queue = connection.req_queue.lock().await;
                if req_queue.is_connected() {
                    log::info!(
                        "Controller request queue already connected to worker {}, skipping",
                        hostname
                    );
                } else {
                    if let Err(e) =
                        req_queue.connect(worker_req_qp_endpoint, worker_req_memory_region.clone())
                    {
                        log::error!(
                            "Failed to connect controller request queue to worker {}: {e}",
                            hostname
                        );
                        return Err(EKError::IoError(std::io::Error::other(format!(
                            "Controller request queue connection failed: {e}"
                        ))));
                    }
                    log::info!(
                        "🚀Controller request queue connected to worker {} successfully",
                        hostname
                    );
                }
            }

            // Connect controller's response queue to worker's response queue (for receiving responses)
            {
                let mut resp_queue = connection.resp_queue.lock().await;
                if resp_queue.0.is_connected() {
                    log::info!(
                        "Controller response queue already connected to worker {}, skipping",
                        hostname
                    );
                } else {
                    if let Err(e) = resp_queue
                        .0
                        .connect(worker_resp_qp_endpoint, worker_resp_memory_region)
                    {
                        log::error!(
                            "Failed to connect controller response queue to worker {}: {e}",
                            hostname
                        );
                        return Err(EKError::IoError(std::io::Error::other(format!(
                            "Controller response queue connection failed: {e}"
                        ))));
                    }
                    log::info!(
                        "🚀Controller response queue connected to worker {} successfully",
                        hostname
                    );
                }
            }
            // Sleep for a while
            std::thread::sleep(std::time::Duration::from_secs(2));

            // Mark connection as established
            connection.connected = true;
            log::info!(
                "🚀RDMA connection to worker {} established successfully",
                hostname
            );
        }

        Ok(())
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
            reader: Box::new(StateReaderImpl::new()),
            writer: StateWriterImpl::new(),
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
