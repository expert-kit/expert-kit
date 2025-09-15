use std::{sync::Arc, time};

use crate::{
    metrics::METRIC_WORKER_EXPERT_LOADING,
    proto::ek::{
        object::v1::Metadata,
        worker::v1::{
            ExchangeReq, ExchangeResp, exchange_resp::ExpertWithState,
            state_service_client::StateServiceClient,
        },
    },
    worker::core::EKInstanceGateAsync,
    x::{EKInstance, get_graceful_shutdown_ch},
};
use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::{ExpertKey, SafeTensorDB};
use tokio::{
    select,
    sync::{RwLock, Semaphore},
    task::{JoinHandle, JoinSet},
};
use tokio_stream::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;

use super::{
    core::get_instance_gate,
    manager::{ExpertDB, get_expert_db},
    x::{self},
    {is_rdma_queue_connected, update_rdma_connection_status},
};

pub struct StateClient {
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    worker_id: String,
    gate_async: &'static EKInstanceGateAsync, // Use async gate for state management
    controller_addr: Endpoint,
    rdma_endpoints: Option<crate::proto::ek::worker::v1::RdmaEndpointPair>,
}

impl StateClient {
    pub fn new(addr: Endpoint, worker_id: &str) -> Self {
        let edb = get_expert_db();
        let gate_async = get_instance_gate(); // Use async gate for state operations
        let tdb = SafeTensorDB::new_shared();
        Self {
            tensor_db: tdb,
            expert_db: edb,
            worker_id: worker_id.to_owned(),
            gate_async,
            controller_addr: addr,
            rdma_endpoints: None,
        }
    }

    pub fn new_with_rdma(
        addr: Endpoint,
        worker_id: &str,
        rdma_endpoints: Option<crate::proto::ek::worker::v1::RdmaEndpointPair>,
    ) -> Self {
        let edb = get_expert_db();
        let gate_async = get_instance_gate();
        let tdb = SafeTensorDB::new_shared();
        Self {
            tensor_db: tdb,
            expert_db: edb,
            worker_id: worker_id.to_owned(),
            gate_async,
            controller_addr: addr,
            rdma_endpoints,
        }
    }

    /// Generate request stream for state exchange
    fn get_request_stream(
        worker_id: String,
        rdma_endpoints: Option<crate::proto::ek::worker::v1::RdmaEndpointPair>,
    ) -> impl Stream<Item = ExchangeReq> {
        let settings = get_ek_settings();
        tokio_stream::iter(1..usize::MAX).map(move |_| ExchangeReq {
            id: worker_id.clone(),
            addr: format!(
                "http://{}:{}",
                settings.worker.broadcast, settings.worker.ports.main
            ),
            channel: if rdma_endpoints.is_some() {
                "rdma".to_string()
            } else {
                settings.worker.channel.clone()
            },
            device: settings.worker.device.clone(),
            last_will: false,
            rdma_endpoints: rdma_endpoints.clone(),
        })
    }

    /// Handle incoming stream messages from controller
    async fn handle_stream_msg(
        &mut self,
        msg: Option<Result<ExchangeResp, tonic::Status>>,
    ) -> EKResult<()> {
        if let Some(m) = msg {
            let msg = m?;
            if let Some(state) = msg.state {
                match self.handle_states(state).await {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("sync remote state error {e:?}");
                    }
                }
            }
            // Handle controller RDMA endpoints for bidirectional connection
            if let Some(controller_rdma_endpoints) = msg.rdma_endpoints {
                if is_rdma_queue_connected() {
                    log::debug!("Rdma connection already established, skip connect logic")
                } else {
                    log::info!("Received controller RDMA endpoints, establishing connection...");
                    if let Err(e) = self
                        .handle_controller_rdma_endpoints(controller_rdma_endpoints)
                        .await
                    {
                        log::error!("Failed to establish RDMA connection with controller: {e:?}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Handle controller RDMA endpoints and establish bidirectional connection
    async fn handle_controller_rdma_endpoints(
        &mut self,
        controller_endpoints: crate::proto::ek::worker::v1::RdmaEndpointPair,
    ) -> EKResult<()> {
        // Get RDMA queues from global storage
        let _req_queue = crate::worker::get_rdma_req_queue().ok_or_else(|| {
            ek_base::error::EKError::NotFound("Worker RDMA request queue not found".into())
        })?;
        let _resp_queue = crate::worker::get_rdma_resp_queue().ok_or_else(|| {
            ek_base::error::EKError::NotFound("Worker RDMA response queue not found".into())
        })?;

        // Extract controller endpoints
        let controller_req_endpoint = controller_endpoints.request_endpoint.ok_or_else(|| {
            ek_base::error::EKError::InvalidInput("Missing controller request endpoint".into())
        })?;

        let controller_resp_endpoint = controller_endpoints.response_endpoint.ok_or_else(|| {
            ek_base::error::EKError::InvalidInput("Missing controller response endpoint".into())
        })?;

        // Deserialize controller request endpoint (where worker receives requests)
        let controller_req_qp_endpoint: ibverbs::QueuePairEndpoint =
            serde_json::from_str(&controller_req_endpoint.qp_endpoint).map_err(|e| {
                ek_base::error::EKError::InvalidInput(format!(
                    "Failed to deserialize controller request QP endpoint: {e}"
                ))
            })?;

        let controller_req_memory_region: ibverbs::RemoteMemoryRegion =
            serde_json::from_str(&controller_req_endpoint.memory_region).map_err(|e| {
                ek_base::error::EKError::InvalidInput(format!(
                    "Failed to deserialize controller request memory region: {e}"
                ))
            })?;

        // Deserialize controller response endpoint (where worker sends responses)
        let controller_resp_qp_endpoint: ibverbs::QueuePairEndpoint =
            serde_json::from_str(&controller_resp_endpoint.qp_endpoint).map_err(|e| {
                ek_base::error::EKError::InvalidInput(format!(
                    "Failed to deserialize controller response QP endpoint: {e}"
                ))
            })?;

        let controller_resp_memory_region: ibverbs::RemoteMemoryRegion =
            serde_json::from_str(&controller_resp_endpoint.memory_region).map_err(|e| {
                ek_base::error::EKError::InvalidInput(format!(
                    "Failed to deserialize controller response memory region: {e}"
                ))
            })?;

        log::info!("Establishing RDMA connection with controller");
        log::info!(
            "Controller request QP endpoint: {:?}",
            controller_req_qp_endpoint
        );
        log::info!(
            "Controller request memory region: {:?}",
            controller_req_memory_region
        );
        log::info!(
            "Controller response QP endpoint: {:?}",
            controller_resp_qp_endpoint
        );
        log::info!(
            "Controller response memory region: {:?}",
            controller_resp_memory_region
        );

        // Connect worker's request queue to controller's request queue (for receiving requests)
        {
            let mut req_queue_lock = _req_queue.lock().unwrap();
            if req_queue_lock.is_connected() {
                log::info!("Worker request queue already connected to controller, skipping");
            } else {
                if let Err(e) = req_queue_lock.connect(
                    controller_req_qp_endpoint,
                    controller_req_memory_region.clone(),
                ) {
                    log::error!("Failed to connect worker request queue to controller: {e}");
                    return Err(ek_base::error::EKError::IoError(std::io::Error::other(
                        format!("RDMA request queue connection failed: {e}"),
                    )));
                }
                log::info!("🚀Worker request queue connected to controller successfully");
            }
        }

        // Connect worker's response queue to controller's response queue (for sending responses)
        {
            let mut resp_queue_lock = _resp_queue.lock().unwrap();
            if resp_queue_lock.is_connected() {
                log::info!("Worker response queue already connected to controller, skipping");
            } else {
                if let Err(e) = resp_queue_lock
                    .connect(controller_resp_qp_endpoint, controller_resp_memory_region)
                {
                    log::error!("Failed to connect worker response queue to controller: {e}");
                    return Err(ek_base::error::EKError::IoError(std::io::Error::other(
                        format!("RDMA response queue connection failed: {e}"),
                    )));
                }
                log::info!("🚀Worker response queue connected to controller successfully");
            }
        }

        update_rdma_connection_status(true);
        log::info!("🚀RDMA bidirectional connection established successfully");
        Ok(())
    }

    /// Inner run loop for state client
    async fn run_inner(&mut self, token: CancellationToken) -> EKResult<()> {
        let mut cli = StateServiceClient::connect(self.controller_addr.clone()).await?;
        let req_stream =
            Self::get_request_stream(self.worker_id.clone(), self.rdma_endpoints.clone())
                .throttle(std::time::Duration::from_secs(3));
        let res = cli.exchange(req_stream).await?;
        let mut stream = res.into_inner();
        loop {
            select! {
                msg = stream.next() => {
                    self.handle_stream_msg(msg).await?;
                },
                _ = token.cancelled() => {
                    log::info!("state client cancelled");
                    break;
                }
            }
        }
        Ok(())
    }

    /// Main run loop with reconnection logic
    pub async fn run(&mut self, token: CancellationToken) -> EKResult<()> {
        loop {
            log::info!("start sync remote state");
            select! {
                e = self.run_inner(token.clone()) => {
                    if let Err(e) = e {
                        log::error!("state client error {e:?}");
                        tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                    }
                },
                _ = token.cancelled() => {
                    log::info!("state client cancelled");
                    break;
                }
            }
        }

        let (rx, _) = get_graceful_shutdown_ch();
        let _ = rx.send(()).await;
        Ok(())
    }

    /// Spawn expert loading task
    fn spawn_expert_loading_task(
        &self,
        js: &mut JoinSet<EKResult<()>>,
        expert: &Metadata,
        token: Arc<Semaphore>,
    ) {
        let settings = get_ek_settings();
        let tdb = self.tensor_db.clone();
        let edb = self.expert_db.clone();
        let expert = expert.clone();
        let instance = EKInstance::default();
        let model_name = &settings.inference.model_name;
        let token = token.clone();
        js.spawn(async move {
            let permit = token.acquire().await.unwrap();
            let id = expert.id.clone();
            log::debug!("load expert {}", &id);
            let ek = ExpertKey::from_expert_id(model_name, &expert.id)?;
            if let Err(e) = x::load_expert_task(tdb, edb.clone(), instance, &ek).await {
                log::error!("error in load expert {e}")
            }
            drop(permit);
            Ok(())
        });
    }

    /// Remove experts that are no longer needed
    async fn remove_stale_experts(&mut self, incoming: &[Metadata], current: &[String]) {
        let incoming_ids: Vec<String> = incoming.iter().map(|e| e.id.clone()).collect();
        for e in current.iter().filter(|e| !incoming_ids.contains(e)) {
            let mut lg = self.expert_db.write().await;
            if let Err(e) = lg.remove(e).await {
                log::error!("remove expert error {e:?}");
            }
        }
    }

    /// Get experts that need to be loaded
    async fn get_new_experts(&self, incoming: &[Metadata]) -> Vec<Metadata> {
        let mut diff = vec![];
        let rg = self.expert_db.read().await;
        for expert in incoming {
            if !rg.has(&expert.id) {
                diff.push(expert.clone());
            }
        }
        diff
    }

    /// Load new experts that were received from controller
    async fn load_new_experts(&mut self, exp_incoming: &[Metadata]) -> EKResult<()> {
        let exp_new = self.get_new_experts(exp_incoming).await;
        if exp_new.is_empty() {
            return Ok(());
        }
        let now = time::Instant::now();
        log::info!("load new experts, len={}", exp_new.len());
        let mut js: JoinSet<EKResult<()>> = JoinSet::new();
        let token = Arc::new(Semaphore::new(64));
        for expert in &exp_new {
            self.spawn_expert_loading_task(&mut js, expert, token.clone());
        }

        js.join_all().await;
        let elapsed_ms = now.elapsed().as_millis();
        log::info!(
            elapsed_ms;
            "experts is loaded.",
        );
        Ok(())
    }

    /// Handle state updates from controller
    async fn handle_states(&mut self, state: ExpertWithState) -> EKResult<()> {
        if state.target.is_none() {
            return Ok(());
        }
        let slice = state.target.unwrap();

        let exp_incoming = slice.expert_meta.clone();
        self.load_new_experts(&exp_incoming).await?;

        // Use async gate for state management operations
        let exp_current = self.gate_async.current_experts().await?;
        self.remove_stale_experts(&exp_incoming, &exp_current).await;
        Ok(())
    }
}

/// Inspector for monitoring expert loading progress
pub struct StateInspector {
    edb: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
}

impl StateInspector {
    /// Inspect current loading state and update metrics
    async fn inspect(&self) {
        let settings = get_ek_settings();
        let rg = self.edb.read().await;
        let loaded = rg.loaded();
        let loading = rg.loading();
        log::info!(loaded, loading; "loading progress");

        // Update metrics
        METRIC_WORKER_EXPERT_LOADING
            .with_label_values(&[
                settings.worker.id.as_str(),
                settings.inference.model_name.as_str(),
                "loaded",
            ])
            .set(loaded as i64);

        METRIC_WORKER_EXPERT_LOADING
            .with_label_values(&[
                settings.worker.id.as_str(),
                settings.inference.model_name.as_str(),
                "loading",
            ])
            .set(loading as i64);
    }

    /// Main run loop for state inspector
    pub async fn run(&self) {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            self.inspect().await;
        }
    }

    /// Spawn state inspector task
    pub fn spawn() -> JoinHandle<()> {
        let si = StateInspector {
            edb: get_expert_db(),
        };
        tokio::task::spawn(async move { si.run().await })
    }
}
