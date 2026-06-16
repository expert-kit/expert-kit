use std::{sync::Arc, time};

use crate::{
    metrics::METRIC_WORKER_EXPERT_LOADING,
    proto::ek::{
        control::v1::{
            GetRoutingReq, SubscribeRoutingReq, routing_service_client::RoutingServiceClient,
            routing_update::ChangeType,
        },
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
use ek_db::{safetensor::ExpertKey, weight_manager::LocalWeightManager};
use tokio::{
    select,
    sync::{RwLock, Semaphore, mpsc},
    task::{JoinHandle, JoinSet},
};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;

use super::{
    core::get_instance_gate,
    manager::{ExpertDB, get_expert_db},
    x::{self},
    {close_rdma_queues, is_rdma_queue_connected},
};

pub struct StateClient {
    weight_manager: Arc<LocalWeightManager>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    worker_id: String,
    gate_async: &'static EKInstanceGateAsync,
    controller_addr: Endpoint,
    rdma_tcp_port: Option<u16>,
    preemption_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl StateClient {
    pub fn new(addr: Endpoint, worker_id: &str, wm: Arc<LocalWeightManager>) -> Self {
        let edb = get_expert_db();
        let gate_async = get_instance_gate();
        Self {
            weight_manager: wm,
            expert_db: edb,
            worker_id: worker_id.to_owned(),
            gate_async,
            controller_addr: addr,
            rdma_tcp_port: None,
            preemption_tx: None,
        }
    }

    pub fn new_with_rdma_tcp_port(
        addr: Endpoint,
        worker_id: &str,
        rdma_tcp_port: Option<u16>,
        wm: Arc<LocalWeightManager>,
    ) -> Self {
        let edb = get_expert_db();
        let gate_async = get_instance_gate();
        Self {
            weight_manager: wm,
            expert_db: edb,
            worker_id: worker_id.to_owned(),
            gate_async,
            controller_addr: addr,
            rdma_tcp_port,
            preemption_tx: None,
        }
    }

    pub fn set_preemption_notifier(&mut self, tx: tokio::sync::oneshot::Sender<()>) {
        self.preemption_tx = Some(tx);
    }

    /// Generate request stream for state exchange with loaded experts reporting.
    /// Spawns a background task that generates heartbeats every 3 seconds,
    /// including the list of currently loaded experts and the WM gRPC address.
    fn start_heartbeat_stream(
        worker_id: String,
        rdma_tcp_port: Option<u16>,
        gate_async: &'static EKInstanceGateAsync,
        cancel_token: CancellationToken,
    ) -> ReceiverStream<ExchangeReq> {
        let (tx, rx) = mpsc::channel::<ExchangeReq>(4);

        tokio::spawn(async move {
            loop {
                let settings = get_ek_settings();
                // Get currently loaded experts
                let loaded_experts = match gate_async.current_experts().await {
                    Ok(experts) => experts,
                    Err(e) => {
                        log::warn!("Failed to get loaded experts for heartbeat: {}", e);
                        vec![]
                    }
                };

                let heartbeat = ExchangeReq {
                    id: worker_id.clone(),
                    addr: format!(
                        "http://{}:{}",
                        settings.worker.broadcast, settings.worker.ports.main
                    ),
                    channel: if rdma_tcp_port.is_some() {
                        "rdma".to_string()
                    } else {
                        settings.worker.channel.clone()
                    },
                    device: settings.worker.device.clone(),
                    last_will: super::get_last_will(),
                    rdma_tcp_port: rdma_tcp_port.map(|p| p as u32).unwrap_or(0),
                    loaded_experts,
                    wm_addr: format!("http://{}", settings.weight.wm_broadcast),
                    mem_capacity_mb: settings.worker.mem_capacity_mb,
                    expert_request_counts: super::server::snapshot_and_reset(),
                };

                if tx.send(heartbeat).await.is_err() {
                    log::debug!("Heartbeat stream closed, stopping heartbeat generator");
                    break;
                }

                select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {},
                    _ = cancel_token.cancelled() => {
                        log::debug!("Heartbeat generator cancelled");
                        break;
                    }
                }
            }
        });

        ReceiverStream::new(rx)
    }

    /// Handle incoming stream messages from controller
    async fn handle_stream_msg(
        &mut self,
        msg: Option<Result<ExchangeResp, tonic::Status>>,
    ) -> EKResult<()> {
        if let Some(m) = msg {
            let msg = m?;
            if msg.preemption_complete {
                log::info!("Received preemption_complete from controller");
                if let Some(tx) = self.preemption_tx.take() {
                    let _ = tx.send(());
                }
            }
            if let Some(state) = msg.state {
                match self.handle_states(state).await {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("sync remote state error {e:?}");
                    }
                }
            }
        }
        Ok(())
    }

    /// Spawn a task that subscribes to routing updates and populates the WM peer index.
    fn spawn_routing_subscription(
        controller_addr: Endpoint,
        wm: Arc<LocalWeightManager>,
        token: CancellationToken,
    ) {
        tokio::spawn(async move {
            // Fetch initial routing table to pre-populate the peer index.
            match RoutingServiceClient::connect(controller_addr.clone()).await {
                Ok(mut cli) => {
                    let resp = cli
                        .get_routing(GetRoutingReq { expert_ids: vec![] })
                        .await;
                    match resp {
                        Ok(resp) => {
                            for (expert_id, ep_list) in resp.into_inner().routing {
                                let wm_addrs: Vec<String> = ep_list
                                    .endpoints
                                    .iter()
                                    .filter(|e| !e.wm_addr.is_empty())
                                    .map(|e| e.wm_addr.clone())
                                    .collect();
                                if !wm_addrs.is_empty() {
                                    wm.update_peer_index(expert_id, wm_addrs);
                                }
                            }
                            log::info!("WM peer index populated from initial routing table");
                        }
                        Err(e) => {
                            log::warn!("Failed to fetch initial routing table for WM: {e}");
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to connect to RoutingService for WM peer index: {e}");
                    return;
                }
            }

            // Subscribe to incremental routing updates.
            let mut routing_cli =
                match RoutingServiceClient::connect(controller_addr).await {
                    Ok(c) => c,
                    Err(e) => {
                        log::warn!("Failed to connect for routing subscription: {e}");
                        return;
                    }
                };
            let res = routing_cli
                .subscribe_routing_updates(SubscribeRoutingReq { current_version: 0 })
                .await;
            let mut stream = match res {
                Ok(r) => r.into_inner(),
                Err(e) => {
                    log::warn!("Failed to subscribe to routing updates: {e}");
                    return;
                }
            };

            loop {
                select! {
                    msg = stream.next() => {
                        match msg {
                            Some(Ok(update)) => {
                                let expert_id = update.expert_id.clone();
                                match update.r#type() {
                                    ChangeType::Removed => {
                                        wm.remove_from_peer_index(&expert_id);
                                    }
                                    _ => {
                                        if let Some(ep_list) = update.endpoints {
                                            let wm_addrs: Vec<String> = ep_list
                                                .endpoints
                                                .iter()
                                                .filter(|e| !e.wm_addr.is_empty())
                                                .map(|e| e.wm_addr.clone())
                                                .collect();
                                            if !wm_addrs.is_empty() {
                                                wm.update_peer_index(expert_id, wm_addrs);
                                            }
                                        }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                log::warn!("Routing update stream error: {e}");
                                break;
                            }
                            None => break,
                        }
                    }
                    _ = token.cancelled() => break,
                }
            }
        });
    }

    /// Inner run loop for state client
    async fn run_inner(&mut self, token: CancellationToken) -> EKResult<()> {
        // Spawn routing subscription to keep WM peer index up-to-date.
        Self::spawn_routing_subscription(
            self.controller_addr.clone(),
            self.weight_manager.clone(),
            token.clone(),
        );

        let mut cli = StateServiceClient::connect(self.controller_addr.clone()).await?;
        let req_stream = Self::start_heartbeat_stream(
            self.worker_id.clone(),
            self.rdma_tcp_port,
            self.gate_async,
            token.clone(),
        );
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
                        if self.rdma_tcp_port.is_some() && is_rdma_queue_connected() {
                            log::info!("🚀 rdma connection lost, resetting rdma queues");
                            close_rdma_queues();
                            log::info!("🚀 rdma queues reset complete");
                        }
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
        let wm = self.weight_manager.clone();
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
            if let Err(e) = x::load_expert_task(wm, edb.clone(), instance, &ek).await {
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
    async fn inspect(&self) {
        let settings = get_ek_settings();
        let rg = self.edb.read().await;
        let loaded = rg.loaded();
        let loading = rg.loading();
        log::info!(loaded, loading; "loading progress");

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

    pub async fn run(&self) {
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            self.inspect().await;
        }
    }

    pub fn spawn() -> JoinHandle<()> {
        let si = StateInspector {
            edb: get_expert_db(),
        };
        tokio::task::spawn(async move { si.run().await })
    }
}
