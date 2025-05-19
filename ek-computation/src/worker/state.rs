use std::{sync::Arc, time};

use crate::{
    proto::ek::{
        object::v1::Metadata,
        worker::v1::{
            ExchangeReq, ExchangeResp, exchange_resp::ExpertWithState,
            state_service_client::StateServiceClient,
        },
    },
    x::{EKInstance, get_graceful_shutdown_ch},
};
use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::{ExpertKey, SafeTensorDB};
use tokio::{
    select,
    sync::{RwLock, Semaphore},
    task::JoinSet,
};
use tokio_stream::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::transport::Endpoint;

use super::{
    core::{GlobalEKInstanceGate, get_instance_gate},
    manager::{ExpertDB, get_expert_db},
    x::{self},
};
pub struct StateClient {
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    worker_id: String,
    gate: GlobalEKInstanceGate,
    controller_addr: Endpoint,
}

impl StateClient {
    pub fn new(addr: Endpoint, worker_id: &str) -> Self {
        let edb = get_expert_db();
        let gate = get_instance_gate();
        let tdb = SafeTensorDB::new_shared();
        Self {
            tensor_db: tdb,
            expert_db: edb,
            worker_id: worker_id.to_owned(),
            gate,
            controller_addr: addr,
        }
    }

    async fn get_request_stream(worker_id: String) -> impl Stream<Item = ExchangeReq> {
        let settings = get_ek_settings();
        let dev = settings.worker.device.clone();
        let dev = dev.unwrap_or("cpu".to_string());
        tokio_stream::iter(1..usize::MAX).map(move |_| ExchangeReq {
            id: worker_id.clone(),
            addr: format!(
                "http://{}:{}",
                settings.worker.broadcast, settings.worker.ports.main
            ),
            channel: "grpc".to_string(),
            device: dev.clone(),
            last_will: false,
        })
    }

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
                        log::error!("sync remote state error {:?}", e);
                    }
                }
            }
        }
        Ok(())
    }

    async fn run_inner(&mut self, token: CancellationToken) -> EKResult<()> {
        let mut cli = StateServiceClient::connect(self.controller_addr.clone()).await?;
        let req_stream = StateClient::get_request_stream(self.worker_id.to_owned())
            .await
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

    pub async fn run(&mut self, token: CancellationToken) -> EKResult<()> {
        loop {
            log::info!("start sync remote state");
            select! {
                e= self.run_inner(token.clone()) =>{

                    if let Err(e) = e {
                        log::error!("state client error {:?}", e);
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
                log::error!("error in load expert {}", e)
            }
            drop(permit);
            Ok(())
        });
    }

    async fn remove_stale_experts(&mut self, incoming: &[Metadata], current: &[String]) {
        let mut lg = self.expert_db.write().await;
        for e in incoming.iter().filter(|e| !current.contains(&e.id)) {
            if let Err(e) = lg.remove(&e.id).await {
                log::error!("remove expert error {:?}", e);
            }
        }
    }

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

        let edb = self.expert_db.clone();
        let v = tokio::spawn(async move {
            let start = time::Instant::now();
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
                {
                    let rg = edb.read().await;
                    let loaded = rg.loaded();
                    let loading = rg.loading();
                    log::info!(
                        "loading progress: loaded={} loading={} elapsed_ms={},",
                        loaded,
                        loading,
                        start.elapsed().as_millis()
                    );
                }
            }
        });

        js.join_all().await;
        v.abort();
        log::info!(
            "experts is loaded. elapsed_ms={}",
            now.elapsed().as_millis()
        );
        Ok(())
    }

    async fn handle_states(&mut self, state: ExpertWithState) -> EKResult<()> {
        if state.target.is_none() {
            return Ok(());
        }
        let slice = state.target.unwrap();

        let exp_incoming = slice.expert_meta.clone();
        self.load_new_experts(&exp_incoming).await?;

        let exp_current = self.gate.read().await.current_experts().await?;
        self.remove_stale_experts(&exp_incoming, &exp_current).await;
        Ok(())
    }
}
