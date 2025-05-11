use std::{sync::Arc, time};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::{ExpertKey, SafeTensorDB};
use tokio::{sync::RwLock, task::JoinSet};
use tonic::transport::Channel;

use crate::{
    proto::ek::{
        object::v1::Metadata,
        worker::v1::{
            RetrieveStateReq, retrieve_state_resp::ExpertWithState,
            state_service_client::StateServiceClient,
        },
    },
    x::EKInstance,
};
use tokio_stream::StreamExt;

use super::{
    core::{GlobalEKInstanceGate, get_instance_gate},
    manager::{ExpertDB, get_expert_db},
    x::{self},
};
pub struct StateClient {
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    cli: StateServiceClient<Channel>,
    worker_id: String,
    gate: GlobalEKInstanceGate,
}

impl StateClient {
    pub fn new(cli: StateServiceClient<Channel>, worker_id: &str) -> Self {
        let edb = get_expert_db();
        let gate = get_instance_gate();
        let tdb = SafeTensorDB::new_shared();
        Self {
            tensor_db: tdb,
            expert_db: edb,
            cli,
            worker_id: worker_id.to_owned(),
            gate,
        }
    }
    pub async fn run(&mut self) -> EKResult<()> {
        log::info!("start sync remote state");
        let req = RetrieveStateReq {
            hostname: self.worker_id.clone(),
        };
        let res = self.cli.retrieve(req).await.unwrap();
        let mut stream = res.into_inner();
        while let Some(msg) = stream.next().await {
            let msg = msg?;
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

    fn spawn_expert_loading_task(&self, js: &mut JoinSet<EKResult<()>>, expert: &Metadata) {
        let settings = get_ek_settings();
        let tdb = self.tensor_db.clone();
        let edb = self.expert_db.clone();
        let expert = expert.clone();
        let instance = EKInstance::default();
        let model_name = &settings.model_name;
        js.spawn(async move {
            let id = expert.id.clone();
            log::debug!("load expert {}", &id);
            let ek = ExpertKey::from_expert_id(model_name, &expert.id)?;
            if let Err(e) = x::load_expert_task(tdb, edb.clone(), instance, &ek).await {
                log::error!("error in load expert {}", e)
            }
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
        for expert in &exp_new {
            self.spawn_expert_loading_task(&mut js, expert);
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

        let exp_current = self.gate.lock().await.current_experts().await?;
        self.remove_stale_experts(&exp_incoming, &exp_current).await;
        Ok(())
    }
}
