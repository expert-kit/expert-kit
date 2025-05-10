use std::{sync::Arc, time};

use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::safetensor::{ExpertKey, SafeTensorDB};
use log::info;
use tokio::{sync::RwLock, task::JoinHandle};
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
        let req = RetrieveStateReq {
            hostname: self.worker_id.clone(),
        };
        let res = self.cli.retrieve(req).await.unwrap();
        let mut stream = res.into_inner();
        while let Some(msg) = stream.next().await {
            let msg = msg?;
            if let Some(state) = msg.state {
                match self.handle_states(state).await {
                    Ok(_) => {
                        info!("sync remote state success");
                    }
                    Err(e) => {
                        log::error!("sync remote state error {:?}", e);
                    }
                }
            }
        }
        Ok(())
    }

    async fn spawn_expert_loading_task(&self, expert: &Metadata) {
        let settings = get_ek_settings();
        let tdb = self.tensor_db.clone();
        let edb = self.expert_db.clone();
        let expert = expert.clone();
        // TODO(multi-model): read instance here
        let instance = EKInstance::default();
        let model_name = &settings.model_name;
        let _model: JoinHandle<EKResult<()>> = tokio::spawn(async move {
            let now = time::Instant::now();
            let id = expert.id.clone();
            log::info!("load expert {}", &id);
            let ek = ExpertKey::from_expert_id(model_name, &expert.id)?;
            if let Err(e) = x::load_expert_task(tdb, edb.clone(), instance, &ek).await {
                log::error!("error in load expert {}", e)
            }
            let rg = edb.read().await;
            let loaded = rg.loaded();
            let loading = rg.loading();
            log::info!(
                "load expert {} done, currently loaded={} loading={}, elapsed_ms={},",
                &id,
                loaded,
                loading,
                now.elapsed().as_millis()
            );
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
        let edb = self.expert_db.clone();
        let rg = edb.read().await;
        for expert in incoming {
            if !rg.has(&expert.id) {
                diff.push(expert.clone());
            }
        }
        diff
    }

    async fn handle_states(&mut self, state: ExpertWithState) -> EKResult<()> {
        let exp_current = self.gate.lock().await.current_experts().await?;
        if state.target.is_none() {
            return Ok(());
        }
        let slice = state.target.unwrap();
        let exp_incoming = slice.expert_meta.clone();
        let exp_new = self.get_new_experts(&exp_incoming).await;
        for expert in exp_new {
            if exp_current.contains(&expert.id) {
                // update
                // TODO: change replication?
                continue;
            }
            self.spawn_expert_loading_task(&expert).await;
        }

        self.remove_stale_experts(&exp_incoming, &exp_current)
            .await;
        Ok(())
    }
}
