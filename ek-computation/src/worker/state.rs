use std::sync::Arc;

use ek_base::error::EKResult;
use ek_db::safetensor::SafeTensorDB;
use log::info;
use tokio::sync::RwLock;
use tonic::transport::Channel;

use crate::{
    proto::ek::worker::v1::{
        RetrieveStateReq, retrieve_state_resp::ExpertWithState,
        state_service_client::StateServiceClient,
    },
    x::EKInstance,
};
use tokio_stream::StreamExt;

use super::{
    core::{GlobalEKInstanceGate, get_instance_gate},
    manager::{ExpertDB, get_expert_db},
    x::{self, get_s3_dal_operator},
};
pub struct StateClient {
    tensor_db: Arc<RwLock<SafeTensorDB>>,
    expert_db: Arc<RwLock<dyn ExpertDB + Sync + Send + 'static>>,
    cli: StateServiceClient<Channel>,
    hostname: String,
    instance: EKInstance,
    gate: GlobalEKInstanceGate,
}

impl StateClient {
    pub fn new(cli: StateServiceClient<Channel>, hostname: &str) -> Self {
        let edb = get_expert_db();
        let gate = get_instance_gate();
        let op = get_s3_dal_operator();
        let tdb = SafeTensorDB::new_shared(op);

        Self {
            tensor_db: tdb,
            expert_db: edb,
            cli,
            hostname: hostname.to_owned(),
            instance: EKInstance::default(),
            gate,
        }
    }
    pub async fn run(&mut self) -> EKResult<()> {
        let req = RetrieveStateReq {
            hostname: self.hostname.clone(),
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

    async fn handle_states(&mut self, state: ExpertWithState) -> EKResult<()> {
        let current_experts = self.gate.lock().await.current_experts().await?;
        if state.target.is_none() {
            return Ok(());
        }
        let slice = state.target.unwrap();
        let incoming_experts = slice.expert_meta.clone();
        for expert in slice.expert_meta {
            if current_experts.contains(&expert.id) {
                // update
                // TODO: change replication?
            } else {
                let tdb = self.tensor_db.clone();
                let edb = self.expert_db.clone();
                let expert = expert.clone();
                let instance = self.instance;
                tokio::spawn(async move {
                    if let Err(e) = x::load_expert_task(tdb, edb, instance, expert).await {
                        log::error!("error in load expert {}", e)
                    }
                });
            }
        }
        {
            // remove
            let mut lg = self.expert_db.write().await;
            for e in incoming_experts
                .iter()
                .filter(|e| !current_experts.contains(&e.id))
            {
                if let Err(e) = lg.remove(&e.id).await {
                    log::error!("remove expert error {:?}", e);
                }
            }
        }
        Ok(())
    }
}
