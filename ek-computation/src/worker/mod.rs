use ek_base::error::EKResult;
mod gate;
mod manager;
mod x;
use gate::GlobalEKInstanceGate;
pub mod server;
use log::info;
use tokio::sync::mpsc;
use tonic::transport::Channel;

use crate::proto::ek::{
    object::v1::ExpertSlice,
    worker::v1::{
        RetrieveStateReq, retrieve_state_resp::ExpertWithState,
        state_service_client::StateServiceClient,
    },
};
use tokio_stream::StreamExt;
pub struct StateClient {
    cli: StateServiceClient<Channel>,
    hostname: String,
    create_ch: mpsc::Sender<ExpertSlice>,
    gate: GlobalEKInstanceGate,
}

impl Default for StateClient {
    fn default() -> Self {
        Self::new()
    }
}

impl StateClient {
    pub fn new() -> Self {
        todo!()
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
                // todo
            } else {
                // create
            }
        }
        // remove
        incoming_experts
            .iter()
            .filter(|e| !current_experts.contains(&e.id))
            .for_each(|e| {
                //remove
            });
        Ok(())
    }
}
