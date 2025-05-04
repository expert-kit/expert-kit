use std::sync::Arc;

use crate::{
    controller::dispatcher::{DISPATCHER, Dispatcher},
    proto::ek::{
        object::v1::ExpertSlice,
        worker::v1::{self, retrieve_state_resp::ExpertWithState},
    },
    state::io::{StateWriter, get_state_writer},
};
use ek_base::error::EKError;
use tokio::sync::{RwLock, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Result, Status};

use crate::proto::ek::worker::v1::{RetrieveStateResp, state_service_server::StateService};
pub struct StateServerImpl {
    writer: Arc<RwLock<dyn StateWriter + Send + Sync>>,
}

#[tonic::async_trait]
impl StateService for StateServerImpl {
    // type RetrieveStream = Pin<Box<dyn Stream<Item = Result<RetrieveStateResp>> + Send + 'static>>;
    type RetrieveStream = ReceiverStream<Result<RetrieveStateResp, Status>>;

    async fn retrieve(
        &self,
        request: tonic::Request<v1::RetrieveStateReq>,
    ) -> Result<Response<Self::RetrieveStream>> {
        let mut lg = DISPATCHER.lock().await;
        let req = request.get_ref();
        let (stream_tx, stream_rx) = mpsc::channel(4);
        let mut rx = lg.subscribe(&req.hostname).await;
        tokio::spawn(async move {
            while let Some(t) = rx.recv().await {
                let resp = RetrieveStateResp {
                    state: Some(ExpertWithState {
                        target: Some(ExpertSlice::from(t)),
                    }),
                };
                if let Err(e) = stream_tx.send(Ok(resp)).await {
                    log::error!("stream error: {}", e)
                };
            }
        });
        Ok(Response::new(Self::RetrieveStream::new(stream_rx)))
    }

    async fn update(
        &self,
        request: tonic::Request<v1::UpdateStateReq>,
    ) -> Result<Response<v1::UpdateStateResp>> {
        let mut lg = self.writer.write().await;
        let req = request.get_ref();
        let slice = req.target.clone().ok_or(EKError::DBError())?;
        lg.upd_expert_state(&req.hostname, slice).await?;
        let resp = v1::UpdateStateResp {};
        Ok(Response::new(resp))
    }
}

impl Default for StateServerImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl StateServerImpl {
    pub fn new() -> Self {
        Self {
            writer: get_state_writer(),
        }
    }
}
