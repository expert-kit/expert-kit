use std::sync::Arc;

use crate::{
    controller::dispatcher::{DISPATCHER, Dispatcher},
    proto::ek::{
        object::v1::ExpertSlice,
        worker::v1::{self, retrieve_state_resp::ExpertWithState},
    },
    state::{
        io::StateWriter,
        models::NewNode,
        writer::{StateWriterImpl, get_state_writer},
    },
};
use ek_base::error::EKError;
use tokio::sync::{RwLock, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Result, Status, Streaming};

use crate::proto::ek::worker::v1::{RetrieveStateResp, state_service_server::StateService};
pub struct StateServerImpl {
    writer: Arc<RwLock<dyn StateWriter + Send + Sync>>,
}

impl StateServerImpl {
    async fn listen_worker_ping(
        mut req: tonic::Request<Streaming<v1::RetrieveStateReq>>,
        id: String,
    ) {
        let w = StateWriterImpl {};
        while let Some(msg) = req.get_mut().message().await.unwrap() {
            let err = w
                .node_upsert(NewNode {
                    hostname: msg.id.clone(),
                    device: msg.device.clone(),
                    config: serde_json::json!({
                        "addr": msg.addr.clone(),
                        "channel": msg.channel.clone(),
                    }),
                })
                .await;
            if let Err(e) = err {
                log::error!("worker ping error, can not upsert node: {}", e);
            }

            let e = w.node_update_seen(&msg.id).await;
            if let Err(e) = e {
                log::error!("worker ping error: {}", e);
            }
        }
        log::warn!("worker ping stream closed for worker_id={}", id);
    }
}

#[tonic::async_trait]
impl StateService for StateServerImpl {
    // type RetrieveStream = Pin<Box<dyn Stream<Item = Result<RetrieveStateResp>> + Send + 'static>>;
    type RetrieveStream = ReceiverStream<Result<RetrieveStateResp, Status>>;

    async fn retrieve(
        &self,
        mut request: tonic::Request<Streaming<v1::RetrieveStateReq>>,
    ) -> Result<Response<Self::RetrieveStream>> {
        let mut lg = DISPATCHER.lock().await;
        let (stream_tx, stream_rx) = mpsc::channel(4);
        let first_message = request
            .get_mut()
            .message()
            .await?
            .ok_or(Status::invalid_argument("no message"))?;
        let worker_id = first_message.id.clone();
        tokio::spawn(async move {
            StateServerImpl::listen_worker_ping(request, worker_id.clone()).await;
        });

        let mut rx = lg.subscribe(&first_message.id).await;
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
