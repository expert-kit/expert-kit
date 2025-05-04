use std::sync::Arc;

use tokio::sync::{Mutex, mpsc};

use crate::{
    controller::executor::{Executor, get_executor},
    proto::ek::worker::v1::{self, computation_service_server::ComputationService},
};

pub struct ComputationProxyServiceImpl {
    executor: Arc<Mutex<dyn Executor + Send>>,
}

#[async_trait::async_trait]
impl ComputationService for ComputationProxyServiceImpl {
    async fn forward(
        &self,
        request: tonic::Request<v1::ForwardReq>,
    ) -> Result<tonic::Response<v1::ForwardResp>, tonic::Status> {
        let mut rx = {
            let mut lg = self.executor.lock().await;
            
            lg.submit(request.get_ref()).await?
        };
        let exec_bg = self.executor.clone();
        let (err_tx, mut err_rx) = mpsc::channel(1);
        tokio::spawn(async move {
            let mut lg = exec_bg.lock().await;
            let res = lg.exec().await;
            if let Err(err) = res {
                log::error!("executor error: {}", err);
                err_tx.send(err).await.unwrap();
            }
        });
        loop {
            tokio::select! {
                err = err_rx.recv() => {
                    if let Some(err) = err {
                        log::error!("executor error: {:?}", err);
                        return Err(tonic::Status::internal(format!("executor error: {:?}", err)))
                    }
                    continue
                }
                res = rx.recv() => {
                    if let Some(resp) = res {
                        return Ok(tonic::Response::new(resp.as_ref().clone()));
                    } else {
                        return Err(tonic::Status::internal("forward error: no data"));
                    }
                }
            }
        }
    }
}

impl Default for ComputationProxyServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputationProxyServiceImpl {
    pub fn new() -> Self {
        Self {
            executor: get_executor(),
        }
    }
}
