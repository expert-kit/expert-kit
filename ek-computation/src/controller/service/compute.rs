use std::sync::Arc;

use tokio::sync::Mutex;

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
        let mut lg = self.executor.lock().await;
        let mut rx = lg.submit(request.get_ref()).await?;
        let exec_bg = self.executor.clone();
        tokio::spawn(async move {
            let mut lg = exec_bg.lock().await;
            let res = lg.exec().await;
            if let Err(err) = res {
                log::error!("executor error: {}", err);
            }
        });
        let res = rx.recv().await;
        if let Some(resp) = res {
            Ok(tonic::Response::new(resp.as_ref().clone()))
        } else {
            Err(tonic::Status::internal("forward error"))
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
