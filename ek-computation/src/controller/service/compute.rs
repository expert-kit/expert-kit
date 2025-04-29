use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    controller::executor::Executor,
    proto::ek::worker::v1::{self, computation_service_server::ComputationService},
};

pub struct ComputationProxyServiceImpl {
    executor: Arc<Mutex<dyn Executor + Send + Sync>>,
}

#[async_trait::async_trait]
impl ComputationService for ComputationProxyServiceImpl {
    async fn forward(
        &self,
        request: tonic::Request<v1::ForwardReq>,
    ) -> Result<tonic::Response<v1::ForwardResp>, tonic::Status> {
        let mut lg = self.executor.lock().await;
        let mut rx = lg.submit(request.get_ref()).await?;
        let res = rx.recv().await;
        if let Some(resp) = res {
            Ok(tonic::Response::new(resp.as_ref().clone()))
        } else {
            Err(tonic::Status::internal("forward error"))
        }
    }
}
