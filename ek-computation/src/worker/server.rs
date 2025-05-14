use std::time::Instant;

use crate::proto::ek::worker::v1::{
    ForwardReq, ForwardResp, computation_service_server::ComputationService,
};
use tonic::{Request, Response, Status};

use super::core::{GlobalEKInstanceGate, get_instance_gate};

// use ekproto::{FfnRequest, FfnResponse};

#[derive(Debug, Default)]
pub struct BasicExpertImpl {
    gate: GlobalEKInstanceGate,
}
impl BasicExpertImpl {
    pub fn new() -> Self {
        let gate = get_instance_gate();
        Self { gate }
    }
}

#[tonic::async_trait]
impl ComputationService for BasicExpertImpl {
    async fn forward(&self, request: Request<ForwardReq>) -> Result<Response<ForwardResp>, Status> {
        log::info!("forward request: seq={}", request.get_ref().sequences.len());
        let start = Instant::now();
        let guard = self.gate.lock().await;
        let res = guard.forward(request.into_inner()).await.map_err(|e| {
            log::error!("forward error {:?}", e);
            Status::internal("forward error")
        })?;
        log::info!(
            "forward request: elapsed_ms={:?}",
            start.elapsed().as_millis()
        );

        Ok(Response::new(res))
    }
}
