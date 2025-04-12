use crate::tch_safetensors::read_safetensors;
use crate::proto::ek::worker::v1::{
        ForwardReq, ForwardResp,
        computation_service_server::ComputationService,
    };
use tonic::{Request, Response, Status};

// use ekproto::{FfnRequest, FfnResponse};

#[derive(Debug, Default)]
pub struct BasicExpertImpl {}
impl BasicExpertImpl {
    pub fn new() -> Self {
        todo!()
    }
}

#[tonic::async_trait]
impl ComputationService for BasicExpertImpl {
    async fn forward(&self, request: Request<ForwardReq>) -> Result<Response<ForwardResp>, Status> {
        let raw_tensor = request.into_inner().tensor;
        let _tensor = read_safetensors(raw_tensor.as_slice())
            .map_err(|_e| Status::invalid_argument("invalid tensor"))?;

        Ok(Response::new(ForwardResp {
            output_tensor: vec![1, 2],
        }))
    }
}
