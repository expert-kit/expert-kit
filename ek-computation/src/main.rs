#![feature(pattern)]
mod proto;
use proto::ek::worker::v1::{
    ForwardReq, ForwardResp,
    computation_service_server::{ComputationService, ComputationServiceServer},
};
// use ekproto::{ExpertForwardReply, ExpertForwardRequest};
use tch_safetensors::read_safetensors;
use tonic::{Request, Response, Status, transport::Server};
pub mod tch_safetensors;

// use ekproto::{FfnRequest, FfnResponse};

#[derive(Debug, Default)]
pub struct BasicExpertImpl {}

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let computation = BasicExpertImpl {};
    Server::builder()
        .add_service(ComputationServiceServer::new(computation))
        .serve(addr)
        .await?;

    Ok(())
}
