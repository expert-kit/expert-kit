use ekproto::expert_computation_server::{ExpertComputation, ExpertComputationServer};
use ekproto::{ExpertForwardReply, ExpertForwardRequest};
use tch_safetensors::read_safetensors;
use tonic::{Request, Response, Status, transport::Server};
pub mod tch_safetensors;

// use ekproto::{FfnRequest, FfnResponse};

#[derive(Debug, Default)]
pub struct BasicExpertImpl {}

#[tonic::async_trait]
impl ExpertComputation for BasicExpertImpl {
    async fn forward(
        &self,
        request: Request<ExpertForwardRequest>,
    ) -> Result<Response<ExpertForwardReply>, Status> {
        let raw_tensor = request.into_inner().tensor;
        let _tensor = read_safetensors(raw_tensor.as_slice())
            .map_err(|_e| Status::invalid_argument("invalid tensor"))?;

        Ok(Response::new(ExpertForwardReply {
            output_tensor: vec![],
        }))
    }
}

pub mod ekproto {
    tonic::include_proto!("ek");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "[::1]:50051".parse()?;
    let computation = BasicExpertImpl {};
    Server::builder()
        .add_service(ExpertComputationServer::new(computation))
        .serve(addr)
        .await?;

    Ok(())
}
