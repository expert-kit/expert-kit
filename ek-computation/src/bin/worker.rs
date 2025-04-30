#![feature(pattern)]
use ek_computation::{
    proto::ek::worker::v1::computation_service_server::ComputationServiceServer,
    worker::{StateClient, server::BasicExpertImpl},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = tokio::task::spawn(async move {
        let mut state_client = StateClient::new();
        if let Err(e) = state_client.run().await {
            log::error!("state client error {:?}", e);
        }
    });
    let srv = tokio::task::spawn(async move {
        let server = BasicExpertImpl::new();
        let addr = "[::1]:50051".parse().unwrap();

        let err = tonic::transport::Server::builder()
            .add_service(ComputationServiceServer::new(server))
            .serve(addr)
            .await;
        if let Err(e) = err {
            log::error!("server error {:?}", e);
        }
    });

    cli.await?;
    srv.await?;
    Ok(())
}
