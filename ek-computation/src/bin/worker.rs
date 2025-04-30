#![feature(pattern)]
use ek_computation::{
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_client::StateServiceClient,
    },
    worker::{server::BasicExpertImpl, state::StateClient, x},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = tokio::task::spawn(async move {
        let hn = x::get_hostname();
        let control_endpoint = x::get_control_plan_addr();
        let cli = StateServiceClient::connect(control_endpoint).await.unwrap();
        let mut state_client = StateClient::new(cli, &hn);
        if let Err(e) = state_client.run().await {
            log::error!("state client error {:?}", e);
        }
    });

    let srv = tokio::task::spawn(async move {
        let server = BasicExpertImpl::new();
        let addr = "[::1]:51234".parse().unwrap();
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
