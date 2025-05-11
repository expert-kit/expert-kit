mod core;
mod manager;
pub mod server;
pub mod state;
pub mod x;

use super::{
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_client::StateServiceClient,
    },
    worker::{server::BasicExpertImpl, state::StateClient},
};
use ek_base::{config::get_ek_settings, error::EKResult};

pub async fn worker_main() -> EKResult<()> {
    let cli = tokio::task::spawn(async move {
        let worker_id = x::get_worker_id();
        log::info!("ek hostname: {:}", worker_id);
        let control_endpoint = x::get_controller_addr();
        log::info!("control endpoint {:}", control_endpoint.uri());
        let cli = StateServiceClient::connect(control_endpoint).await.unwrap();
        let mut state_client = StateClient::new(cli, &worker_id);
        if let Err(e) = state_client.run().await {
            log::error!("state client error {:}", e);
        }
    });

    let srv = tokio::task::spawn(async move {
        let server = BasicExpertImpl::new();
        let settings = &get_ek_settings().worker;
        let addr = format!("{}:{}", settings.listen, settings.ports.main)
            .parse()
            .unwrap();
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
