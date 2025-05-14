mod core;
mod manager;
pub mod server;
pub mod state;
pub mod x;

use super::{
    proto::ek::worker::v1::computation_service_server::ComputationServiceServer,
    worker::{server::BasicExpertImpl, state::StateClient},
};
use ek_base::{config::get_ek_settings, error::EKResult};

pub async fn worker_main() -> EKResult<()> {
    let cli = tokio::task::spawn(async move {
        let worker_id = x::get_worker_id();
        log::info!("ek hostname: {:}", worker_id);
        let control_endpoint = x::get_controller_addr();
        log::info!("control endpoint {:}", control_endpoint.uri());
        let mut state_client = StateClient::new( control_endpoint, &worker_id);
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
            .add_service(
                ComputationServiceServer::new(server)
                    .max_decoding_message_size(200 * 1024 * 1024)
                    .max_encoding_message_size(200 * 1024 * 1024),
            )
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
