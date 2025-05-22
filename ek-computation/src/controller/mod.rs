pub mod dispatcher;
pub mod executor;
pub mod poller;
pub mod registry;
pub mod service;


use crate::metrics;
use ek_base::error::EKResult;
use metrics::spawn_metrics_server;

use super::{
    controller::{self, poller::start_poll},
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_server::StateServiceServer,
    },
};

pub async fn controller_main() -> EKResult<()> {
    let settings = ek_base::config::get_ek_settings();

    spawn_metrics_server("0.0.0.0:9080");

    let state_srv = tokio::task::spawn(async {
        let srv = controller::service::state::StateServerImpl::new();
        let intra_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.intra
        )
        .parse()
        .unwrap();
        log::info!("state server listening on {}", intra_addr);
        let err = tonic::transport::Server::builder()
            .add_service(StateServiceServer::new(srv))
            .serve(intra_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {:?}", e);
        }
    });

    let computation_srv = tokio::task::spawn(async {
        let srv = controller::service::compute::ComputationProxyServiceImpl::new();
        let inter_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.inter
        )
        .parse()
        .unwrap();
        log::info!("computation server listening on {}", inter_addr);
        let err = tonic::transport::Server::builder()
            .add_service(
                ComputationServiceServer::new(srv)
                    .max_decoding_message_size(1024 * 1024 * 1024)
                    .max_encoding_message_size(1024 * 1024 * 1024),
            )
            .serve(inter_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {:?}", e);
        }
    });

    start_poll();

    log::info!("expert kit controller started");
    state_srv.await?;
    computation_srv.await?;
    Ok(())
}
