mod core;

use state::StateInspector;
use tokio::select;
use tokio::signal;
use tokio_util::sync::CancellationToken;

mod manager;
pub mod server;
pub mod state;
pub mod x;

use crate::metrics::spawn_metrics_server;
use crate::x::get_graceful_shutdown_ch;

use super::{
    proto::ek::worker::v1::computation_service_server::ComputationServiceServer,
    worker::{server::BasicExpertImpl, state::StateClient},
};
use ek_base::{config::get_ek_settings, error::EKResult};

pub async fn worker_main() -> EKResult<()> {
    let settings = get_ek_settings();
    spawn_metrics_server(&settings.worker.metrics);

    let token = CancellationToken::new();
    let cli_cancel = token.clone();
    let cli = tokio::task::spawn(async move {
        let worker_id = x::get_worker_id();
        log::info!("ek hostname: {:}", worker_id);
        let control_endpoint = x::get_controller_addr();
        log::info!("control endpoint {:}", control_endpoint.uri());
        let mut state_client = StateClient::new(control_endpoint, &worker_id);
        if let Err(e) = state_client.run(cli_cancel).await {
            log::error!("state client error {:}", e);
        }
    });

    let srv = tokio::task::spawn(async move {
        let server = BasicExpertImpl::new();
        let settings = &get_ek_settings().worker;
        let addr = format!("{}:{}", settings.listen, settings.ports.main)
            .parse()
            .unwrap();
        log::info!("worker server listening on {}", addr);
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

    let state_inspect = StateInspector::spawn();

    select! {
        _ = cli => { },
        _ = srv => { },
        _ = state_inspect => { },
        _ = signal::ctrl_c() => {
            log::info!("ctrl-c signal received, shutting down");
            token.clone().cancel();
            let(_,rx) = get_graceful_shutdown_ch();
            rx.lock().await.recv().await;
            log::info!("graceful shutdown channel received, shutting down now");
        }
    };

    Ok(())
}
