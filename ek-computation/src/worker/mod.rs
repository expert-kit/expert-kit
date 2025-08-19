mod core;

use std::env;
use std::sync::Arc;
use std::sync::Mutex;
use std::time;
use std::time::Duration;

use ek_base::tracing::grpc::OTelGrpcServerMiddleware;
use state::StateInspector;
use tokio::select;
use tokio::signal;
use tokio_util::sync::CancellationToken;

mod manager;
pub mod server;
pub mod state;
pub mod x;

use crate::controller::registry::LocalShmWorkerReq;
use crate::controller::registry::LocalShmWorkerResp;
use crate::metrics::spawn_metrics_server;
use crate::proto::ek::worker::v1::computation_service_server::ComputationServiceServer;
use crate::shmq::ShmQueue;
use crate::worker::core::EKInstanceGateSync;
use crate::worker::server::BasicExpertImpl;
use crate::x::get_graceful_shutdown_ch;

use super::worker::state::StateClient;
use ek_base::{config::get_ek_settings, error::EKResult};

/// Main worker entry point
pub async fn worker_main() -> EKResult<()> {
    let settings = get_ek_settings();

    spawn_metrics_server(&settings.worker.metrics);

    let token = CancellationToken::new();
    let cli_cancel = token.clone();

    // Spawn state client task (handles expert loading/unloading)
    let cli = tokio::task::spawn(async move {
        let worker_id = x::get_worker_id();
        log::info!("ek hostname: {worker_id:}");
        let control_endpoint = x::get_controller_addr();
        log::info!("control endpoint {:}", control_endpoint.uri());
        let mut state_client = StateClient::new(control_endpoint, &worker_id);
        if let Err(e) = state_client.run(cli_cancel).await {
            log::error!("state client error {e:}");
        }
    });

    // Spawn state inspector task (monitors loading progress)
    let state_inspect = StateInspector::spawn();

    let async_srv;
    let mut sync_srvs = Vec::new();
    let poison = Arc::new(Mutex::new(false));
    tch::set_num_threads(1);

    if settings.worker.channel == "grpc" {
        // Spawn gRPC server task (handles computation requests)
        let srv = tokio::task::spawn(async move {
            let server = BasicExpertImpl::new(); // Uses both sync and async gates
            let settings = &get_ek_settings().worker;
            let addr = format!("{}:{}", settings.listen, settings.ports.main)
                .parse()
                .unwrap();
            log::info!("worker server listening on {addr}");

            // Set up gRPC server with OpenTelemetry middleware
            let layer = tower::ServiceBuilder::new()
                .layer_fn(OTelGrpcServerMiddleware::new)
                .into_inner();

            let err = tonic::transport::Server::builder()
                .layer(layer)
                .add_service(
                    ComputationServiceServer::new(server)
                        .max_decoding_message_size(200 * 1024 * 1024)
                        .max_encoding_message_size(200 * 1024 * 1024),
                )
                .serve(addr)
                .await;
            if let Err(e) = err {
                log::error!("server error {e:?}");
            }
        });
        async_srv = srv;
    } else {
        let node_name = x::get_worker_id();
        let recv_channel = loop {
            if let Some(channel) =
                ShmQueue::<LocalShmWorkerReq>::open(&format!("ek-shmq-req-{}", node_name))
            {
                break Arc::new(Mutex::new(channel));
            }
        };
        let send_channel = loop {
            if let Some(channel) =
                ShmQueue::<LocalShmWorkerResp>::open(&format!("ek-shmq-resp-{}", node_name))
            {
                break Arc::new(Mutex::new(channel));
            }
        };
        let thread_count: usize = env::var("EK_WORKER_THREADS")
            .map(|v| v.parse().unwrap_or(1))
            .unwrap_or(1);

        for _ in 0..thread_count {
            let recv_channel = recv_channel.clone();
            let send_channel = send_channel.clone();
            let gate = EKInstanceGateSync::default();
            let poison = poison.clone();
            let srv = std::thread::spawn(move || {
                'main: loop {
                    let req = loop {
                        if *poison.lock().unwrap() {
                            break 'main;
                        }
                        if let Ok(req) = recv_channel.lock().unwrap().recv() {
                            break req;
                        }
                        std::thread::sleep(Duration::from_micros(100));
                    };
                    log::debug!(
                        "received request: id={} expert={}",
                        req.id(),
                        req.expert_id()
                    );
                    let now = time::Instant::now();
                    let expert_id = req.expert_id();
                    let input_tensor = req.input_tensor();
                    let output_tensor = loop {
                        match gate.forward_sync_core(&expert_id, input_tensor) {
                            Ok(result) => {
                                log::debug!("forward_sync_core completed for expert={}", expert_id);
                                break result;
                            }
                            Err(err) => log::warn!("forward_sync_core {err}, retrying..."),
                        }
                        std::thread::sleep(Duration::from_secs(1));
                    };
                    let resp = LocalShmWorkerResp::new(req.id(), output_tensor);
                    while send_channel.lock().unwrap().send(&resp).is_err() {
                        log::warn!("send_channel full, retrying...");
                        std::thread::sleep(Duration::from_micros(100));
                    }
                    log::info!(
                        "request id={} expert={} processed in {}us",
                        req.id(),
                        req.expert_id(),
                        now.elapsed().as_micros(),
                    );
                }
            });
            sync_srvs.push(srv);
        }
        let token = token.clone();
        async_srv = tokio::spawn(async move {
            select! {
                _ = token.cancelled() => {
                    log::info!("async service cancelled");
                }
            }
        });
    }

    // Wait for any task to complete or receive shutdown signal
    select! {
        _ = cli => { },
        _ = async_srv => { },
        _ = state_inspect => { },
        _ = signal::ctrl_c() => {
            log::info!("ctrl-c signal received, shutting down");
            *poison.lock().unwrap() = true;
            token.clone().cancel();
            let(_,rx) = get_graceful_shutdown_ch();
            rx.lock().await.recv().await;
            for srv in sync_srvs {
                srv.join().unwrap();
            }
            log::info!("graceful shutdown channel received, shutting down now");
        }
    };

    Ok(())
}
