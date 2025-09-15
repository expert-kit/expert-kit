mod core;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    {Arc, LazyLock, Mutex, OnceLock},
};
use std::time;
use std::time::Duration;
use std::{env, panic};

use ek_base::tracing::grpc::OTelGrpcServerMiddleware;
use state::StateInspector;
use tokio::select;
use tokio::signal;
use tokio_util::sync::CancellationToken;
mod manager;
pub mod server;
pub mod state;
pub mod x;

use crate::controller::registry::{
    LocalShmWorkerReq, LocalShmWorkerResp, RdmaWorkerReq, RdmaWorkerResp,
};
use crate::metrics::spawn_metrics_server;
use crate::shmq::{ShmQueue, rdma_impl::RdmaQueue};
use crate::worker::core::EKInstanceGateSync;
use crate::x::get_graceful_shutdown_ch;

use super::worker::state::StateClient;
use crate::proto::ek::worker::v1::{RdmaEndpoint, RdmaEndpointPair};
use ek_base::{config::get_ek_settings, error::EKResult};

// Global storage for RDMA queues
static RDMA_REQ_QUEUE: OnceLock<Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>> = OnceLock::new();
static RDMA_RESP_QUEUE: OnceLock<Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>> = OnceLock::new();
static RDMA_CONNECTION_STATUS: AtomicBool = AtomicBool::new(false);

/// Get the global RDMA request queue
pub fn get_rdma_req_queue() -> Option<&'static Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>> {
    RDMA_REQ_QUEUE.get()
}

/// Get the global RDMA response queue
pub fn get_rdma_resp_queue() -> Option<&'static Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>> {
    RDMA_RESP_QUEUE.get()
}

/// Get the global RDMA connection status
pub fn is_rdma_queue_connected() -> bool {
    RDMA_CONNECTION_STATUS.load(Ordering::Relaxed)
}

pub fn update_rdma_connection_status(connected: bool) {
    RDMA_CONNECTION_STATUS.store(connected, Ordering::Relaxed);
}

/// Create RDMA queues and return endpoint information
async fn create_rdma_queues() -> EKResult<RdmaEndpointPair> {
    // Worker receives requests (sender=false) and sends responses (sender=true)
    let req_queue = RdmaQueue::<RdmaWorkerReq>::new(None, 256, false)?;
    let resp_queue = RdmaQueue::<RdmaWorkerResp>::new(None, 256, true)?;

    let req_endpoint = req_queue.endpoint()?;
    let req_memory = req_queue.memory_region();
    let resp_endpoint = resp_queue.endpoint()?;
    let resp_memory = resp_queue.memory_region();

    // Store the queues globally
    RDMA_REQ_QUEUE
        .set(Arc::new(Mutex::new(req_queue)))
        .map_err(|_| {
            ek_base::error::EKError::InvalidInput("Failed to set RDMA request queue".into())
        })?;
    RDMA_RESP_QUEUE
        .set(Arc::new(Mutex::new(resp_queue)))
        .map_err(|_| {
            ek_base::error::EKError::InvalidInput("Failed to set RDMA response queue".into())
        })?;

    // Serialize endpoint and memory region data as JSON strings
    let req_endpoint_json = serde_json::to_string(&req_endpoint).map_err(|e| {
        ek_base::error::EKError::InvalidInput(format!("Failed to serialize QP endpoint: {e}"))
    })?;

    let req_memory_json = serde_json::to_string(&req_memory).map_err(|e| {
        ek_base::error::EKError::InvalidInput(format!("Failed to serialize memory region: {e}"))
    })?;

    let resp_endpoint_json = serde_json::to_string(&resp_endpoint).map_err(|e| {
        ek_base::error::EKError::InvalidInput(format!("Failed to serialize QP endpoint: {e}"))
    })?;

    let resp_memory_json = serde_json::to_string(&resp_memory).map_err(|e| {
        ek_base::error::EKError::InvalidInput(format!("Failed to serialize memory region: {e}"))
    })?;

    Ok(RdmaEndpointPair {
        request_endpoint: Some(RdmaEndpoint {
            qp_endpoint: req_endpoint_json,
            memory_region: req_memory_json,
        }),
        response_endpoint: Some(RdmaEndpoint {
            qp_endpoint: resp_endpoint_json,
            memory_region: resp_memory_json,
        }),
    })
}
use crate::proto::ek::worker::v1::computation_service_server::ComputationServiceServer;
use crate::worker::server::BasicExpertImpl;

static WORKER_PARALLEL: LazyLock<usize> = LazyLock::new(|| {
    env::var("EK_WORKER_PARALLEL")
        .map(|v| v.parse().unwrap_or(1))
        .unwrap_or(1)
});

/// Main worker entry point
pub async fn worker_main() -> EKResult<()> {
    let settings = get_ek_settings();

    spawn_metrics_server(&settings.worker.metrics);

    let token = CancellationToken::new();
    let cli_cancel = token.clone();

    // Determine queue type based on configuration
    // Note: Channel should be created before stateClient start for endpoint exchange
    let rdma_endpoints: Option<RdmaEndpointPair> = if settings.worker.channel == "rdma" {
        match create_rdma_queues().await {
            Ok(endpoint_pair) => {
                log::info!("RDMA queues created successfully");
                Some(endpoint_pair)
            }
            Err(e) => {
                log::error!("Failed to create RDMA queues: {e}");
                return Err(e);
            }
        }
    } else {
        None
    };

    // Spawn state client task (handles expert loading/unloading)
    let cli = tokio::task::spawn(async move {
        let worker_id = x::get_worker_id();
        log::info!("ek hostname: {worker_id:}");
        let control_endpoint = x::get_controller_addr();
        log::info!("control endpoint {:}", control_endpoint.uri());
        let mut state_client =
            StateClient::new_with_rdma(control_endpoint, &worker_id, rdma_endpoints);
        if let Err(e) = state_client.run(cli_cancel).await {
            log::error!("state client error {e:}");
        }
    });

    // Spawn state inspector task (monitors loading progress)
    let state_inspect = StateInspector::spawn();

    let async_srv;
    let mut sync_srvs = Vec::new();
    let poison = Arc::new(Mutex::new(false));
    tch::set_num_threads(*WORKER_PARALLEL as _);

    match settings.worker.channel.as_str() {
        "grpc" => {
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
        }
        "shm" => {
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
                                    log::debug!(
                                        "forward_sync_core completed for expert={}",
                                        expert_id
                                    );
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
        "rdma" => {
            let recv_channel = get_rdma_req_queue()
                .ok_or_else(|| {
                    ek_base::error::EKError::NotFound("RDMA request queue not found".into())
                })?
                .clone();
            let send_channel = get_rdma_resp_queue()
                .ok_or_else(|| {
                    ek_base::error::EKError::NotFound("RDMA response queue not found".into())
                })?
                .clone();

            let thread_count: usize = env::var("EK_WORKER_THREADS")
                .map(|v| v.parse().unwrap_or(1))
                .unwrap_or(1);

            for idx in 0..thread_count {
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
                            if !is_rdma_queue_connected() {
                                std::thread::sleep(Duration::from_secs(2));
                            }
                            match recv_channel.lock().unwrap().recv() {
                                Ok(req) => break req,
                                Err(_) => {
                                    std::thread::sleep(Duration::from_micros(100));
                                    continue;
                                }
                            }
                        };

                        log::debug!(
                            "thread {} received RDMA request: id={} expert={}",
                            idx,
                            req.id(),
                            req.expert_id()
                        );
                        let now = time::Instant::now();
                        let expert_id = req.expert_id();
                        let input_tensor = req.input_tensor();
                        let output_tensor = loop {
                            match gate.forward_sync_core(&expert_id, input_tensor) {
                                Ok(result) => {
                                    log::debug!(
                                        "forward_sync_core completed for expert={}",
                                        expert_id
                                    );
                                    break result;
                                }
                                Err(err) => log::warn!("forward_sync_core {err}, retrying..."),
                            }
                            std::thread::sleep(Duration::from_secs(1));
                        };
                        let resp = RdmaWorkerResp::new(req.id(), output_tensor);
                        let send_start = time::Instant::now();
                        while send_channel.lock().unwrap().send(&resp).is_err() {
                            log::warn!("RDMA send_channel full, retrying...");
                            std::thread::sleep(Duration::from_micros(100));
                        }
                        log::info!(
                            "thread {} RDMA request id={} expert={} processed in {}us, send wait {}us",
                            idx,
                            req.id(),
                            req.expert_id(),
                            now.elapsed().as_micros(),
                            send_start.elapsed().as_micros(),
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
        _ => {
            panic!("Unsupported worker channel: {}", settings.worker.channel);
        }
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
