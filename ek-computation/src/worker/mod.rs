mod core;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    {Arc, LazyLock, Mutex, OnceLock},
};
use std::time;
use std::time::Duration;
use std::{env, panic};

use ek_base::tracing::grpc::OTelGrpcServerMiddleware;
use ek_db::weight_manager::{LocalWeightManager, peer_server};
use state::StateInspector;
use tokio::select;
use tokio::signal;
use tokio_util::sync::CancellationToken;
mod manager;
pub mod server;
pub mod state;
pub mod x;

use crate::controller::registry::{ShmqWorkerReq, ShmqWorkerResp};
use crate::metrics::spawn_metrics_server;
use crate::proto::ek::worker::v1::computation_service_server::ComputationServiceServer;
use crate::shmq::{RdmaEndpointServer, ShmQueue, rdma_impl::RdmaQueue};
use crate::worker::core::EKInstanceGateSync;
use crate::worker::server::BasicExpertImpl;
use crate::x::get_graceful_shutdown_ch;

use super::worker::state::StateClient;
use ek_base::{config::get_ek_settings, error::EKResult};

/// Set to true on SIGTERM; the heartbeat stream reads this and sets last_will=true
/// in all subsequent heartbeats so the controller can start proactive migration.
static LAST_WILL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn get_last_will() -> bool {
    LAST_WILL.load(Ordering::Relaxed)
}

// Global storage for RDMA queues and TCP server
static RDMA_REQ_QUEUE: OnceLock<Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>> = OnceLock::new();
static RDMA_RESP_QUEUE: OnceLock<Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>> = OnceLock::new();
static RDMA_CONNECTION_STATUS: AtomicBool = AtomicBool::new(false);
static RDMA_TCP_PORT: OnceLock<u16> = OnceLock::new();
static WORKER_PARALLEL: LazyLock<usize> = LazyLock::new(|| {
    env::var("EK_WORKER_PARALLEL")
        .map(|v| v.parse().unwrap_or(1))
        .unwrap_or(1)
});

/// Get the global RDMA request queue
pub fn get_rdma_req_queue() -> Option<&'static Arc<Mutex<RdmaQueue<ShmqWorkerReq>>>> {
    RDMA_REQ_QUEUE.get()
}

/// Get the global RDMA response queue
pub fn get_rdma_resp_queue() -> Option<&'static Arc<Mutex<RdmaQueue<ShmqWorkerResp>>>> {
    RDMA_RESP_QUEUE.get()
}

/// Get the global RDMA connection status
pub fn is_rdma_queue_connected() -> bool {
    RDMA_CONNECTION_STATUS.load(Ordering::Relaxed)
}

pub fn update_rdma_connection_status(connected: bool) {
    RDMA_CONNECTION_STATUS.store(connected, Ordering::Relaxed);
}

pub fn close_rdma_queues() {
    if let Some(req_queue) = RDMA_REQ_QUEUE.get() {
        let mut rq = req_queue.lock().unwrap();
        rq.disconnect();
    }
    if let Some(resp_queue) = RDMA_RESP_QUEUE.get() {
        let mut rq = resp_queue.lock().unwrap();
        rq.disconnect();
    }
    update_rdma_connection_status(false);
}

/// Get the global RDMA TCP port
pub fn get_rdma_tcp_port() -> Option<u16> {
    RDMA_TCP_PORT.get().copied()
}

/// Create RDMA queues and start TCP server for endpoint exchange
async fn create_rdma_queues_with_tcp_server(
    poison: Arc<Mutex<bool>>,
) -> EKResult<(u16, std::thread::JoinHandle<()>)> {
    // Worker receives requests (sender=false) and sends responses (sender=true)
    let req_queue = RdmaQueue::<ShmqWorkerReq>::new(None, 256, false)?;
    let resp_queue = RdmaQueue::<ShmqWorkerResp>::new(None, 256, true)?;

    let req_queue_arc = Arc::new(Mutex::new(req_queue));
    let resp_queue_arc = Arc::new(Mutex::new(resp_queue));

    // Store the queues globally
    RDMA_REQ_QUEUE.set(req_queue_arc.clone()).map_err(|_| {
        ek_base::error::EKError::InvalidInput("Failed to set RDMA request queue".into())
    })?;
    RDMA_RESP_QUEUE.set(resp_queue_arc.clone()).map_err(|_| {
        ek_base::error::EKError::InvalidInput("Failed to set RDMA response queue".into())
    })?;

    // Create TCP server for endpoint exchange
    let endpoint_server = RdmaEndpointServer::new(req_queue_arc, resp_queue_arc, poison)
        .map_err(ek_base::error::EKError::IoError)?;
    let tcp_port = endpoint_server.port();

    // Store TCP port globally
    RDMA_TCP_PORT
        .set(tcp_port)
        .map_err(|_| ek_base::error::EKError::InvalidInput("Failed to set RDMA TCP port".into()))?;

    // Start the TCP server and return its handle
    let endpoint_server_handle = std::thread::spawn(move || match endpoint_server.start() {
        Ok(()) => {
            log::info!("RDMA TCP endpoint server completed successfully");
        }
        Err(e) => {
            log::error!("RDMA TCP endpoint server failed: {}", e);
        }
    });

    Ok((tcp_port, endpoint_server_handle))
}

/// Main worker entry point
pub async fn worker_main() -> EKResult<()> {
    let settings = get_ek_settings();

    spawn_metrics_server(&settings.worker.metrics);

    let token = CancellationToken::new();
    let cli_cancel = token.clone();

    // Create poison flag for graceful shutdown
    let poison = Arc::new(Mutex::new(false));

    // Spawn state inspector task (monitors loading progress)
    let state_inspect = StateInspector::spawn();

    let async_srv;
    let mut sync_srvs = Vec::new();
    tch::set_num_threads(*WORKER_PARALLEL as _);

    // Initialize LocalWeightManager (shared between WeightService and StateClient)
    let wm = LocalWeightManager::new_shared();
    log::info!(
        "LocalWeightManager initialized (mem_cache={}MB)",
        settings.weight.mem_cache_mb
    );

    // Start peer weight HTTP server
    {
        let wm_listen = settings.weight.wm_listen.clone();
        let wm_clone = wm.clone();
        let wm_addr: std::net::SocketAddr =
            wm_listen.parse().expect("invalid weight.wm_listen address");
        tokio::spawn(async move {
            log::info!("Peer weight HTTP server listening on {wm_addr}");
            match peer_server::start_peer_server(wm_clone, &wm_addr).await {
                Ok(server) => {
                    if let Err(e) = server.await {
                        log::error!("Peer weight server error: {e}");
                    }
                }
                Err(e) => log::error!("Failed to start peer weight server: {e}"),
            }
        });
    }

    // Determine queue type based on configuration
    // Note: Channel should be created before stateClient start for endpoint exchange
    let rdma_tcp_port: Option<u16> = if settings.worker.channel == "rdma" {
        match create_rdma_queues_with_tcp_server(poison.clone()).await {
            Ok((tcp_port, handle)) => {
                log::info!("RDMA queues and TCP server created successfully");
                sync_srvs.push(handle);
                Some(tcp_port)
            }
            Err(e) => {
                log::error!("Failed to create RDMA queues: {e}");
                return Err(e);
            }
        }
    } else {
        None
    };

    // Channel for controller to notify that proactive migration is complete.
    let (preemption_tx, preemption_rx) = tokio::sync::oneshot::channel::<()>();

    // Spawn state client task (handles expert loading/unloading)
    let cli = tokio::task::spawn(async move {
        let worker_id = x::get_worker_id();
        log::info!("ek hostname: {worker_id:}");
        let control_endpoint = x::get_controller_addr();
        log::info!("control endpoint {:}", control_endpoint.uri());
        let mut state_client =
            StateClient::new_with_rdma_tcp_port(control_endpoint, &worker_id, rdma_tcp_port, wm);
        state_client.set_preemption_notifier(preemption_tx);
        if let Err(e) = state_client.run(cli_cancel).await {
            log::error!("state client error {e:}");
        }
    });

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

            log::info!("Creating shared memory queues for worker {}", node_name);

            let recv_channel = ShmQueue::<ShmqWorkerReq>::new(
                &format!("ek-shmq-req-{}", node_name),
                256, // Queue capacity
            );
            log::info!("Created request queue: /dev/shm/ek-shmq-req-{}", node_name);

            let send_channel = ShmQueue::<ShmqWorkerResp>::new(
                &format!("ek-shmq-resp-{}", node_name),
                256, // Queue capacity
            );
            log::info!(
                "Created response queue: /dev/shm/ek-shmq-resp-{}",
                node_name
            );

            let recv_channel = Arc::new(Mutex::new(recv_channel));
            let send_channel = Arc::new(Mutex::new(send_channel));
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
                            std::hint::spin_loop();
                            std::thread::yield_now();
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
                        let resp = ShmqWorkerResp::new(req.id(), output_tensor);
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
                                    std::hint::spin_loop();
                                    std::thread::yield_now();
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
                        let resp = ShmqWorkerResp::new(req.id(), output_tensor);
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

    // SIGTERM listener: use a oneshot channel so the select! arms are platform-independent.
    // On unix the spawned task forwards SIGTERM; on other platforms the sender is held
    // forever so sigterm_rx never fires.
    let (sigterm_tx, mut sigterm_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        #[cfg(unix)]
        {
            if let Ok(mut sig) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            {
                sig.recv().await;
                let _ = sigterm_tx.send(());
                return;
            }
        }
        // Non-unix or signal setup failure: hold sender indefinitely so sigterm_rx
        // never fires.
        let _keep_alive = sigterm_tx;
        std::future::pending::<()>().await
    });

    // Wait for any task to complete or receive a shutdown signal.
    // sync_srvs is joined AFTER the select so both signal arms can reference it.
    let shutdown_signal: Option<&str> = select! {
        _ = cli => None,
        _ = async_srv => None,
        _ = state_inspect => None,
        _ = signal::ctrl_c() => Some("ctrl-c"),
        _ = &mut sigterm_rx => Some("SIGTERM"),
    };

    // Graceful preemption for ctrl-c / SIGTERM
    if let Some(signal) = shutdown_signal {
        log::info!(
            "{signal} received, starting graceful preemption (grace={}s)",
            settings.worker.shutdown_grace_secs
        );
        // 1. Signal last_will — next heartbeat(s) will carry last_will=true
        LAST_WILL.store(true, Ordering::Relaxed);
        // 2. Poison compute gate — stop accepting new expert requests immediately
        *poison.lock().unwrap() = true;
        // 3. Wait for controller to confirm migration complete, or timeout
        match tokio::time::timeout(
            Duration::from_secs(settings.worker.shutdown_grace_secs),
            preemption_rx,
        )
        .await
        {
            Ok(Ok(())) => {
                log::info!("Controller confirmed preemption complete");
            }
            Ok(Err(_)) => {
                log::warn!("Preemption channel dropped, proceeding with shutdown");
            }
            Err(_) => {
                log::warn!(
                    "Preemption timeout after {}s, forcing shutdown",
                    settings.worker.shutdown_grace_secs
                );
            }
        }
        // 4. Cancel async tasks (heartbeat, state client, etc.)
        token.cancel();
        let (_, rx) = get_graceful_shutdown_ch();
        rx.lock().await.recv().await;
        log::info!("Graceful preemption complete");
    }

    for srv in sync_srvs {
        srv.join().unwrap();
    }

    Ok(())
}
