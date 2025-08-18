mod core;

use std::env;
use std::sync::{Arc, Mutex, OnceLock};
use std::time;
use std::time::Duration;

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
use crate::proto::ek::worker::v1::RdmaEndpoint;
use ek_base::{
    config::{ExpertRegistryBackend, get_ek_settings},
    error::EKResult,
};

// Global storage for RDMA queues
static RDMA_REQ_QUEUE: OnceLock<Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>> = OnceLock::new();
static RDMA_RESP_QUEUE: OnceLock<Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>> = OnceLock::new();

/// Get the global RDMA request queue
pub fn get_rdma_req_queue() -> Option<&'static Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>> {
    RDMA_REQ_QUEUE.get()
}

/// Get the global RDMA response queue
pub fn get_rdma_resp_queue() -> Option<&'static Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>> {
    RDMA_RESP_QUEUE.get()
}

/// Create RDMA queues and return endpoint information
async fn create_rdma_queues() -> EKResult<RdmaEndpoint> {
    let req_queue = RdmaQueue::<RdmaWorkerReq>::new(None, 128, true)?;
    let resp_queue = RdmaQueue::<RdmaWorkerResp>::new(None, 128, false)?;

    let req_endpoint = req_queue.endpoint()?;
    let req_memory = req_queue.memory_region();
    let _resp_endpoint = resp_queue.endpoint()?;
    let _resp_memory = resp_queue.memory_region();

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
    let qp_endpoint_json = serde_json::to_string(&req_endpoint).map_err(|e| {
        ek_base::error::EKError::InvalidInput(format!("Failed to serialize QP endpoint: {e}"))
    })?;

    let memory_region_json = serde_json::to_string(&req_memory).map_err(|e| {
        ek_base::error::EKError::InvalidInput(format!("Failed to serialize memory region: {e}"))
    })?;

    Ok(RdmaEndpoint {
        qp_endpoint: qp_endpoint_json.into_bytes(),
        memory_region: memory_region_json.into_bytes(),
    })
}

/// Main worker entry point
pub async fn worker_main() -> EKResult<()> {
    let settings = get_ek_settings();

    spawn_metrics_server(&settings.worker.metrics);

    let token = CancellationToken::new();
    let cli_cancel = token.clone();

    // Determine queue type based on configuration
    let use_rdma = matches!(
        settings.controller.registry_backend,
        ExpertRegistryBackend::Rdma
    );
    let rdma_endpoint = if use_rdma {
        // Create RDMA queues and get endpoint info
        match create_rdma_queues().await {
            Ok(endpoint) => {
                log::info!("RDMA queues created successfully");
                Some(endpoint)
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
            StateClient::new_with_rdma(control_endpoint, &worker_id, rdma_endpoint);
        if let Err(e) = state_client.run(cli_cancel).await {
            log::error!("state client error {e:}");
        }
    });

    let node_name = x::get_worker_id();

    // Choose queue type based on configuration
    enum QueueType {
        Shm {
            recv_channel: Arc<Mutex<ShmQueue<'static, LocalShmWorkerReq>>>,
            send_channel: Arc<Mutex<ShmQueue<'static, LocalShmWorkerResp>>>,
        },
        Rdma {
            recv_channel: Arc<Mutex<RdmaQueue<RdmaWorkerReq>>>,
            send_channel: Arc<Mutex<RdmaQueue<RdmaWorkerResp>>>,
        },
    }

    let queue_type = if use_rdma {
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
        QueueType::Rdma {
            recv_channel,
            send_channel,
        }
    } else {
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
        QueueType::Shm {
            recv_channel,
            send_channel,
        }
    };
    let mut srvs = Vec::new();
    let thread_count: usize = env::var("EK_WORKER_THREADS")
        .map(|v| v.parse().unwrap_or(1))
        .unwrap_or(1);

    let poison = Arc::new(Mutex::new(false));

    for _ in 0..thread_count {
        let queue_type = match &queue_type {
            QueueType::Shm {
                recv_channel,
                send_channel,
            } => QueueType::Shm {
                recv_channel: recv_channel.clone(),
                send_channel: send_channel.clone(),
            },
            QueueType::Rdma {
                recv_channel,
                send_channel,
            } => QueueType::Rdma {
                recv_channel: recv_channel.clone(),
                send_channel: send_channel.clone(),
            },
        };
        let gate = EKInstanceGateSync::default();
        let poison = poison.clone();
        let srv = std::thread::spawn(move || {
            'main: loop {
                if *poison.lock().unwrap() {
                    break 'main;
                }

                match &queue_type {
                    QueueType::Shm {
                        recv_channel,
                        send_channel,
                    } => {
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
                            "received SHM request: id={} expert={}",
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
                            "SHM request id={} expert={} processed in {}us",
                            req.id(),
                            req.expert_id(),
                            now.elapsed().as_micros(),
                        );
                    }
                    QueueType::Rdma {
                        recv_channel,
                        send_channel,
                    } => {
                        let req = loop {
                            if *poison.lock().unwrap() {
                                break 'main;
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
                            "received RDMA request: id={} expert={}",
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
                        while send_channel.lock().unwrap().send(&resp).is_err() {
                            log::warn!("RDMA send_channel full, retrying...");
                            std::thread::sleep(Duration::from_micros(100));
                        }
                        log::info!(
                            "RDMA request id={} expert={} processed in {}us",
                            req.id(),
                            req.expert_id(),
                            now.elapsed().as_micros(),
                        );
                    }
                }
            }
        });
        srvs.push(srv);
    }

    // Spawn state inspector task (monitors loading progress)
    let state_inspect = StateInspector::spawn();

    // Wait for any task to complete or receive shutdown signal
    select! {
        _ = cli => { },
        // _ = srv => { },
        _ = state_inspect => { },
        _ = signal::ctrl_c() => {
            log::info!("ctrl-c signal received, shutting down");
            *poison.lock().unwrap() = true;
            token.clone().cancel();
            let(_,rx) = get_graceful_shutdown_ch();
            rx.lock().await.recv().await;
            for srv in srvs {
                srv.join().unwrap();
            }
            log::info!("graceful shutdown channel received, shutting down now");
        }
    };

    Ok(())
}
