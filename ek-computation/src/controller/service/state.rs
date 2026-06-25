use std::time::Duration;

use ek_base::config::get_ek_settings;

use crate::{
    controller::{
        dispatcher::{DISPATCHER, Dispatcher},
        elastic::{frequency::get_freq_tracker, progressive, recovery::recover_unique_experts},
        poller::request_immediate_poll,
        registry::get_registry,
        routing_broadcaster::get_broadcaster,
    },
    proto::ek::{
        object::v1::ExpertSlice,
        worker::v1::{self, ExchangeResp},
    },
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewNode,
        writer::StateWriterImpl,
    },
};
use tokio::{sync::mpsc, time::timeout};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Response, Result, Status, Streaming};

use crate::proto::ek::worker::v1::state_service_server::StateService;
pub struct StateServerImpl {}

impl StateServerImpl {
    /// Process a single heartbeat message.  Returns `true` if the stream should
    /// continue, `false` if the worker signalled graceful shutdown and the caller
    /// should stop.
    async fn process_heartbeat(
        msg: &v1::ExchangeReq,
        hostname: &str,
        w: &StateWriterImpl,
        graceful_shutdown_handled: &mut bool,
        progressive_triggered: &mut bool,
        stream_tx: &mpsc::Sender<Result<ExchangeResp, Status>>,
    ) {
        // Get existing config from database to preserve controller endpoints
        let reader = StateReaderImpl::new();
        let existing = reader.node_by_hostname(&msg.id).await;

        // Treat a node as "new" (needing progressive_assign) whenever it is
        // not currently active.  This covers brand-new nodes, returning nodes
        // after crash/shutdown, and nodes whose capacity has changed.
        // progressive_assign cleans stale expert assignments before reassigning.
        let is_new = match &existing {
            Ok(None) => true,
            Ok(Some(_)) => {
                // Node exists in DB — check if it's in the active set.
                // If not active, treat as returning (needs fresh assignment).
                let active = reader.active_nodes().await.unwrap_or_default();
                !active.iter().any(|n| n.hostname == msg.id)
            }
            Err(_) => false,
        };

        let mut config = match existing {
            Ok(Some(existing_node)) => {
                log::debug!("Preserving existing config for worker {}", msg.id);
                existing_node.config
            }
            Ok(None) => {
                log::debug!("Creating new config for worker {}", msg.id);
                serde_json::json!({})
            }
            Err(_) => {
                log::warn!(
                    "Failed to read existing config for worker {}, creating new",
                    msg.id
                );
                serde_json::json!({})
            }
        };

        // Update worker-specific fields
        config["addr"] = serde_json::json!(msg.addr.clone());
        config["channel"] = serde_json::json!(msg.channel.clone());

        // Store RDMA TCP port if provided
        if msg.rdma_tcp_port > 0 {
            config["rdma_tcp_port"] = serde_json::json!(msg.rdma_tcp_port);
        }

        // Store WM address if provided
        if !msg.wm_addr.is_empty() {
            config["wm_addr"] = serde_json::json!(msg.wm_addr.clone());
        }

        // Store memory capacity (reported once at startup; update if non-zero)
        if msg.mem_capacity_mb > 0 {
            config["mem_capacity_mb"] = serde_json::json!(msg.mem_capacity_mb);
        }

        // For returning workers: clean stale expert rows BEFORE the node
        // becomes active (node_update_seen).  Otherwise the poller/registry
        // sees the node as active with old "loaded" experts and routes
        // requests to it before it has loaded anything.
        if is_new && !*progressive_triggered {
            // Allow the registry to create channels to this node again
            get_registry().lock().await.reregister(&msg.id);

            if let Ok(Some(node)) = reader.node_by_hostname(&msg.id).await
                && let Ok(old) = reader.experts_by_node(node.id).await
                    && !old.is_empty() {
                        log::info!(
                            "Clearing {} stale expert rows for returning worker {}",
                            old.len(),
                            msg.id
                        );
                        let _ = w.delete_experts_by_node(node.id).await;
                    }
        }

        let err = w
            .node_upsert(NewNode {
                hostname: msg.id.clone(),
                device: msg.device.clone(),
                config,
            })
            .await;

        if let Err(e) = err {
            log::error!("worker ping error, can not upsert node: {e}");
        }

        let e = w.node_update_seen(&msg.id).await;
        if let Err(e) = e {
            log::error!("worker ping error: {e}");
        }

        // Process loaded experts for progressive startup support
        if !msg.loaded_experts.is_empty() {
            log::debug!(
                "Worker {} reports {} loaded experts",
                msg.id,
                msg.loaded_experts.len()
            );
        }
        if let Err(e) = w.update_expert_load_states(&msg.id, &msg.loaded_experts).await {
            log::error!("Failed to update expert load states for {}: {}", msg.id, e);
        }

        // Accumulate per-expert request counts for frequency-based decisions
        if !msg.expert_request_counts.is_empty() {
            get_freq_tracker().add_counts(msg.expert_request_counts.clone());
        }

        // Trigger progressive assignment for new/returning workers
        if is_new && !*progressive_triggered {
            *progressive_triggered = true;
            let hn = msg.id.clone();
            tokio::spawn(async move {
                progressive::progressive_assign(&hn).await;
            });
        }

        // Proactive migration: worker signals graceful preemption via last_will.
        //
        // Key invariant: do NOT remove the dying worker from routing yet.  Its
        // gRPC server is still alive during the grace period, so it can keep
        // serving requests while we copy unique experts to other workers.  The
        // node is removed from routing only when the heartbeat stream closes
        // (grace period expired / worker exited).
        if msg.last_will && !*graceful_shutdown_handled {
            *graceful_shutdown_handled = true;
            log::info!(
                "Worker {} signaled graceful shutdown, starting proactive migration \
                 (keeping in routing until stream closes)",
                hostname
            );
            let hn = hostname.to_owned();
            let tx = stream_tx.clone();
            tokio::spawn(async move {
                recover_unique_experts(&hn).await;
                // Notify the worker that migration is complete and it may shut down.
                let resp = ExchangeResp {
                    state: None,
                    preemption_complete: true,
                };
                if let Err(e) = tx.send(Ok(resp)).await {
                    log::warn!("Failed to send preemption_complete to {hn}: {e}");
                }
            });
        }
    }

    async fn listen_worker_ping(
        mut req: tonic::Request<Streaming<v1::ExchangeReq>>,
        hostname: String,
        first_message: Option<v1::ExchangeReq>,
        stream_tx: mpsc::Sender<Result<ExchangeResp, Status>>,
    ) {
        let w = StateWriterImpl {};
        let mut graceful_shutdown_handled = false;
        let mut progressive_triggered = false;

        // Process the first message that was already consumed in exchange()
        if let Some(msg) = &first_message {
            Self::process_heartbeat(
                msg, &hostname, &w,
                &mut graceful_shutdown_handled, &mut progressive_triggered,
                &stream_tx,
            ).await;
        }

        loop {
            match timeout(Duration::from_secs(get_ek_settings().controller.fault_detection.heartbeat_timeout_secs), req.get_mut().message()).await {
                Ok(Ok(Some(msg))) => {
                    Self::process_heartbeat(
                        &msg, &hostname, &w,
                        &mut graceful_shutdown_handled, &mut progressive_triggered,
                        &stream_tx,
                    ).await;
                    continue;
                }
                Ok(Ok(None)) => {
                    log::warn!("worker ping stream closed for worker_id={hostname}");
                    let _ = w.deactivate_node(&hostname).await;
                    get_registry().lock().await.deregister(&hostname).await;
                    if graceful_shutdown_handled {
                        log::info!("Graceful shutdown complete for {hostname}, letting poller update routing");
                        // Recovery already completed during grace — safe to
                        // clean stale expert rows immediately.
                        let _ = w.delete_experts_by_node_hostname(&hostname).await;
                    } else {
                        get_broadcaster().remove_node(&hostname).await;
                        let hn = hostname.clone();
                        // Recovery runs first (reads expert rows), then cleans up.
                        tokio::spawn(async move {
                            recover_unique_experts(&hn).await;
                            let w = StateWriterImpl::new();
                            let _ = w.delete_experts_by_node_hostname(&hn).await;
                        });
                    }
                    request_immediate_poll();
                    return;
                }
                Ok(Err(e)) => {
                    log::error!("worker ping stream error for worker_id={hostname}, {e}");
                    let _ = w.deactivate_node(&hostname).await;
                    get_registry().lock().await.deregister(&hostname).await;
                    if graceful_shutdown_handled {
                        log::info!("Graceful shutdown complete for {hostname}, letting poller update routing");
                        let _ = w.delete_experts_by_node_hostname(&hostname).await;
                    } else {
                        get_broadcaster().remove_node(&hostname).await;
                        let hn = hostname.clone();
                        tokio::spawn(async move {
                            recover_unique_experts(&hn).await;
                            let w = StateWriterImpl::new();
                            let _ = w.delete_experts_by_node_hostname(&hn).await;
                        });
                    }
                    request_immediate_poll();
                    return;
                }
                Err(e) => {
                    log::error!("worker ping stream timeout for worker_id={hostname}, {e}");
                    let _ = w.deactivate_node(&hostname).await;
                    get_registry().lock().await.deregister(&hostname).await;
                    if graceful_shutdown_handled {
                        log::info!("Graceful shutdown complete for {hostname}, letting poller update routing");
                        let _ = w.delete_experts_by_node_hostname(&hostname).await;
                    } else {
                        get_broadcaster().remove_node(&hostname).await;
                        let hn = hostname.clone();
                        tokio::spawn(async move {
                            recover_unique_experts(&hn).await;
                            let w = StateWriterImpl::new();
                            let _ = w.delete_experts_by_node_hostname(&hn).await;
                        });
                    }
                    request_immediate_poll();
                    return;
                }
            }
        }
    }
}

#[tonic::async_trait]
impl StateService for StateServerImpl {
    // type RetrieveStream = Pin<Box<dyn Stream<Item = Result<RetrieveStateResp>> + Send + 'static>>;
    type ExchangeStream = ReceiverStream<Result<ExchangeResp, Status>>;

    async fn exchange(
        &self,
        mut request: tonic::Request<Streaming<v1::ExchangeReq>>,
    ) -> Result<Response<Self::ExchangeStream>> {
        let mut dispather_guard = DISPATCHER.lock().await;
        let (stream_tx, stream_rx) = mpsc::channel(4);
        let first_message = request
            .get_mut()
            .message()
            .await?
            .ok_or(Status::invalid_argument("no message"))?;
        let worker_id = first_message.id.clone();

        // Log RDMA TCP port if provided
        if first_message.channel == "rdma" && first_message.rdma_tcp_port > 0 {
            log::info!(
                "Worker {} using RDMA with TCP port {}",
                worker_id,
                first_message.rdma_tcp_port
            );
        }

        // Handle incoming worker requests: Ping
        // Pass the first_message so it is processed as a heartbeat too.
        let worker_id_for_ping = worker_id.clone();
        let first_msg_clone = first_message.clone();
        let stream_tx_for_ping = stream_tx.clone();
        tokio::spawn(async move {
            StateServerImpl::listen_worker_ping(request, worker_id_for_ping.clone(), Some(first_msg_clone), stream_tx_for_ping).await;
        });

        // Watcher experts updates for the worker
        let mut rx = dispather_guard.subscribe(&first_message.id).await;

        // Handle outgoing messages to the worker: New Experts
        tokio::spawn(async move {
            while let Some(t) = rx.recv().await {
                let resp = ExchangeResp {
                    state: Some(v1::exchange_resp::ExpertWithState {
                        target: Some(ExpertSlice::from(t)),
                    }),
                    preemption_complete: false,
                };
                if let Err(e) = stream_tx.send(Ok(resp)).await {
                    log::error!("stream error: {e}")
                };
            }
        });

        Ok(Response::new(Self::ExchangeStream::new(stream_rx)))
    }
}

impl Default for StateServerImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl StateServerImpl {
    pub fn new() -> Self {
        Self {}
    }
}
