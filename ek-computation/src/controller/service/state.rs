use std::time::Duration;

use crate::{
    controller::{
        dispatcher::{DISPATCHER, Dispatcher},
        registry::get_registry,
    },
    proto::ek::{
        object::v1::ExpertSlice,
        worker::v1::{self, ExchangeResp, RdmaEndpoint, RdmaEndpointPair},
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
    async fn listen_worker_ping(
        mut req: tonic::Request<Streaming<v1::ExchangeReq>>,
        hostname: String,
    ) {
        let w = StateWriterImpl {};
        loop {
            match timeout(Duration::from_secs(60), req.get_mut().message()).await {
                Ok(Ok(Some(msg))) => {
                    // Get existing config from database to preserve controller endpoints
                    let reader = StateReaderImpl::new();
                    let mut config = match reader.node_by_hostname(&msg.id).await {
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

                    if let Some(rdma_endpoints) = &msg.rdma_endpoints {
                        if let Some(worker_req_endpoint) = &rdma_endpoints.request_endpoint {
                            config["worker_rdma_request_endpoint"] = serde_json::json!({
                                "qp_endpoint": worker_req_endpoint.qp_endpoint,
                                "memory_region": worker_req_endpoint.memory_region,
                            });
                        }

                        if let Some(worker_resp_endpoint) = &rdma_endpoints.response_endpoint {
                            config["worker_rdma_response_endpoint"] = serde_json::json!({
                                "qp_endpoint": worker_resp_endpoint.qp_endpoint,
                                "memory_region": worker_resp_endpoint.memory_region,
                            });
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
                    continue;
                }
                Ok(Ok(None)) => {
                    log::warn!("worker ping stream closed for worker_id={hostname}");
                    let _ = w.deactivate_node(&hostname).await;
                    get_registry().lock().await.deregister(&hostname).await;
                    return;
                }
                Ok(Err(e)) => {
                    log::error!("worker ping stream error for worker_id={hostname}, {e}");
                    let _ = w.deactivate_node(&hostname).await;
                    get_registry().lock().await.deregister(&hostname).await;
                    return;
                }
                Err(e) => {
                    log::error!("worker ping stream timeout for worker_id={hostname}, {e}");
                    let _ = w.deactivate_node(&hostname).await;
                    get_registry().lock().await.deregister(&hostname).await;
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

        // If channel is rdma, clear deprecated RDMA endpoints from database
        if first_message.channel == "rdma" {
            let w = StateWriterImpl {};
            if let Err(e) = w.clear_node_config_by_hostname(&worker_id).await {
                log::error!(
                    "Failed to clear deprecated RDMA endpoints for worker {}: {e}",
                    worker_id
                );
            } else {
                log::info!("Cleared deprecated RDMA endpoints for worker {}", worker_id);
            }
        }

        // Handle incoming worker requests: Ping
        let worker_id_for_ping = worker_id.clone();
        tokio::spawn(async move {
            // Upsert worker node and update last seen time in database
            StateServerImpl::listen_worker_ping(request, worker_id_for_ping.clone()).await;
        });

        // Watcher experts updates for the worker
        let mut rx = dispather_guard.subscribe(&first_message.id).await;

        // Handle outgoing messages to the worker: New Experts
        let worker_id_for_spawn = worker_id.clone();
        tokio::spawn(async move {
            let reader = StateReaderImpl::new();

            while let Some(t) = rx.recv().await {
                // Check if the worker is using RDMA backend and fetch controller endpoints
                let rdma_endpoints = if let Ok(Some(worker_node)) =
                    reader.node_by_hostname(&worker_id_for_spawn).await
                {
                    if worker_node.config.get("channel").and_then(|v| v.as_str()) == Some("rdma") {
                        // Extract controller endpoints from worker's database record
                        let req_endpoint = worker_node
                            .config
                            .get("controller_rdma_request_endpoint")
                            .and_then(|data| {
                                Some(RdmaEndpoint {
                                    qp_endpoint: data.get("qp_endpoint")?.as_str()?.to_string(),
                                    memory_region: data.get("memory_region")?.as_str()?.to_string(),
                                })
                            });

                        let resp_endpoint = worker_node
                            .config
                            .get("controller_rdma_response_endpoint")
                            .and_then(|data| {
                                Some(RdmaEndpoint {
                                    qp_endpoint: data.get("qp_endpoint")?.as_str()?.to_string(),
                                    memory_region: data.get("memory_region")?.as_str()?.to_string(),
                                })
                            });

                        if req_endpoint.is_some() || resp_endpoint.is_some() {
                            Some(RdmaEndpointPair {
                                request_endpoint: req_endpoint,
                                response_endpoint: resp_endpoint,
                            })
                        } else {
                            log::debug!(
                                "No controller RDMA endpoints found for worker {}",
                                worker_id_for_spawn
                            );
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    log::warn!("Worker node {} not found in database", worker_id_for_spawn);
                    None
                };

                let resp = ExchangeResp {
                    state: Some(v1::exchange_resp::ExpertWithState {
                        target: Some(ExpertSlice::from(t)),
                    }),
                    rdma_endpoints: rdma_endpoints.clone(),
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
