use std::time::Duration;

use crate::{
    controller::{
        dispatcher::{DISPATCHER, Dispatcher},
        registry::get_registry,
    },
    proto::ek::{
        object::v1::ExpertSlice,
        worker::v1::{self, ExchangeResp, RdmaEndpoint, RdmaEndpoints},
    },
    state::{
        models::NewNode,
        writer::StateWriterImpl,
        io::{StateReader, StateReaderImpl},
    },
};
use ek_base::config::{ExpertRegistryBackend, get_ek_settings};
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
                    let mut config = serde_json::json!({
                        "addr": msg.addr.clone(),
                        "channel": msg.channel.clone(),
                    });

                    if let Some(rdma_endpoints) = &msg.rdma_endpoints {
                        // Extract worker's request endpoint (where controller sends requests)
                        if let Some(worker_req_endpoint) = &rdma_endpoints.request_endpoint {
                            let req_qp_json = String::from_utf8(worker_req_endpoint.qp_endpoint.clone())
                                .unwrap_or_else(|_| "invalid_req_qp_data".to_string());
                            let req_memory_json = String::from_utf8(worker_req_endpoint.memory_region.clone())
                                .unwrap_or_else(|_| "invalid_req_memory_data".to_string());
                            
                            config["worker_rdma_request_endpoint"] = serde_json::json!({
                                "qp_endpoint": req_qp_json,
                                "memory_region": req_memory_json,
                            });
                        }

                        // Extract worker's response endpoint (where worker sends responses)
                        if let Some(worker_resp_endpoint) = &rdma_endpoints.response_endpoint {
                            let resp_qp_json = String::from_utf8(worker_resp_endpoint.qp_endpoint.clone())
                                .unwrap_or_else(|_| "invalid_resp_qp_data".to_string());
                            let resp_memory_json = String::from_utf8(worker_resp_endpoint.memory_region.clone())
                                .unwrap_or_else(|_| "invalid_resp_memory_data".to_string());
                            
                            config["worker_rdma_response_endpoint"] = serde_json::json!({
                                "qp_endpoint": resp_qp_json,
                                "memory_region": resp_memory_json,
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

        // Handle incoming worker requests: Ping
        tokio::spawn(async move {
            // Upsert worker node and update last seen time in database
            StateServerImpl::listen_worker_ping(request, worker_id.clone()).await;
        });

        // Watcher experts updates for the worker
        let mut rx = dispather_guard.subscribe(&first_message.id).await;

        // Handle outgoing messages to the worker: New Experts
        let worker_hostname = first_message.id.clone();
        tokio::spawn(async move {
            let reader = StateReaderImpl::new();
            
            while let Some(t) = rx.recv().await {
                // Check if we're using RDMA backend and should include controller endpoints
                let rdma_endpoints = if matches!(
                    get_ek_settings().controller.registry_backend,
                    ExpertRegistryBackend::Rdma
                ) {
                    // Fetch real controller endpoints from database
                    match reader.node_by_hostname(&worker_hostname).await {
                        Ok(Some(node)) => {
                            // Extract controller request endpoint data
                            let req_endpoint = if let Some(controller_req_data) = node.config.get("controller_rdma_request_endpoint") {
                                if let (Some(qp_json), Some(memory_json)) = (
                                    controller_req_data.get("qp_endpoint").and_then(|v| v.as_str()),
                                    controller_req_data.get("memory_region").and_then(|v| v.as_str())
                                ) {
                                    Some(RdmaEndpoint {
                                        qp_endpoint: qp_json.as_bytes().to_vec(),
                                        memory_region: memory_json.as_bytes().to_vec(),
                                    })
                                } else { None }
                            } else { None };

                            // Extract controller response endpoint data
                            let resp_endpoint = if let Some(controller_resp_data) = node.config.get("controller_rdma_response_endpoint") {
                                if let (Some(qp_json), Some(memory_json)) = (
                                    controller_resp_data.get("qp_endpoint").and_then(|v| v.as_str()),
                                    controller_resp_data.get("memory_region").and_then(|v| v.as_str())
                                ) {
                                    Some(RdmaEndpoint {
                                        qp_endpoint: qp_json.as_bytes().to_vec(),
                                        memory_region: memory_json.as_bytes().to_vec(),
                                    })
                                } else { None }
                            } else { None };

                            if req_endpoint.is_some() || resp_endpoint.is_some() {
                                Some(RdmaEndpoints {
                                    request_endpoint: req_endpoint,
                                    response_endpoint: resp_endpoint,
                                })
                            } else {
                                log::debug!("No controller RDMA endpoints found for worker {}", worker_hostname);
                                None
                            }
                        }
                        Ok(None) => {
                            log::warn!("Worker node {} not found in database", worker_hostname);
                            None
                        }
                        Err(e) => {
                            log::error!("Failed to fetch worker node from database: {e:?}");
                            None
                        }
                    }
                } else {
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
