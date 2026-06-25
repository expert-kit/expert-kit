use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Result, Status};

use crate::{
    controller::routing_broadcaster::RoutingBroadcaster,
    proto::ek::control::v1::{
        GetRoutingReq, GetRoutingResp, RoutingUpdate, SubscribeRoutingReq,
        routing_service_server::RoutingService,
    },
};

/// RoutingServiceImpl implements the gRPC RoutingService
/// This service provides routing information to frontends for direct worker communication
pub struct RoutingServiceImpl {
    broadcaster: Arc<RoutingBroadcaster>,
}

impl RoutingServiceImpl {
    pub fn new(broadcaster: Arc<RoutingBroadcaster>) -> Self {
        Self {
            broadcaster,
        }
    }

    /// Get global broadcaster instance (singleton pattern)
    /// This will be initialized in controller_main
    pub fn get_broadcaster() -> Arc<RoutingBroadcaster> {
        static INSTANCE: std::sync::OnceLock<Arc<RoutingBroadcaster>> = std::sync::OnceLock::new();
        INSTANCE
            .get_or_init(|| Arc::new(RoutingBroadcaster::new(1000)))
            .clone()
    }
}

#[tonic::async_trait]
impl RoutingService for RoutingServiceImpl {
    async fn get_routing(
        &self,
        request: Request<GetRoutingReq>,
    ) -> Result<Response<GetRoutingResp>, Status> {
        let req = request.into_inner();

        // Convert to Option<Vec<String>>
        let expert_ids = if req.expert_ids.is_empty() {
            None
        } else {
            Some(req.expert_ids)
        };

        let response = self.broadcaster.get_routing(expert_ids).await;

        log::debug!(
            "GetRouting: returned {} experts, version {}",
            response.routing.len(),
            response.version
        );

        Ok(Response::new(response))
    }

    type SubscribeRoutingUpdatesStream = ReceiverStream<Result<RoutingUpdate, Status>>;

    async fn subscribe_routing_updates(
        &self,
        request: Request<SubscribeRoutingReq>,
    ) -> Result<Response<Self::SubscribeRoutingUpdatesStream>, Status> {
        let req = request.into_inner();
        let current_version = req.current_version;

        log::info!(
            "Frontend subscribed to routing updates (current_version: {})",
            current_version
        );

        // Create channel for streaming updates
        let (tx, rx) = mpsc::channel(128);

        // Subscribe to broadcaster
        let mut update_rx = self.broadcaster.subscribe();

        // Spawn task to forward updates to gRPC stream
        tokio::spawn(async move {
            while let Ok(update) = update_rx.recv().await {
                // Only send updates newer than client's current version
                if update.version > current_version
                    && tx.send(Ok(update)).await.is_err() {
                        // Client disconnected
                        log::debug!("Frontend disconnected from routing updates");
                        break;
                    }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}
