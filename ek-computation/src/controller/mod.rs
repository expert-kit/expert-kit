pub mod dispatcher;
pub mod elastic;
pub mod executor;
pub mod load_tracker;
pub mod poller;
pub mod registry;
pub mod routing_broadcaster;
pub mod scheduler;
pub mod service;
pub mod v2_state;
pub mod v2_store;

use crate::{
    metrics,
    proto::ek::control::{
        v1::{
            plan_service_server::PlanServiceServer, routing_service_server::RoutingServiceServer,
        },
        v2::{
            topology_service_server::TopologyServiceServer,
            weight_control_service_server::WeightControlServiceServer,
            worker_lifecycle_service_server::WorkerLifecycleServiceServer,
        },
    },
    state::io::StateReaderImpl,
};
use ek_base::error::EKResult;
use metrics::spawn_metrics_server;
use service::{
    control::PlanServiceImpl,
    v2::{DatabaseLifecycleHooks, TopologyServiceImpl, WorkerLifecycleServiceImpl},
    v2_weight::{DatabaseWeightControlHooks, WeightControlServiceImpl},
};
use std::{sync::Arc, time::Duration};

use super::{
    controller::{self, poller::start_poll},
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_server::StateServiceServer,
    },
};

pub async fn controller_main() -> EKResult<()> {
    let settings = ek_base::config::get_ek_settings();
    let v2_state = v2_state::ControllerV2State::restore(
        256,
        v2_store::PostgresControllerStateStore::shared(),
    )
    .await
    .map_err(|error| ek_base::error::EKError::RuntimeError(error.to_string()))?;
    let lifecycle_service = WorkerLifecycleServiceImpl::new(
        v2_state.clone(),
        Arc::new(DatabaseLifecycleHooks::new()),
        Duration::from_secs(
            settings
                .controller
                .fault_detection
                .heartbeat_timeout_secs,
        ),
    );
    let weight_control_service = WeightControlServiceImpl::new(
        v2_state.clone(),
        Arc::new(DatabaseWeightControlHooks::new()),
    );
    let topology_service = TopologyServiceImpl::new(v2_state);

    spawn_metrics_server("0.0.0.0:9080");

    // Initialize routing infrastructure
    let broadcaster = routing_broadcaster::get_broadcaster();
    let load_tracker = Arc::new(load_tracker::LoadTracker::new());
    let state_reader = Arc::new(StateReaderImpl::new());
    let _scheduler = Arc::new(scheduler::WorkerScheduler::new(
        state_reader,
        load_tracker.clone(),
    ));

    // Clone for use in computation server and intra server
    let broadcaster_clone = broadcaster.clone();
    let broadcaster_intra = broadcaster.clone();

    let state_srv = tokio::task::spawn(async move {
        let srv = controller::service::state::StateServerImpl::new();
        let routing_srv =
            controller::service::routing::RoutingServiceImpl::new(broadcaster_intra);
        let intra_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.intra
        )
        .parse()
        .unwrap();
        log::info!("state server listening on {intra_addr}");
        let err = tonic::transport::Server::builder()
            .add_service(StateServiceServer::new(srv))
            .add_service(
                WorkerLifecycleServiceServer::new(lifecycle_service)
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024),
            )
            .add_service(
                WeightControlServiceServer::new(weight_control_service)
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024),
            )
            .add_service(RoutingServiceServer::new(routing_srv))
            .serve(intra_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {e:?}");
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

        // let layer = tower::ServiceBuilder::new()
        //     .layer_fn(OTelGrpcServerMiddleware::new)
        //     .into_inner();

        log::info!("computation server listening on {inter_addr}");
        let plan_srv = PlanServiceImpl::new();
        let routing_srv =
            controller::service::routing::RoutingServiceImpl::new(broadcaster_clone);
        let err = tonic::transport::Server::builder()
            // .layer(layer)
            .add_service(
                ComputationServiceServer::new(srv)
                    .max_decoding_message_size(1024 * 1024 * 1024)
                    .max_encoding_message_size(1024 * 1024 * 1024),
            )
            .add_service(PlanServiceServer::new(plan_srv))
            .add_service(
                TopologyServiceServer::new(topology_service)
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024),
            )
            .add_service(RoutingServiceServer::new(routing_srv))
            .serve(inter_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {e:?}");
        }
    });

    start_poll(broadcaster);

    log::info!("expert kit controller started");
    state_srv.await?;
    computation_srv.await?;
    Ok(())
}
