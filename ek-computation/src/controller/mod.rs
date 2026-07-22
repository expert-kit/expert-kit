pub mod dispatcher;
pub mod elastic;
pub mod poller;
pub mod runtime_state;
pub mod scheduler;
pub mod service;
pub mod state_store;

use crate::proto::ek::control::{
    v1::plan_service_server::PlanServiceServer,
    v2::{
        instance_service_server::InstanceServiceServer,
        topology_service_server::TopologyServiceServer,
        weight_control_service_server::WeightControlServiceServer,
        worker_lifecycle_service_server::WorkerLifecycleServiceServer,
    },
};
use ek_base::error::EKResult;
use service::{
    control::PlanServiceImpl,
    instance::{DatabaseDefaultInstanceResolver, DefaultInstanceResolver, InstanceServiceImpl},
    v2::{DatabaseLifecycleHooks, TopologyServiceImpl, WorkerLifecycleServiceImpl},
    v2_weight::{DatabaseWeightControlHooks, WeightControlServiceImpl},
};
use std::{sync::Arc, time::Duration};

use super::controller::poller::start_poll;

pub async fn controller_main() -> EKResult<()> {
    let settings = ek_base::config::get_ek_settings();
    let runtime_state = runtime_state::ControllerRuntimeState::restore(
        256,
        state_store::PostgresControllerStateStore::shared(),
    )
    .await
    .map_err(|error| ek_base::error::EKError::RuntimeError(error.to_string()))?;
    let instance_resolver: Arc<dyn DefaultInstanceResolver> =
        Arc::new(DatabaseDefaultInstanceResolver::new(
            settings.inference.model_name.clone(),
            settings.inference.instance_name.clone(),
        ));
    let worker_instance_service = InstanceServiceImpl::new(instance_resolver.clone());
    let frontend_instance_service = InstanceServiceImpl::new(instance_resolver.clone());
    let lifecycle_service = WorkerLifecycleServiceImpl::new(
        runtime_state.clone(),
        Arc::new(DatabaseLifecycleHooks::new()),
        instance_resolver.clone(),
        Duration::from_secs(settings.controller.fault_detection.heartbeat_timeout_secs),
    );
    let weight_control_service = WeightControlServiceImpl::new(
        runtime_state.clone(),
        Arc::new(DatabaseWeightControlHooks::new()),
    );
    let topology_service = TopologyServiceImpl::new(runtime_state, instance_resolver);

    let worker_control_srv = tokio::task::spawn(async move {
        let intra_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.intra
        )
        .parse()
        .unwrap();
        log::info!("worker control server listening on {intra_addr}");
        let err = tonic::transport::Server::builder()
            .add_service(
                InstanceServiceServer::new(worker_instance_service)
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024),
            )
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
            .serve(intra_addr)
            .await;
        if let Err(e) = err {
            log::error!("worker control server error {e:?}");
        }
    });

    let frontend_control_srv = tokio::task::spawn(async {
        let inter_addr = format!(
            "{}:{}",
            settings.controller.listen, settings.controller.ports.inter
        )
        .parse()
        .unwrap();

        log::info!("frontend control server listening on {inter_addr}");
        let plan_srv = PlanServiceImpl::new();
        let err = tonic::transport::Server::builder()
            .add_service(PlanServiceServer::new(plan_srv))
            .add_service(
                InstanceServiceServer::new(frontend_instance_service)
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024),
            )
            .add_service(
                TopologyServiceServer::new(topology_service)
                    .max_decoding_message_size(1024 * 1024)
                    .max_encoding_message_size(1024 * 1024),
            )
            .serve(inter_addr)
            .await;
        if let Err(e) = err {
            log::error!("frontend control server error {e:?}");
        }
    });

    start_poll();

    log::info!("expert kit controller started");
    worker_control_srv.await?;
    frontend_control_srv.await?;
    Ok(())
}
