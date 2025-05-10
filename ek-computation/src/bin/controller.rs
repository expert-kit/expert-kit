use ek_computation::{
    controller::{self, poller::start_poll},
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_server::StateServiceServer,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("debug"));
    let settings = ek_base::config::get_ek_settings();

    let state_srv = tokio::task::spawn(async {
        let srv = controller::service::state::StateServerImpl::new();
        let intra_addr = settings.controller.intra_listen.to_socket_addr();
        log::info!("state server listening on {}", intra_addr);
        let err = tonic::transport::Server::builder()
            .add_service(StateServiceServer::new(srv))
            .serve(intra_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {:?}", e);
        }
    });

    let computation_srv = tokio::task::spawn(async {
        let srv = controller::service::compute::ComputationProxyServiceImpl::new();
        let inter_addr = settings.controller.inter_listen.to_socket_addr();
        log::info!("computation server listening on {}", inter_addr);
        let err = tonic::transport::Server::builder()
            .add_service(ComputationServiceServer::new(srv))
            .serve(inter_addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {:?}", e);
        }
    });

    start_poll();

    log::info!("expert kit controller started");
    state_srv.await?;
    computation_srv.await?;
    Ok(())
}
