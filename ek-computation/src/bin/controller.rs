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

    let state_srv = tokio::task::spawn(async {
        let srv = controller::service::state::StateServerImpl::new();
        let addr = "[::1]:5001".parse().unwrap();
        log::info!("state server listening on {}", addr);
        let err = tonic::transport::Server::builder()
            .add_service(StateServiceServer::new(srv))
            .serve(addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {:?}", e);
        }
    });

    let computation_srv = tokio::task::spawn(async {
        let srv = controller::service::compute::ComputationProxyServiceImpl::new();
        let addr = "[::1]:5002".parse().unwrap();
        log::info!("computation server listening on {}", addr);
        let err = tonic::transport::Server::builder()
            .add_service(ComputationServiceServer::new(srv))
            .serve(addr)
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
