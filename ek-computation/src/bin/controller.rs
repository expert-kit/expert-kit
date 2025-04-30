use ek_computation::{
    controller,
    proto::ek::worker::v1::{
        computation_service_server::ComputationServiceServer,
        state_service_server::StateServiceServer,
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("hello");
    let state_srv = tokio::task::spawn(async {
        let srv = controller::service::state::StateServerImpl::new();
        let addr = "[::1]:5001".parse().unwrap();
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
        let err = tonic::transport::Server::builder()
            .add_service(ComputationServiceServer::new(srv))
            .serve(addr)
            .await;
        if let Err(e) = err {
            log::error!("state server error {:?}", e);
        }
    });

    state_srv.await?;
    computation_srv.await?;
    Ok(())
}
