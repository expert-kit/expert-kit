use ek_base::error::EKResult;
use ek_computation::controller::controller_main;

#[tokio::main]
async fn main() -> EKResult<()> {
    controller_main().await
}
