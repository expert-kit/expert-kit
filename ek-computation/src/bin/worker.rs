#![feature(pattern)]
use ek_base::error::EKResult;
use ek_computation::worker::worker_main;

#[tokio::main(flavor = "multi_thread", worker_threads = 48)]
async fn main() -> EKResult<()> {
    worker_main().await
}
