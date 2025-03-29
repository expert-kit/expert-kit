pub mod ekproto {
    tonic::include_proto!("ek");
}

use ek_computation::tch_safetensors::write_safetensors;
use ekproto::ExpertForwardRequest;
use ekproto::expert_computation_client::ExpertComputationClient;
use tch::Device;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ExpertComputationClient::connect("http://[::1]:50051").await?;
    let tensor = tch::Tensor::rand([1, 2048], (tch::Kind::Float, Device::Cpu));
    let vec = write_safetensors(&[("input".to_string(), tensor)]).unwrap();
    let request = tonic::Request::new(ExpertForwardRequest {
        tensor: vec,
        layer: 0,
        idx: 0,
        batch_size: 0,
    });
    let resp = client.forward(request).await?;
    let output_size = resp.get_ref().output_tensor.len();
    println!("simple forward success output_size={}", output_size);
    Ok(())
}
