use ek_computation::proto::ek::worker::v1::{
    ForwardReq, computation_service_client::ComputationServiceClient,
};
use ek_computation::tch_safetensors::write_safetensors;
use tch::Device;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = ComputationServiceClient::connect("http://[::1]:50051").await?;
    let tensor = tch::Tensor::rand([1, 2048], (tch::Kind::Float, Device::Cpu));
    let vec = write_safetensors(&[("input".to_string(), tensor)]).unwrap();
    let request = tonic::Request::new(ForwardReq {
        tensor: vec,
        instance_id: "0".into(),
        sequences: vec![],
    });
    let resp = client.forward(request).await?;
    let output_size = resp.get_ref().output_tensor.len();
    println!("simple forward success output_size={}", output_size);
    Ok(())
}
