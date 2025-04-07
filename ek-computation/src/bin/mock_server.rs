use std::io::Cursor;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use log::info;
use tokio::sync::Mutex;
use tokio::time::sleep;
use tonic::{transport::Server, Request, Response, Status};

use ek_computation::proto::ek::worker::v1::computation_service_server::{ComputationService, ComputationServiceServer};
use ek_computation::proto::ek::worker::v1::{
    ForwardReq, ForwardResp
};

// Configuration struct
#[derive(Debug, Clone)]
struct Config {
    expert_dim: usize,
    latency_ms: u64,
    num_experts: usize,
}

// Stats tracking
#[derive(Debug, Default)]
struct ServerStats {
    request_count: u64,
    total_tokens_processed: u64,
    total_unique_tokens_processed: u64,
    total_processing_time_ms: u64,
}

// Mock service implementation
struct ExpertKitService {
    config: Config,
    stats: Arc<Mutex<ServerStats>>,
}

impl ExpertKitService {
    fn new(config: Config) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ServerStats::default())),
        }
    }
}

#[tonic::async_trait]
impl ComputationService for ExpertKitService {
    async fn forward(
        &self,
        request: Request<ForwardReq>,
    ) -> Result<Response<ForwardResp>, Status> {
        let start_time = Instant::now();
        let request = request.into_inner();
        
        // Extract stats
        let mut stats = self.stats.lock().await;
        stats.request_count += 1;
        
        // Log request info
        let num_sequences = request.sequences.len();
        stats.total_tokens_processed += num_sequences as u64;
        
        info!("Received request for instance {}", request.instance_id);
        info!("Processing {} sequences", num_sequences);
        
        // Apply configured latency
        if self.config.latency_ms > 0 {
            sleep(Duration::from_millis(self.config.latency_ms)).await;
        }
        
        // Process the request
        match self.process_request(request, &mut stats).await {
            Ok(resp) => {
                // Record processing time
                let processing_time = start_time.elapsed();
                stats.total_processing_time_ms += processing_time.as_millis() as u64;
                
                // Log performance stats
                info!("Request processed in {:.2?}", processing_time);
                let avg_time_per_token = stats.total_processing_time_ms as f64 / stats.total_tokens_processed as f64;
                info!("Average processing time: {:.2}ms per token", avg_time_per_token);
                
                if stats.total_tokens_processed > 0 {
                    let unique_ratio = stats.total_unique_tokens_processed as f64 / stats.total_tokens_processed as f64;
                    info!("Duplication ratio: {:.2}% unique tokens overall", unique_ratio * 100.0);
                }
                
                Ok(Response::new(resp))
            },
            Err(e) => Err(e),
        }
    }
}

impl ExpertKitService {
    async fn process_request(&self, request: ForwardReq, stats: &mut ServerStats) -> Result<ForwardResp, Status> {
        // Deserialize the tensor data to extract shape
        let input_tensor = match tch::Tensor::load_from_stream(&mut Cursor::new(&request.tensor)) {
            Ok(tensor) => tensor,
            Err(e) => return Err(Status::internal(format!("Failed to deserialize tensor: {}", e))),
        };
        
        // Get dimensions
        let shapes = input_tensor.size();
        if shapes.len() != 2 {
            return Err(Status::invalid_argument(format!(
                "Expected 2D tensor, got {}D", shapes.len()
            )));
        }
        
        let batch_size = shapes[0] as usize;
        let hidden_dim = shapes[1] as usize;
        
        // Validate batch size
        if batch_size != request.sequences.len() {
            return Err(Status::invalid_argument(format!(
                "Batch size mismatch: tensor has {} but sequences has {}",
                batch_size, request.sequences.len()
            )));
        }
        
        info!("Input tensor shape: [{}, {}]", batch_size, hidden_dim);
        
        // Estimate unique tensors
        stats.total_unique_tokens_processed += 1;
        
        // Count experts per sequence
        let mut num_experts_per_seq = Vec::new();
        for seq in &request.sequences {
            num_experts_per_seq.push(seq.experts.len());
        }
        let max_experts = num_experts_per_seq.iter().max().unwrap_or(&0);
        
        
        // let output_shape = vec![batch_size, *max_experts, self.config.expert_dim];
        let output_tensor = tch::Tensor::rand(
            &[batch_size as i64, *max_experts as i64, self.config.expert_dim as i64],
            (tch::Kind::Float, tch::Device::Cpu)
        );
        
        info!("Output tensor shape: [{}, {}, {}]", batch_size, max_experts, self.config.expert_dim);
        
        // Serialize the tensor to bytes
        let mut buffer = Vec::new();
        match output_tensor.save_to_stream(&mut buffer) {
            Ok(_) => (),
            Err(e) => return Err(Status::internal(format!("Failed to serialize tensor: {}", e))),
        }
        
        Ok(ForwardResp {
            output_tensor: buffer,
        })
    }
}

// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct CliArgs {
    /// Port to listen on
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Dimension of expert output
    #[arg(long, default_value_t = 2048)]
    expert_dim: usize,

    /// Simulated latency in milliseconds
    #[arg(long, default_value_t = 0)]
    latency_ms: u64,

    /// Number of experts
    #[arg(short, long, default_value_t = 256)]
    num_experts: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    
    // Parse command line args
    let args = CliArgs::parse();
    
    // Create config from command line arguments
    let config = Config {
        expert_dim: args.expert_dim,
        latency_ms: args.latency_ms,
        num_experts: args.num_experts,
    };
    
    info!("Configuration: expert_dim={}, latency_ms={}, num_experts={}", 
          config.expert_dim, config.latency_ms, config.num_experts);
    
    // Create the expert service
    let expert_service = ExpertKitService::new(config.clone());
    let expert_service_server = ComputationServiceServer::new(expert_service);
    
    // Start the gRPC server
    let addr = format!("0.0.0.0:{}", args.port).parse()?;
    info!("Starting server on {}", addr);
    
    Server::builder()
        .add_service(expert_service_server)
        .serve(addr)
        .await?;
    Ok(())
}