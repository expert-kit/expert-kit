use ek_computation::shmq::RdmaQueue;
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpListener;
use std::thread;
use std::time::Duration;

const QUEUE_CAPACITY: usize = 10;
const TCP_PORT: u16 = 8888;

#[derive(serde::Serialize, serde::Deserialize)]
struct ConnectionInfo {
    endpoint: String,
    memory_region: String,
}

fn main() -> io::Result<()> {
    env_logger::init();
    println!("🚀 Starting RDMA Controller (Sender)");

    // Create sender queue
    let mut sender_queue: RdmaQueue<String> = RdmaQueue::new(None, QUEUE_CAPACITY, true)?;
    println!("✅ Created sender queue with capacity: {}", sender_queue.capacity());

    // Get local endpoint and memory region info
    let local_endpoint = sender_queue.endpoint()?;
    let local_memory = sender_queue.memory_region();
    
    println!("📡 Generated local RDMA endpoint and memory region");

    // Serialize the connection info
    let endpoint_json = serde_json::to_string(&local_endpoint)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to serialize endpoint: {}", e)))?;
    let memory_json = serde_json::to_string(&local_memory)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to serialize memory region: {}", e)))?;
    
    let controller_info = ConnectionInfo {
        endpoint: endpoint_json,
        memory_region: memory_json,
    };

    // Start TCP server to exchange RDMA endpoint info
    println!("\n🌐 Starting TCP server on port {} for endpoint exchange", TCP_PORT);
    let listener = TcpListener::bind(format!("0.0.0.0:{}", TCP_PORT))?;
    println!("✅ TCP server listening on 0.0.0.0:{}", TCP_PORT);
    println!("💡 Start the worker and provide this controller's IP address and port {}", TCP_PORT);

    println!("\n⏳ Waiting for worker connection...");
    let (mut stream, addr) = listener.accept()?;
    println!("✅ Worker connected from: {}", addr);

    // Exchange RDMA connection information
    println!("\n🔄 Exchanging RDMA connection information...");
    
    // Send controller's RDMA info
    let controller_info_json = serde_json::to_string(&controller_info)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to serialize controller info: {}", e)))?;
    writeln!(stream, "{}", controller_info_json)?;
    stream.flush()?;
    println!("📤 Sent controller RDMA info to worker");

    // Receive worker's RDMA info
    let mut reader = BufReader::new(stream);
    let mut worker_info_line = String::new();
    reader.read_line(&mut worker_info_line)?;
    let worker_info: ConnectionInfo = serde_json::from_str(worker_info_line.trim())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to parse worker info: {}", e)))?;
    println!("📥 Received worker RDMA info");

    // Parse worker's RDMA connection info
    let worker_endpoint: ibverbs::QueuePairEndpoint = serde_json::from_str(&worker_info.endpoint)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to parse worker endpoint: {}", e)))?;
    let worker_memory: ibverbs::RemoteMemoryRegion = serde_json::from_str(&worker_info.memory_region)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Failed to parse worker memory region: {}", e)))?;

    println!("✅ Successfully parsed worker RDMA connection info");

    // Establish RDMA connection
    println!("\n🔗 Establishing RDMA connection...");
    match sender_queue.connect(worker_endpoint, worker_memory) {
        Ok(()) => {
            println!("✅ RDMA connection established successfully!");
        }
        Err(e) => {
            println!("❌ Failed to establish RDMA connection: {}", e);
            return Err(e);
        }
    }

    // Get the stream back for writing
    let mut stream = reader.into_inner();
    
    // Signal readiness over TCP
    writeln!(stream, "RDMA_READY")?;
    stream.flush()?;
    
    // Wait for worker to be ready
    let mut reader = BufReader::new(stream);
    let mut ready_line = String::new();
    reader.read_line(&mut ready_line)?;
    if ready_line.trim() != "RDMA_READY" {
        return Err(io::Error::new(io::ErrorKind::Other, "Worker not ready"));
    }
    
    println!("✅ Both sides ready for RDMA communication");

    // Close TCP connection (no longer needed)
    drop(reader);
    println!("🔌 TCP connection closed, switching to RDMA");

    // Send test messages via RDMA
    println!("\n📤 Sending test messages via RDMA...");
    
    let test_messages = vec![
        "Hello, RDMA World!".to_string(),
        "Message 2: Testing RDMA queue".to_string(),
        "Message 3: RDMA is working perfectly!".to_string(),
        "Final message from controller".to_string(),
    ];
    
    for (i, message) in test_messages.iter().enumerate() {
        match sender_queue.send(message) {
            Ok(()) => {
                println!("✅ Sent message {}: '{}'", i + 1, message);
            }
            Err(e) => {
                println!("❌ Failed to send message {}: {}", i + 1, e);
            }
        }
        thread::sleep(Duration::from_millis(1000));
    }
    
    println!("\n🎉 Controller finished sending all messages via RDMA!");
    println!("💡 The worker should have received and printed these messages");
    
    // Keep the queue alive for a bit longer
    println!("\n⏳ Keeping connection alive for 5 seconds...");
    thread::sleep(Duration::from_secs(5));
    
    // Clean up
    sender_queue.close();
    println!("🔄 Controller shutting down");
    
    Ok(())
}