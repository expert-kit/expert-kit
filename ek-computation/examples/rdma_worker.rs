use ek_computation::shmq::{RdmaQueue, RdmaQueueError};
use std::io::{self, BufRead, BufReader, Write};
use std::net::TcpStream;
use std::thread;
use std::time::Duration;

const QUEUE_CAPACITY: usize = 10;

#[derive(serde::Serialize, serde::Deserialize)]
struct ConnectionInfo {
    endpoint: String,
    memory_region: String,
}

fn get_user_input(prompt: &str) -> io::Result<String> {
    println!("{}", prompt);
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn main() -> io::Result<()> {
    env_logger::init();
    println!("🔧 Starting RDMA Worker (Receiver)");

    // Create receiver queue
    let mut receiver_queue: RdmaQueue<String> = RdmaQueue::new(None, QUEUE_CAPACITY, false)?;
    println!(
        "✅ Created receiver queue with capacity: {}",
        receiver_queue.capacity()
    );

    // Get local endpoint and memory region info
    let local_endpoint = receiver_queue.endpoint()?;
    let local_memory = receiver_queue.memory_region();

    println!("📡 Generated local RDMA endpoint and memory region");

    // Serialize the connection info
    let endpoint_json = serde_json::to_string(&local_endpoint)
        .map_err(|e| io::Error::other(format!("Failed to serialize endpoint: {}", e)))?;
    let memory_json = serde_json::to_string(&local_memory)
        .map_err(|e| io::Error::other(format!("Failed to serialize memory region: {}", e)))?;

    let worker_info = ConnectionInfo {
        endpoint: endpoint_json,
        memory_region: memory_json,
    };

    // Get controller connection details from user
    println!("\n🌐 TCP Connection Setup");
    let controller_ip = get_user_input("📝 Enter controller IP address (default: 127.0.0.1):")?;
    let controller_ip = if controller_ip.is_empty() {
        "127.0.0.1".to_string()
    } else {
        controller_ip
    };

    let controller_port = get_user_input("📝 Enter controller port (default: 8888):")?;
    let controller_port: u16 = if controller_port.is_empty() {
        8888
    } else {
        controller_port.parse().map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid port: {}", e))
        })?
    };

    // Connect to controller via TCP
    println!(
        "\n🔗 Connecting to controller at {}:{}...",
        controller_ip, controller_port
    );
    let stream = TcpStream::connect(format!("{}:{}", controller_ip, controller_port))?;
    println!("✅ Connected to controller");

    // Exchange RDMA connection information
    println!("\n🔄 Exchanging RDMA connection information...");

    // Receive controller's RDMA info
    let mut reader = BufReader::new(stream);
    let mut controller_info_line = String::new();
    reader.read_line(&mut controller_info_line)?;
    let controller_info: ConnectionInfo = serde_json::from_str(controller_info_line.trim())
        .map_err(|e| io::Error::other(format!("Failed to parse controller info: {}", e)))?;
    println!("📥 Received controller RDMA info");

    // Get the stream back for writing
    let mut stream = reader.into_inner();

    // Send worker's RDMA info
    let worker_info_json = serde_json::to_string(&worker_info)
        .map_err(|e| io::Error::other(format!("Failed to serialize worker info: {}", e)))?;
    writeln!(stream, "{}", worker_info_json)?;
    stream.flush()?;
    println!("📤 Sent worker RDMA info to controller");

    // Parse controller's RDMA connection info
    let controller_endpoint: ibverbs::QueuePairEndpoint =
        serde_json::from_str(&controller_info.endpoint)
            .map_err(|e| io::Error::other(format!("Failed to parse controller endpoint: {}", e)))?;
    let controller_memory: ibverbs::RemoteMemoryRegion =
        serde_json::from_str(&controller_info.memory_region).map_err(|e| {
            io::Error::other(format!("Failed to parse controller memory region: {}", e))
        })?;

    println!("✅ Successfully parsed controller RDMA connection info");

    // Establish RDMA connection
    println!("\n🔗 Establishing RDMA connection...");
    match receiver_queue.connect(controller_endpoint, controller_memory) {
        Ok(()) => {
            println!("✅ RDMA connection established successfully!");
        }
        Err(e) => {
            println!("❌ Failed to establish RDMA connection: {}", e);
            return Err(e);
        }
    }

    // Wait for controller readiness signal
    let mut reader = BufReader::new(stream);
    let mut ready_line = String::new();
    reader.read_line(&mut ready_line)?;
    if ready_line.trim() != "RDMA_READY" {
        return Err(io::Error::other("Controller not ready"));
    }

    // Signal our readiness
    let mut stream = reader.into_inner();
    writeln!(stream, "RDMA_READY")?;
    stream.flush()?;

    println!("✅ Both sides ready for RDMA communication");

    // Close TCP connection (no longer needed)
    drop(stream);
    println!("🔌 TCP connection closed, switching to RDMA");

    // Start receiving messages via RDMA
    println!("\n📥 Starting to receive messages via RDMA...");
    let mut received_count = 0;
    let max_attempts = 100; // Give more time for RDMA messages

    for attempt in 1..=max_attempts {
        match receiver_queue.recv() {
            Ok(message) => {
                received_count += 1;
                println!("✅ Received message {}: '{}'", received_count, message);

                // If we've received several messages, we might be done
                if received_count >= 4 {
                    println!("🎉 Received expected number of messages!");
                    break;
                }
            }
            Err(RdmaQueueError::Empty) => {
                if attempt % 20 == 0 {
                    println!("⏳ Waiting for RDMA messages... (attempt {})", attempt);
                }
                thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                println!("❌ Error receiving message (attempt {}): {}", attempt, e);
                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    if received_count == 0 {
        println!("⚠️  No messages received. Possible issues:");
        println!("   1. Controller not sending messages");
        println!("   2. RDMA devices not properly configured");
        println!("   3. RDMA connection establishment failed");
        println!("   4. Network connectivity issues");
    } else {
        println!(
            "\n🎉 Worker successfully received {} messages via RDMA!",
            received_count
        );
    }

    // Keep the queue alive for a bit longer
    println!("\n⏳ Keeping connection alive for 2 seconds...");
    thread::sleep(Duration::from_secs(2));

    // Clean up
    receiver_queue.disconnect();
    println!("🔄 Worker shutting down");

    Ok(())
}
