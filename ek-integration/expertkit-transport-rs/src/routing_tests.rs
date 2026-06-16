use super::*;

#[tokio::test]
async fn test_routing_client_creation() {
    let client = RoutingClient::new("localhost:5002".to_string());
    assert_eq!(client.get_version().await, 0);
    assert!(client.get_all_routing().await.is_empty());
}

#[tokio::test]
async fn test_routing_table_operations() {
    let client = RoutingClient::new("localhost:5002".to_string());

    // Manually populate for testing (now with Vec<WorkerEndpoint>)
    {
        let mut table = client.routing_table.write().await;
        table.insert(
            "expert_1".to_string(),
            vec![WorkerEndpoint {
                grpc_addr: "worker1:50051".to_string(),
                channel: "grpc".to_string(),
                rdma_tcp_port: 0,
                shm_queue_prefix: "".to_string(),
                device: "cpu".to_string(),
                wm_addr: "".to_string(),
            }],
        );
        table.insert(
            "expert_2".to_string(),
            vec![WorkerEndpoint {
                grpc_addr: "worker2:50051".to_string(),
                channel: "grpc".to_string(),
                rdma_tcp_port: 0,
                shm_queue_prefix: "".to_string(),
                device: "cpu".to_string(),
                wm_addr: "".to_string(),
            }],
        );
    }

    let worker1 = client.get_worker("expert_1").await;
    assert!(worker1.is_some());
    assert_eq!(worker1.unwrap().grpc_addr, "worker1:50051");

    let worker2 = client.get_worker("expert_2").await;
    assert!(worker2.is_some());
    assert_eq!(worker2.unwrap().grpc_addr, "worker2:50051");

    assert_eq!(client.get_worker("expert_3").await, None);
}

#[tokio::test]
async fn test_multi_worker_selection() {
    let client = RoutingClient::new("localhost:5002".to_string());

    // Setup: expert with two workers
    {
        let mut table = client.routing_table.write().await;
        table.insert(
            "expert_1".to_string(),
            vec![
                WorkerEndpoint {
                    grpc_addr: "worker1:50051".to_string(),
                    channel: "grpc".to_string(),
                    rdma_tcp_port: 0,
                    shm_queue_prefix: "".to_string(),
                    device: "cpu".to_string(),
                    wm_addr: "".to_string(),
                },
                WorkerEndpoint {
                    grpc_addr: "worker2:50051".to_string(),
                    channel: "grpc".to_string(),
                    rdma_tcp_port: 0,
                    shm_queue_prefix: "".to_string(),
                    device: "cpu".to_string(),
                    wm_addr: "".to_string(),
                },
            ],
        );
    }

    // Initialize stats for workers
    {
        let mut stats = client.worker_stats.write().await;
        stats.insert(
            "worker1:50051".to_string(),
            Arc::new(WorkerStats::default()),
        );
        stats.insert(
            "worker2:50051".to_string(),
            Arc::new(WorkerStats::default()),
        );
    }

    // With equal stats, should return one of the workers
    let worker = client.select_worker("expert_1").await;
    assert!(worker.is_some());

    // Simulate worker1 being slower
    {
        let stats = client.worker_stats.read().await;
        if let Some(s) = stats.get("worker1:50051") {
            // Update RTT to be higher
            for _ in 0..10 {
                s.update_rtt(100.0);
            }
        }
        if let Some(s) = stats.get("worker2:50051") {
            // Keep worker2 fast
            for _ in 0..10 {
                s.update_rtt(10.0);
            }
        }
    }

    // Should now prefer worker2
    let worker = client.select_worker("expert_1").await;
    assert!(worker.is_some());
    assert_eq!(worker.unwrap().grpc_addr, "worker2:50051");
}

#[tokio::test]
async fn test_inflight_penalty() {
    let client = RoutingClient::new("localhost:5002".to_string());

    // Setup: expert with two workers
    {
        let mut table = client.routing_table.write().await;
        table.insert(
            "expert_1".to_string(),
            vec![
                WorkerEndpoint {
                    grpc_addr: "worker1:50051".to_string(),
                    channel: "grpc".to_string(),
                    rdma_tcp_port: 0,
                    shm_queue_prefix: "".to_string(),
                    device: "cpu".to_string(),
                    wm_addr: "".to_string(),
                },
                WorkerEndpoint {
                    grpc_addr: "worker2:50051".to_string(),
                    channel: "grpc".to_string(),
                    rdma_tcp_port: 0,
                    shm_queue_prefix: "".to_string(),
                    device: "cpu".to_string(),
                    wm_addr: "".to_string(),
                },
            ],
        );
    }

    // Initialize stats with equal RTT
    {
        let mut stats = client.worker_stats.write().await;
        let stats1 = Arc::new(WorkerStats::default());
        let stats2 = Arc::new(WorkerStats::default());
        // Set equal RTT
        stats1.update_rtt(20.0);
        stats2.update_rtt(20.0);
        stats.insert("worker1:50051".to_string(), stats1);
        stats.insert("worker2:50051".to_string(), stats2);
    }

    // Add inflight requests to worker1
    let worker1_endpoint = WorkerEndpoint {
        grpc_addr: "worker1:50051".to_string(),
        channel: "grpc".to_string(),
        rdma_tcp_port: 0,
        shm_queue_prefix: "".to_string(),
        device: "cpu".to_string(),
        wm_addr: "".to_string(),
    };

    // inflight requests to worker1
    for _ in 0..3 {
        client.on_request_start(&worker1_endpoint).await;
    }

    // Should now prefer worker2 due to inflight penalty
    let worker = client.select_worker("expert_1").await;
    assert!(worker.is_some());
    assert_eq!(worker.unwrap().grpc_addr, "worker2:50051");
}

#[tokio::test]
async fn test_atomic_selection_distributes_load() {
    let client = RoutingClient::new("localhost:5002".to_string());

    // Setup: expert with two workers
    {
        let mut table = client.routing_table.write().await;
        table.insert(
            "expert_1".to_string(),
            vec![
                WorkerEndpoint {
                    grpc_addr: "worker1:50051".to_string(),
                    channel: "grpc".to_string(),
                    rdma_tcp_port: 0,
                    shm_queue_prefix: "".to_string(),
                    device: "cpu".to_string(),
                    wm_addr: "".to_string(),
                },
                WorkerEndpoint {
                    grpc_addr: "worker2:50051".to_string(),
                    channel: "grpc".to_string(),
                    rdma_tcp_port: 0,
                    shm_queue_prefix: "".to_string(),
                    device: "cpu".to_string(),
                    wm_addr: "".to_string(),
                },
            ],
        );
    }

    // Call select_worker_and_mark_inflight multiple times
    // With equal stats, the atomic selection should distribute load
    let mut worker1_count = 0;
    let mut worker2_count = 0;

    for _ in 0..10 {
        let worker = client.select_worker_and_mark_inflight("expert_1").await;
        assert!(worker.is_some());
        match worker.unwrap().grpc_addr.as_str() {
            "worker1:50051" => worker1_count += 1,
            "worker2:50051" => worker2_count += 1,
            _ => panic!("Unexpected worker"),
        }
    }

    // Due to inflight penalty with equal RTT, selections should alternate
    println!("worker1: {}, worker2: {}", worker1_count, worker2_count);
    assert!(
        worker1_count >= 3 && worker2_count >= 3,
        "Load should be distributed: worker1={}, worker2={}",
        worker1_count,
        worker2_count
    );
}

// --- assign_layer_batch / LPT tests ---

fn make_endpoint(addr: &str) -> WorkerEndpoint {
    WorkerEndpoint {
        grpc_addr: addr.to_string(),
        channel: "grpc".to_string(),
        rdma_tcp_port: 0,
        shm_queue_prefix: "".to_string(),
        device: "cpu".to_string(),
        wm_addr: "".to_string(),
    }
}

fn make_endpoint_with_device(addr: &str, device: &str) -> WorkerEndpoint {
    WorkerEndpoint {
        grpc_addr: addr.to_string(),
        channel: "grpc".to_string(),
        rdma_tcp_port: 0,
        shm_queue_prefix: "".to_string(),
        device: device.to_string(),
        wm_addr: "".to_string(),
    }
}

/// Helper: populate a routing table with experts that each have exactly the listed workers.
async fn setup_routing(client: &RoutingClient, entries: &[(&str, &[&str])]) {
    let mut table = client.routing_table.write().await;
    for (expert_id, workers) in entries {
        table.insert(
            expert_id.to_string(),
            workers.iter().map(|a| make_endpoint(a)).collect(),
        );
    }
}

/// Helper: set throughput_tpm for a worker directly via WorkerStats.
async fn set_throughput(client: &RoutingClient, addr: &str, tpm: f64) {
    let mut stats = client.worker_stats.write().await;
    let entry = stats
        .entry(addr.to_string())
        .or_insert_with(|| Arc::new(WorkerStats::default()));
    // Drive the EMA to roughly `tpm` by applying many identical samples
    for _ in 0..50 {
        entry.update_throughput(1, 1.0 / tpm); // token_count=1, rtt=1/tpm → sample_tpm=tpm
    }
}

#[tokio::test]
async fn test_lpt_assigns_all_experts() {
    let client = RoutingClient::new("localhost:5002".to_string());
    setup_routing(
        &client,
        &[
            ("expert_1", &["w1:50051", "w2:50051"]),
            ("expert_2", &["w1:50051", "w2:50051"]),
            ("expert_3", &["w1:50051", "w2:50051"]),
        ],
    )
    .await;

    let calls = vec![
        ("expert_1".to_string(), 8),
        ("expert_2".to_string(), 4),
        ("expert_3".to_string(), 2),
    ];
    let assignments = client.assign_layer_batch(&calls).await;

    assert_eq!(assignments.len(), 3, "all experts must be assigned");
    for (eid, _) in &calls {
        let worker = assignments
            .get(eid)
            .expect("expert missing from assignments");
        assert!(
            worker.grpc_addr == "w1:50051" || worker.grpc_addr == "w2:50051",
            "unknown worker assigned"
        );
    }
}

#[tokio::test]
async fn test_lpt_prefers_faster_worker() {
    // Worker 1 is 4× faster than worker 2.
    // All experts have both workers available.
    // LPT should route most tokens to worker 1.
    let client = RoutingClient::new("localhost:5002".to_string());
    setup_routing(
        &client,
        &[
            ("expert_1", &["w1:50051", "w2:50051"]),
            ("expert_2", &["w1:50051", "w2:50051"]),
            ("expert_3", &["w1:50051", "w2:50051"]),
            ("expert_4", &["w1:50051", "w2:50051"]),
        ],
    )
    .await;

    set_throughput(&client, "w1:50051", 4.0).await; // fast: 4 tokens/ms
    set_throughput(&client, "w2:50051", 1.0).await; // slow: 1 token/ms

    // All experts with equal token counts — LPT should assign 4:1 split.
    let calls: Vec<(String, usize)> = (1..=4).map(|i| (format!("expert_{}", i), 4)).collect();
    let assignments = client.assign_layer_batch(&calls).await;

    let w1_count = assignments
        .values()
        .filter(|w| w.grpc_addr == "w1:50051")
        .count();
    let w2_count = assignments
        .values()
        .filter(|w| w.grpc_addr == "w2:50051")
        .count();

    assert!(
        w1_count > w2_count,
        "faster worker should receive more assignments: w1={}, w2={}",
        w1_count,
        w2_count
    );
}

#[tokio::test]
async fn test_lpt_homogeneous_workers_distribute_evenly() {
    // Equal throughput → LPT should spread experts roughly 50/50.
    let client = RoutingClient::new("localhost:5002".to_string());
    let n = 8usize;
    let entries: Vec<(String, Vec<&str>)> = (1..=n)
        .map(|i| (format!("expert_{}", i), vec!["w1:50051", "w2:50051"]))
        .collect();
    let entry_refs: Vec<(&str, &[&str])> = entries
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    setup_routing(&client, &entry_refs).await;

    set_throughput(&client, "w1:50051", 2.0).await;
    set_throughput(&client, "w2:50051", 2.0).await;

    let calls: Vec<(String, usize)> = (1..=n).map(|i| (format!("expert_{}", i), 4)).collect();
    let assignments = client.assign_layer_batch(&calls).await;

    let w1 = assignments
        .values()
        .filter(|w| w.grpc_addr == "w1:50051")
        .count();
    let w2 = assignments
        .values()
        .filter(|w| w.grpc_addr == "w2:50051")
        .count();

    assert_eq!(w1 + w2, n);
    // Equal capacity → should split within 1 of 50/50
    assert!(
        (w1 as i64 - w2 as i64).abs() <= 2,
        "equal workers should be balanced: w1={}, w2={}",
        w1,
        w2
    );
}

#[tokio::test]
async fn test_lpt_single_worker_experts_assigned_directly() {
    // Experts with only one worker should still be assigned, even mixed with multi-worker experts.
    let client = RoutingClient::new("localhost:5002".to_string());
    setup_routing(
        &client,
        &[
            ("expert_a", &["w1:50051"]),             // single worker
            ("expert_b", &["w1:50051", "w2:50051"]), // two workers
        ],
    )
    .await;

    let calls = vec![("expert_a".to_string(), 8), ("expert_b".to_string(), 4)];
    let assignments = client.assign_layer_batch(&calls).await;

    assert_eq!(assignments.len(), 2);
    assert_eq!(assignments["expert_a"].grpc_addr, "w1:50051");
    assert!(
        assignments["expert_b"].grpc_addr == "w1:50051"
            || assignments["expert_b"].grpc_addr == "w2:50051"
    );
}

#[tokio::test]
async fn test_lpt_missing_expert_excluded_from_result() {
    let client = RoutingClient::new("localhost:5002".to_string());
    setup_routing(&client, &[("expert_1", &["w1:50051"])]).await;

    let calls = vec![
        ("expert_1".to_string(), 4),
        ("expert_missing".to_string(), 4), // not in routing table
    ];
    let assignments = client.assign_layer_batch(&calls).await;

    assert!(assignments.contains_key("expert_1"));
    assert!(!assignments.contains_key("expert_missing"));
}

#[tokio::test]
async fn test_lpt_increments_inflight() {
    let client = RoutingClient::new("localhost:5002".to_string());
    setup_routing(
        &client,
        &[("expert_1", &["w1:50051"]), ("expert_2", &["w1:50051"])],
    )
    .await;

    let calls = vec![("expert_1".to_string(), 4), ("expert_2".to_string(), 4)];
    client.assign_layer_batch(&calls).await;

    let stats = client.worker_stats.read().await;
    let inflight = stats["w1:50051"].get_inflight();
    assert_eq!(
        inflight, 2,
        "inflight should equal number of assigned experts"
    );
}

#[tokio::test]
async fn test_throughput_tracking_via_finalize_layer_tpm() {
    let client = RoutingClient::new("localhost:5002".to_string());
    let endpoint = make_endpoint("w1:50051");

    // Simulate a layer: one request with 10 tokens, RTT = 5 ms → sample tpm = 2.0
    client.on_request_start(&endpoint).await;
    client.on_request_complete(&endpoint, 5.0, 10).await;

    // tpm must NOT be updated yet (before layer barrier)
    {
        let stats = client.worker_stats.read().await;
        let tpm = stats["w1:50051"].get_throughput_tpm();
        assert!(
            (tpm - DEFAULT_THROUGHPUT_TPM).abs() < 1e-9,
            "tpm should not change before finalize_layer_tpm: got {}",
            tpm
        );
        assert_eq!(
            stats["w1:50051"].get_inflight(),
            0,
            "inflight should be 0 after complete"
        );
    }

    // After the layer barrier, tpm is updated with aggregate stats
    client.finalize_layer_tpm().await;

    let stats = client.worker_stats.read().await;
    let tpm = stats["w1:50051"].get_throughput_tpm();
    // EMA: 0.2 * 2.0 + 0.8 * DEFAULT_THROUGHPUT_TPM (0.5) = 0.4 + 0.4 = 0.8
    assert!(
        (tpm - 0.8).abs() < 1e-9,
        "throughput EMA should converge towards layer sample: got {}",
        tpm
    );
}

// --- LB simulation tests for heterogeneous workloads ---

/// Simulate a layer dispatch and compute makespan (max worker completion time).
///
/// Given assignments (expert→worker) and per-worker throughput, calculates
/// each worker's total processing time = sum(token_count / tpm) for assigned experts.
/// Returns (makespan, per_worker_times) where makespan = max of all worker times.
fn simulate_makespan(
    assignments: &HashMap<String, WorkerEndpoint>,
    expert_calls: &[(String, usize)],
    worker_tpm: &HashMap<String, f64>,
) -> (f64, HashMap<String, f64>) {
    let mut worker_times: HashMap<String, f64> = HashMap::new();
    for (expert_id, token_count) in expert_calls {
        if let Some(worker) = assignments.get(expert_id) {
            let tpm = worker_tpm.get(&worker.grpc_addr).copied().unwrap_or(0.5);
            let time = *token_count as f64 / tpm;
            *worker_times.entry(worker.grpc_addr.clone()).or_insert(0.0) += time;
        }
    }
    let makespan = worker_times.values().cloned().fold(0.0_f64, f64::max);
    (makespan, worker_times)
}

/// Helper: setup heterogeneous routing table where all experts have all workers.
async fn setup_hetero_routing(
    client: &RoutingClient,
    num_experts: usize,
    workers: &[(&str, &str)], // (addr, device)
) {
    let mut table = client.routing_table.write().await;
    for i in 0..num_experts {
        let expert_id = format!("model/l0-e{}", i);
        let endpoints: Vec<WorkerEndpoint> = workers
            .iter()
            .map(|(addr, dev)| make_endpoint_with_device(addr, dev))
            .collect();
        table.insert(expert_id, endpoints);
    }
}

#[tokio::test]
async fn test_lb_sim_heterogeneous_rtt_vs_roundrobin() {
    // Scenario: 2 workers (GPU 10× faster than CPU), 16 experts, each with 8 tokens.
    // With multi-replica setup, RTT/LPT should route ~10× more work to GPU.
    // RoundRobin should split 50/50 → much worse makespan.
    let num_experts = 16;
    let tokens_per_expert = 8;

    let gpu_tpm = 10.0; // GPU: 10 tokens/ms
    let cpu_tpm = 1.0; // CPU: 1 token/ms

    let workers = vec![
        ("gpu:50051", "cuda:0"),
        ("cpu:50051", "cpu"),
    ];
    let worker_tpm: HashMap<String, f64> = vec![
        ("gpu:50051".to_string(), gpu_tpm),
        ("cpu:50051".to_string(), cpu_tpm),
    ]
    .into_iter()
    .collect();

    // --- RTT (LPT) algorithm ---
    let rtt_client =
        RoutingClient::new_with_algorithm("localhost:5002".to_string(), lb::LbAlgorithm::Rtt);
    setup_hetero_routing(&rtt_client, num_experts, &workers).await;
    set_throughput(&rtt_client, "gpu:50051", gpu_tpm).await;
    set_throughput(&rtt_client, "cpu:50051", cpu_tpm).await;

    let calls: Vec<(String, usize)> = (0..num_experts)
        .map(|i| (format!("model/l0-e{}", i), tokens_per_expert))
        .collect();
    let rtt_assignments = rtt_client.assign_layer_batch(&calls).await;
    let (rtt_makespan, rtt_times) = simulate_makespan(&rtt_assignments, &calls, &worker_tpm);

    // --- RoundRobin algorithm ---
    let rr_client = RoutingClient::new_with_algorithm(
        "localhost:5002".to_string(),
        lb::LbAlgorithm::RoundRobin,
    );
    setup_hetero_routing(&rr_client, num_experts, &workers).await;

    let rr_assignments = rr_client.assign_layer_batch(&calls).await;
    let (rr_makespan, rr_times) = simulate_makespan(&rr_assignments, &calls, &worker_tpm);

    // Print results
    println!("=== Heterogeneous LB Simulation (GPU 10×, CPU 1×) ===");
    println!(
        "RTT/LPT: makespan={:.1}ms, gpu={:.1}ms, cpu={:.1}ms",
        rtt_makespan,
        rtt_times.get("gpu:50051").unwrap_or(&0.0),
        rtt_times.get("cpu:50051").unwrap_or(&0.0),
    );
    println!(
        "RoundRobin: makespan={:.1}ms, gpu={:.1}ms, cpu={:.1}ms",
        rr_makespan,
        rr_times.get("gpu:50051").unwrap_or(&0.0),
        rr_times.get("cpu:50051").unwrap_or(&0.0),
    );

    // RTT/LPT should achieve significantly lower makespan than RoundRobin
    assert!(
        rtt_makespan < rr_makespan,
        "RTT/LPT makespan ({:.1}ms) should be lower than RoundRobin ({:.1}ms)",
        rtt_makespan,
        rr_makespan
    );

    // RTT/LPT should give GPU more work than CPU
    let rtt_gpu_count = rtt_assignments
        .values()
        .filter(|w| w.grpc_addr == "gpu:50051")
        .count();
    let rtt_cpu_count = rtt_assignments
        .values()
        .filter(|w| w.grpc_addr == "cpu:50051")
        .count();
    assert!(
        rtt_gpu_count > rtt_cpu_count,
        "LPT should assign more experts to GPU: gpu={}, cpu={}",
        rtt_gpu_count,
        rtt_cpu_count
    );
}

#[tokio::test]
async fn test_lb_sim_all_algorithms_heterogeneous() {
    // Compare all 7 algorithms on a heterogeneous 3-worker setup.
    // Workers: GPU-fast (tpm=8), GPU-slow (tpm=4), CPU (tpm=1).
    // 24 experts, varying token counts (simulating prefill with uneven routing).
    let num_experts = 24;
    let workers = vec![
        ("gpu-fast:50051", "cuda:0"),
        ("gpu-slow:50051", "cuda:1"),
        ("cpu:50051", "cpu"),
    ];
    let worker_tpm: HashMap<String, f64> = vec![
        ("gpu-fast:50051".to_string(), 8.0),
        ("gpu-slow:50051".to_string(), 4.0),
        ("cpu:50051".to_string(), 1.0),
    ]
    .into_iter()
    .collect();

    // Varying token counts: simulate uneven MoE routing in prefill
    let calls: Vec<(String, usize)> = (0..num_experts)
        .map(|i| {
            let tokens = match i % 4 {
                0 => 32, // hot expert
                1 => 16,
                2 => 8,
                _ => 4, // cold expert
            };
            (format!("model/l0-e{}", i), tokens)
        })
        .collect();

    let algorithms = vec![
        ("Rtt", lb::LbAlgorithm::Rtt),
        ("GreedyRtt", lb::LbAlgorithm::GreedyRtt),
        ("GreedyThroughput", lb::LbAlgorithm::GreedyThroughput),
        ("LeastInflight", lb::LbAlgorithm::LeastInflight),
        ("RoundRobin", lb::LbAlgorithm::RoundRobin),
        ("Random", lb::LbAlgorithm::Random),
        ("Hash", lb::LbAlgorithm::Hash),
    ];

    println!("\n=== All Algorithms: 3 heterogeneous workers (8/4/1 tpm) ===");
    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>12}",
        "Algorithm", "Makespan", "GPU-fast", "GPU-slow", "CPU"
    );

    let mut results: Vec<(String, f64)> = Vec::new();

    for (name, algo) in &algorithms {
        let client = RoutingClient::new_with_algorithm(
            "localhost:5002".to_string(),
            *algo,
        );
        setup_hetero_routing(&client, num_experts, &workers).await;
        // Set measured throughput for throughput-aware algorithms
        set_throughput(&client, "gpu-fast:50051", 8.0).await;
        set_throughput(&client, "gpu-slow:50051", 4.0).await;
        set_throughput(&client, "cpu:50051", 1.0).await;

        let assignments = client.assign_layer_batch(&calls).await;
        let (makespan, times) = simulate_makespan(&assignments, &calls, &worker_tpm);

        println!(
            "{:<20} {:>8.1}ms {:>10.1}ms {:>10.1}ms {:>10.1}ms",
            name,
            makespan,
            times.get("gpu-fast:50051").unwrap_or(&0.0),
            times.get("gpu-slow:50051").unwrap_or(&0.0),
            times.get("cpu:50051").unwrap_or(&0.0),
        );
        results.push((name.to_string(), makespan));
    }

    // RTT (LPT) should be the best or near-best
    let rtt_makespan = results.iter().find(|(n, _)| n == "Rtt").unwrap().1;
    let rr_makespan = results.iter().find(|(n, _)| n == "RoundRobin").unwrap().1;

    assert!(
        rtt_makespan < rr_makespan,
        "RTT/LPT ({:.1}ms) should beat RoundRobin ({:.1}ms) on heterogeneous workers",
        rtt_makespan,
        rr_makespan
    );
}

#[tokio::test]
async fn test_lb_sim_device_aware_coldstart() {
    // Test that device-aware warm-start defaults correctly bias cold assignments.
    // Before any measurements, LPT should assign more work to GPU even with zero measurements.
    let num_experts = 10;
    let workers = vec![
        ("gpu:50051", "cuda:0"),
        ("cpu:50051", "cpu"),
    ];

    let client =
        RoutingClient::new_with_algorithm("localhost:5002".to_string(), lb::LbAlgorithm::Rtt);
    setup_hetero_routing(&client, num_experts, &workers).await;
    // Do NOT call set_throughput — we want to test cold-start behavior

    let calls: Vec<(String, usize)> = (0..num_experts)
        .map(|i| (format!("model/l0-e{}", i), 8))
        .collect();
    let assignments = client.assign_layer_batch(&calls).await;

    let gpu_count = assignments
        .values()
        .filter(|w| w.grpc_addr == "gpu:50051")
        .count();
    let cpu_count = assignments
        .values()
        .filter(|w| w.grpc_addr == "cpu:50051")
        .count();

    println!(
        "\n=== Cold-start device-aware test ===\nGPU assigned: {}, CPU assigned: {}",
        gpu_count, cpu_count
    );

    // Device-aware defaults: GPU has 20× higher tpm than CPU.
    // LPT should assign significantly more to GPU even without measurements.
    assert!(
        gpu_count > cpu_count,
        "Cold-start LPT should favor GPU (gpu={}, cpu={})",
        gpu_count,
        cpu_count
    );
}

#[tokio::test]
async fn test_lb_sim_single_replica_no_difference() {
    // Confirm that with single-replica experts, all algorithms produce identical results.
    // Each expert lives on exactly one worker (some on GPU, some on CPU).
    let algorithms = vec![
        lb::LbAlgorithm::Rtt,
        lb::LbAlgorithm::RoundRobin,
        lb::LbAlgorithm::Random,
        lb::LbAlgorithm::LeastInflight,
        lb::LbAlgorithm::Hash,
    ];

    let calls: Vec<(String, usize)> = (0..8)
        .map(|i| (format!("model/l0-e{}", i), 4))
        .collect();

    let mut all_assignments: Vec<Vec<String>> = Vec::new();

    for algo in &algorithms {
        let client = RoutingClient::new_with_algorithm("localhost:5002".to_string(), *algo);
        {
            let mut table = client.routing_table.write().await;
            for i in 0..8 {
                let expert_id = format!("model/l0-e{}", i);
                // Alternate experts between GPU and CPU — but each has only ONE worker
                if i % 2 == 0 {
                    table.insert(
                        expert_id,
                        vec![make_endpoint_with_device("gpu:50051", "cuda:0")],
                    );
                } else {
                    table.insert(
                        expert_id,
                        vec![make_endpoint_with_device("cpu:50051", "cpu")],
                    );
                }
            }
        }

        let assignments = client.assign_layer_batch(&calls).await;
        let addrs: Vec<String> = calls
            .iter()
            .map(|(eid, _)| assignments[eid].grpc_addr.clone())
            .collect();
        all_assignments.push(addrs);
    }

    // All algorithms should produce the exact same assignment
    for i in 1..all_assignments.len() {
        assert_eq!(
            all_assignments[0], all_assignments[i],
            "Single-replica: all algorithms should produce identical assignments"
        );
    }

    println!("\n=== Single-replica test: all algorithms identical ✓ ===");
}

#[tokio::test]
async fn test_reset_stats_clears_ema() {
    // Verify that reset_stats() clears all EMA values, preventing
    // earlier benchmark iterations from biasing later ones.
    let client = RoutingClient::new("localhost:5002".to_string());
    let endpoint = make_endpoint("w1:50051");

    // Build up stats: 10 requests with RTT=5ms → EMA converges away from default
    for _ in 0..10 {
        client.on_request_start(&endpoint).await;
        client.on_request_complete(&endpoint, 5.0, 10).await;
    }
    client.finalize_layer_tpm().await;

    // Confirm stats have diverged from defaults
    {
        let stats = client.worker_stats.read().await;
        let s = &stats["w1:50051"];
        assert!(s.has_measured_rtt(), "RTT should be measured after requests");
        assert_ne!(s.get_avg_rtt(), DEFAULT_RTT);
    }

    // Reset
    client.reset_stats().await;

    // Verify stats are back to defaults
    {
        let stats = client.worker_stats.read().await;
        let s = &stats["w1:50051"];
        assert!(
            !s.has_measured_rtt(),
            "RTT should be back to default after reset"
        );
        assert_eq!(s.get_avg_rtt(), DEFAULT_RTT);
        assert_eq!(s.get_inflight(), 0);
        assert!(!s.has_measured_tpm());
    }
}

#[tokio::test]
async fn test_find_missing_experts() {
    let client = RoutingClient::new("localhost:5002".to_string());

    // Setup: only expert_1 exists
    {
        let mut table = client.routing_table.write().await;
        table.insert(
            "expert_1".to_string(),
            vec![WorkerEndpoint {
                grpc_addr: "worker1:50051".to_string(),
                channel: "grpc".to_string(),
                rdma_tcp_port: 0,
                shm_queue_prefix: "".to_string(),
                device: "cpu".to_string(),
                wm_addr: "".to_string(),
            }],
        );
    }

    // Check for missing experts
    let missing = client
        .find_missing_experts(&[
            "expert_1".to_string(),
            "expert_2".to_string(),
            "expert_3".to_string(),
        ])
        .await;

    assert_eq!(missing.len(), 2);
    assert!(missing.contains(&"expert_2".to_string()));
    assert!(missing.contains(&"expert_3".to_string()));
}
