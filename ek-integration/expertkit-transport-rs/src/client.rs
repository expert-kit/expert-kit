use anyhow::Result;
use log::{debug, error, info, warn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tonic::transport::Channel;

use crate::routing::RoutingClient;
use crate::transport::{ExpertRequest, Transport, auto::AutoTransport};
use crate::transport::grpc::proto::ek::worker::v1::{
    ForwardReq, forward_req::SequenceInfo, computation_service_client::ComputationServiceClient,
};
use crate::utils::{deserialize_safetensor_2_tch_tensor, serialize_tch_tensor_2_safetensor};

use tch::Tensor;

/// Result of a single expert task execution
enum ExpertTaskResult {
    Success(String, Tensor),
    Failed(String, String, Option<String>), // expert_id, error message, worker_addr
}

/// High-level client with worker-level batching and routing
/// Uses RTT-based adaptive load balancing for multi-replica scenarios
/// Supports controller fallback when direct worker requests fail
pub struct ExpertKitClient {
    controller_addr: String,
    routing: Arc<RoutingClient>,
    transport: Arc<AutoTransport>,
    /// Controller channel for fallback requests
    controller_channel: Option<Channel>,
    /// Timeout for controller fallback requests
    timeout: std::time::Duration,
}

impl ExpertKitClient {
    pub fn new(controller_addr: String, timeout_sec: f64) -> Self {
        Self {
            controller_addr: controller_addr.clone(),
            routing: Arc::new(RoutingClient::new(controller_addr)),
            transport: Arc::new(AutoTransport::new(timeout_sec)),
            controller_channel: None,
            timeout: std::time::Duration::from_secs_f64(timeout_sec),
        }
    }

    /// Connect to controller and fetch initial routing
    /// Also starts background routing refresh for progressive startup support
    pub async fn connect(&mut self) -> Result<()> {
        // Get mutable access to routing client for connection setup
        let routing = Arc::get_mut(&mut self.routing)
            .ok_or_else(|| anyhow::anyhow!("Cannot get mutable access to routing client"))?;
        routing.connect().await?;
        routing.fetch_routing(None).await?;

        // Establish controller channel for fallback requests
        let uri = if self.controller_addr.starts_with("http://") || self.controller_addr.starts_with("https://") {
            self.controller_addr.clone()
        } else {
            format!("http://{}", self.controller_addr)
        };
        self.controller_channel = Some(Channel::from_shared(uri)?.connect().await?);
        info!("[Client] Controller fallback channel established");

        // Warm up connections to all known workers
        let routing_table = self.routing.get_all_routing().await;
        let endpoints: Vec<_> = routing_table.values().cloned().collect();
        self.transport.warmup_connections(&endpoints).await?;

        // Start background routing refresh for progressive startup support
        // This enables the client to discover newly loaded experts as workers load them
        self.routing.start_background_refresh();
        info!("[Client] Background routing refresh started");

        Ok(())
    }

    /// Execute a single expert task with error handling.
    /// `pre_assigned_worker` is already inflight-incremented by the caller (assign_layer_batch
    /// or select_worker_and_mark_inflight); this function only decrements on completion.
    async fn execute_expert_task(
        expert_id: String,
        seq_positions: Vec<(usize, usize)>,
        hidden_state: Tensor,
        transport: Arc<AutoTransport>,
        routing: Arc<RoutingClient>,
        pre_assigned_worker: crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint,
        task_create_t: std::time::Instant,
    ) -> ExpertTaskResult {
        let sub_task_t = std::time::Instant::now();

        debug!(
            "[Client-Time] ⚡ Task SPAWNED for expert {} at {:?} μs from task_create_t",
            expert_id,
            task_create_t.elapsed().as_micros()
        );

        let worker_endpoint = pre_assigned_worker;

        // Tensor operations run directly
        let prep_start = std::time::Instant::now();

        // Extract sequence indices
        let seq_indices: Vec<i64> = seq_positions
            .iter()
            .map(|(seq_idx, _)| *seq_idx as i64)
            .collect();

        // Create index tensor and slice
        let index_tensor = Tensor::from_slice(&seq_indices);
        let expert_input = hidden_state.index_select(0, &index_tensor);

        // Serialize for network transfer
        let tensor_bytes = match serialize_tch_tensor_2_safetensor(&expert_input) {
            Ok(bytes) => bytes,
            Err(e) => {
                return ExpertTaskResult::Failed(
                    expert_id,
                    format!("Failed to serialize tensor: {}", e),
                    Some(worker_endpoint.grpc_addr.clone()),
                );
            }
        };
        let num_sequences = seq_indices.len();

        debug!(
            "[Client-Time] 🛸 BEFORE SEND for expert {} to worker {} with {} sequences, prep took {:?} μs, cost from task create time {:?} μs",
            expert_id,
            worker_endpoint.grpc_addr,
            num_sequences,
            prep_start.elapsed().as_micros(),
            task_create_t.elapsed().as_micros()
        );

        // Create request
        let request = ExpertRequest::new(expert_id.clone(), tensor_bytes, num_sequences);

        let send_t = std::time::Instant::now();
        let result = transport
            .send_batch(&worker_endpoint, vec![request])
            .await;
        let rtt_ms = send_t.elapsed().as_secs_f64() * 1000.0;

        // Track request completion (decrement inflight, update RTT and throughput)
        routing.on_request_complete(&worker_endpoint, rtt_ms, num_sequences).await;

        let addr = worker_endpoint.grpc_addr.clone();
        match result {
            Ok(responses) => {
                if let Some(response) = responses.into_iter().next() {
                    match deserialize_safetensor_2_tch_tensor(&response.tensor_data) {
                        Ok(tensor) => {
                            debug!(
                                "[Client-Time] 🔚 Sub-task for expert {} on worker {} completed in {:?} μs, send_batch took {:?} μs (RTT={:.2}ms), cost from task create time {:?} μs",
                                expert_id,
                                addr,
                                sub_task_t.elapsed().as_micros(),
                                send_t.elapsed().as_micros(),
                                rtt_ms,
                                task_create_t.elapsed().as_micros()
                            );
                            ExpertTaskResult::Success(expert_id, tensor)
                        }
                        Err(e) => ExpertTaskResult::Failed(
                            expert_id,
                            format!("Failed to deserialize response: {}", e),
                            Some(addr),
                        ),
                    }
                } else {
                    ExpertTaskResult::Failed(expert_id, "Empty response from worker".to_string(), Some(addr))
                }
            }
            Err(e) => {
                warn!(
                    "[Client] Expert {} failed on worker {}: {}",
                    expert_id, addr, e
                );
                ExpertTaskResult::Failed(expert_id, format!("Worker request failed: {}", e), Some(addr))
            }
        }
    }

    /// Forward expert computation with direct tensor access
    /// Handles worker failures gracefully with automatic fallback
    pub async fn forward_expert_tensor(
        &self,
        expert_ids: Vec<Vec<String>>, // [batch_size, n_routed_experts]
        hidden_state: Tensor,         // Direct tensor access (CPU or CUDA)
    ) -> Result<Tensor> {
        let batch_size = hidden_state.size()[0] as usize;
        let hidden_dim = hidden_state.size()[1] as usize;

        debug!(
            "[Client] forward_expert_tensor: batch_size={}, hidden_dim={}, device={:?}",
            batch_size,
            hidden_dim,
            hidden_state.device()
        );

        // Decompose by expert
        let mut expert_to_sequences: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

        for (seq_idx, experts) in expert_ids.iter().enumerate() {
            for (expert_idx, expert_id) in experts.iter().enumerate() {
                expert_to_sequences
                    .entry(expert_id.clone())
                    .or_default()
                    .push((seq_idx, expert_idx));
            }
        }

        info!(
            "[Client] Decomposed {} sequences into {} unique experts",
            expert_ids.len(),
            expert_to_sequences.len()
        );

        // Check for missing experts - if some are missing, refresh routing and retry
        let unique_experts: Vec<String> = expert_to_sequences.keys().cloned().collect();
        let missing = self.routing.find_missing_experts(&unique_experts).await;

        if !missing.is_empty() {
            warn!(
                "[Client] {} experts missing from routing, refreshing...",
                missing.len()
            );
            // Try to refresh routing
            if let Err(e) = self.routing.fetch_routing(None).await {
                warn!("[Client] Failed to refresh routing: {}", e);
            }

            // Check again after refresh
            let still_missing = self.routing.find_missing_experts(&unique_experts).await;
            if !still_missing.is_empty() {
                return Err(anyhow::anyhow!(
                    "Experts not found in routing table after refresh: {:?}",
                    still_missing
                ));
            }
        }

        // Track if any failure occurred for routing refresh
        let had_failure = Arc::new(AtomicBool::new(false));

        // Pre-assign workers for all experts using LPT to minimize layer makespan.
        // All routing decisions are made here before any task is spawned, allowing
        // us to balance the entire layer's workload across workers holistically.
        let expert_calls: Vec<(String, usize)> = expert_to_sequences
            .iter()
            .map(|(eid, positions)| (eid.clone(), positions.len()))
            .collect();
        let mut assignments = self.routing.assign_layer_batch(&expert_calls).await;

        // Build requests: slice tensor directly and spawn tasks immediately
        let mut jobs: tokio::task::JoinSet<ExpertTaskResult> = tokio::task::JoinSet::new();
        let mut expert_metadata: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

        let task_create_t = std::time::Instant::now();
        for (expert_id, seq_positions) in expert_to_sequences.iter() {
            // Store metadata for reconstruction and retry
            expert_metadata.insert(expert_id.clone(), seq_positions.clone());

            // Use pre-assigned worker (inflight already incremented by assign_layer_batch).
            // Fall back to dynamic selection if expert wasn't covered (shouldn't happen
            // after routing validation above, but keeps the code robust).
            let worker = match assignments.remove(expert_id) {
                Some(w) => w,
                None => {
                    warn!("[Client] Expert {} missing from LPT assignment, falling back to dynamic selection", expert_id);
                    match self.routing.select_worker_and_mark_inflight(expert_id).await {
                        Some(w) => w,
                        None => {
                            let eid = expert_id.clone();
                            jobs.spawn(async move {
                                ExpertTaskResult::Failed(eid, "No worker available".to_string(), None)
                            });
                            continue;
                        }
                    }
                }
            };

            let hidden_state_ref = hidden_state.shallow_clone();
            let expert_id_clone = expert_id.clone();
            let seq_positions_clone = seq_positions.clone();
            let transport = self.transport.clone();
            let routing = self.routing.clone();

            jobs.spawn(async move {
                Self::execute_expert_task(
                    expert_id_clone,
                    seq_positions_clone,
                    hidden_state_ref,
                    transport,
                    routing,
                    worker,
                    task_create_t,
                )
                .await
            });
        }

        // Execute all requests in parallel and collect results
        info!("[Client] Spawned {} parallel tasks", jobs.len());
        let t = std::time::Instant::now();

        let task_results: Vec<ExpertTaskResult> = jobs.join_all().await;

        // Update per-worker tpm using aggregate layer statistics (total tokens / max RTT).
        // Must happen after the layer barrier so all requests for this layer are included.
        self.routing.finalize_layer_tpm().await;

        // Separate successes and failures, collecting dead worker addresses
        let mut successful_results: HashMap<String, Tensor> = HashMap::new();
        let mut failed_experts: Vec<(String, String)> = Vec::new();
        let mut dead_addrs: HashSet<String> = HashSet::new();

        for result in task_results {
            match result {
                ExpertTaskResult::Success(expert_id, tensor) => {
                    successful_results.insert(expert_id, tensor);
                }
                ExpertTaskResult::Failed(expert_id, error, addr) => {
                    if let Some(a) = addr {
                        dead_addrs.insert(a);
                    }
                    failed_experts.push((expert_id, error));
                    had_failure.store(true, Ordering::Relaxed);
                }
            }
        }

        // If any expert tasks failed, return error immediately so the caller
        // (forward_expert_tensor_with_fallback) can fall back to the controller.
        //
        // The controller has retry logic with full state visibility — it knows
        // when recovered experts become available and can hold the request until
        // a worker loads the expert.  Retrying on the frontend is wasteful
        // because it re-selects from a potentially stale routing table.
        if !failed_experts.is_empty() {
            let failed_ids: Vec<String> = failed_experts
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            warn!(
                "[Client] {} expert tasks failed on direct path, deferring to controller fallback: {:?}",
                failed_ids.len(),
                &failed_ids[..failed_ids.len().min(5)],
            );
            return Err(anyhow::anyhow!(
                "Direct path failed for {} experts",
                failed_ids.len()
            ));
        }

        debug!(
            "[Client-Time] 🚀 All expert tasks completed in {:?} μs",
            t.elapsed().as_micros()
        );

        // Reconstruct output: place expert outputs back in original positions
        let t = std::time::Instant::now();
        let n_experts_per_seq = expert_ids[0].len();
        let mut output_tensors: Vec<Vec<Option<Tensor>>> = Vec::new();
        for _ in 0..batch_size {
            let mut row = Vec::new();
            for _ in 0..n_experts_per_seq {
                row.push(None);
            }
            output_tensors.push(row);
        }
        debug!(
            "[Client-Time] 🧩 Starting reconstruction of output tensors in {:?} μs",
            t.elapsed().as_micros()
        );

        let t = std::time::Instant::now();
        for (expert_id_resp, resp_tensor) in successful_results.iter() {
            let seq_positions = expert_metadata
                .get(expert_id_resp)
                .ok_or_else(|| anyhow::anyhow!("Missing metadata for expert {}", expert_id_resp))?;

            // Place each sequence's output in the correct position
            for (output_idx, (seq_idx, expert_pos)) in seq_positions.iter().enumerate() {
                let seq_output = resp_tensor.get(output_idx as i64);
                output_tensors[*seq_idx][*expert_pos] = Some(seq_output);
            }
        }
        debug!(
            "[Client-Time] 🧩 Completed reconstruction of output tensors in {:?} μs",
            t.elapsed().as_micros()
        );

        // Stack tensors to create final output [batch_size, n_experts, expert_dim]
        let mut final_output_rows = Vec::new();

        let t = std::time::Instant::now();
        for seq_outputs in output_tensors {
            let outputs: Vec<Tensor> = seq_outputs
                .into_iter()
                .map(|opt| opt.ok_or_else(|| anyhow::anyhow!("Missing expert output")))
                .collect::<Result<Vec<_>>>()?;

            // Stack along expert dimension
            let seq_output = Tensor::stack(&outputs, 0);
            final_output_rows.push(seq_output);
        }

        // Stack all sequences
        let final_output = Tensor::stack(&final_output_rows, 0);

        debug!(
            "[Client] Reconstructed output shape: {:?}, device: {:?}",
            final_output.size(),
            final_output.device()
        );

        debug!(
            "[Client-Time] 🧩 Completed stacking of final output tensors in {:?} μs",
            t.elapsed().as_micros()
        );

        Ok(final_output)
    }

    /// Reset all worker stats (RTT EMA, throughput EMA, inflight counters).
    /// Call between benchmark iterations to prevent earlier runs from biasing later ones.
    pub async fn reset_stats(&self) {
        self.routing.reset_stats().await;
    }

    /// Refresh routing table
    pub async fn refresh_routing(&self) -> Result<()> {
        self.routing.fetch_routing(None).await?;

        // Warm up connections to any new workers
        let routing_table = self.routing.get_all_routing().await;
        let endpoints: Vec<_> = routing_table.values().cloned().collect();
        self.transport.warmup_connections(&endpoints).await?;

        Ok(())
    }

    /// Get current worker stats for monitoring/debugging
    #[allow(unused)]
    pub async fn get_worker_stats(&self) -> HashMap<String, (f64, u32)> {
        // Returns (avg_rtt_ms, inflight_count) per worker
        let all_workers = self.routing.get_all_workers().await;
        let mut stats = HashMap::new();

        for endpoints in all_workers.values() {
            for endpoint in endpoints {
                // Note: we can't directly access internal stats, but this method
                // could be extended if needed for monitoring
                stats.entry(endpoint.grpc_addr.clone()).or_insert((0.0, 0));
            }
        }

        stats
    }

    /// Forward request via controller fallback path
    /// Used when direct worker requests fail
    async fn forward_via_controller(
        &self,
        expert_ids: &[Vec<String>],
        hidden_state: &Tensor,
    ) -> Result<Tensor> {
        let channel = self.controller_channel.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Controller channel not established"))?;

        // Serialize the hidden state tensor
        let tensor_bytes = serialize_tch_tensor_2_safetensor(hidden_state)?;

        // Build sequences info for the request
        let sequences: Vec<SequenceInfo> = expert_ids
            .iter()
            .map(|experts| SequenceInfo {
                experts: experts.clone(),
            })
            .collect();

        let req = ForwardReq {
            instance_id: "0".to_string(),
            tensor: tensor_bytes,
            sequences,
        };

        info!(
            "[Client] Forwarding via controller fallback: {} sequences",
            expert_ids.len()
        );

        // Create client with large message size for tensors
        let max_message_size = 1024 * 1024 * 1024; // 1 GB
        let mut client = ComputationServiceClient::new(channel.clone())
            .max_decoding_message_size(max_message_size)
            .max_encoding_message_size(max_message_size);

        // Send with timeout
        let response = tokio::time::timeout(self.timeout, client.forward(req))
            .await
            .map_err(|_| anyhow::anyhow!("Controller fallback timeout after {:?}", self.timeout))?
            .map_err(|e| anyhow::anyhow!("Controller fallback gRPC error: {}", e))?;

        let output_tensor = response.into_inner().output_tensor;
        let result = deserialize_safetensor_2_tch_tensor(&output_tensor)?;

        info!("[Client] Controller fallback completed successfully");

        Ok(result)
    }

    /// Forward expert computation with fallback to controller
    /// First tries direct worker path (with retry), falls back to controller on complete failure
    pub async fn forward_expert_tensor_with_fallback(
        &self,
        expert_ids: Vec<Vec<String>>,
        hidden_state: Tensor,
    ) -> Result<Tensor> {
        // Try direct worker path first (it already has retry logic)
        match self.forward_expert_tensor(expert_ids.clone(), hidden_state.shallow_clone()).await {
            Ok(result) => Ok(result),
            Err(e) => {
                warn!(
                    "[Client] Direct worker path failed: {}, refreshing routing and falling back to controller",
                    e
                );

                // Refresh routing from controller — the direct-path failure
                // may be due to stale entries pointing to a dead/restarted
                // worker.  The controller's routing table is authoritative
                // and only includes "loaded" experts.
                if let Err(re) = self.routing.fetch_routing(None).await {
                    warn!("[Client] Routing refresh failed: {}", re);
                }

                // Fallback to controller as last resort
                match self.forward_via_controller(&expert_ids, &hidden_state).await {
                    Ok(result) => {
                        info!("[Client] Controller fallback succeeded");
                        Ok(result)
                    }
                    Err(fallback_err) => {
                        error!(
                            "[Client] Controller fallback also failed: {}",
                            fallback_err
                        );
                        Err(anyhow::anyhow!(
                            "All paths failed. Direct: {}. Controller: {}",
                            e,
                            fallback_err
                        ))
                    }
                }
            }
        }
    }
}
