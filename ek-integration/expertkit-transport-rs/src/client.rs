use anyhow::Result;
use log::{debug, info};
use std::collections::HashMap;
use std::sync::Arc;

use crate::routing::RoutingClient;
use crate::transport::{ExpertRequest, ExpertResponse, Transport, auto::AutoTransport};
use crate::utils::{deserialize_safetensor_2_tch_tensor, serialize_tch_tensor_2_safetensor};

use tch::Tensor;

/// High-level client with worker-level batching and routing
pub struct ExpertKitClient {
    routing: RoutingClient,
    transport: Arc<AutoTransport>,
}

impl ExpertKitClient {
    pub fn new(controller_addr: String, timeout_sec: f64) -> Self {
        Self {
            routing: RoutingClient::new(controller_addr),
            transport: Arc::new(AutoTransport::new(timeout_sec)),
        }
    }

    /// Connect to controller and fetch initial routing
    pub async fn connect(&mut self) -> Result<()> {
        self.routing.connect().await?;
        self.routing.fetch_routing(None).await?;

        // Warm up connections to all known workers
        let routing_table = self.routing.get_all_routing().await;
        let endpoints: Vec<_> = routing_table.values().cloned().collect();
        self.transport.warmup_connections(&endpoints).await?;

        Ok(())
    }

    /// Forward expert computation with direct tensor access
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

        // Look up workers for all experts
        let unique_experts: Vec<String> = expert_to_sequences.keys().cloned().collect();
        let workers = self.routing.get_workers(&unique_experts).await;

        // Check for missing experts
        let missing: Vec<_> = workers
            .iter()
            .filter_map(|(id, worker)| if worker.is_none() { Some(id) } else { None })
            .collect();

        if !missing.is_empty() {
            return Err(anyhow::anyhow!(
                "Experts not found in routing table: {:?}",
                missing
            ));
        }

        // Build requests: slice tensor directly and spawn tasks immediately
        let mut jobs: tokio::task::JoinSet<Result<(String, Tensor)>> = tokio::task::JoinSet::new();
        let mut expert_metadata: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

        let task_create_t = std::time::Instant::now();
        for (expert_id, seq_positions) in expert_to_sequences.iter() {
            let worker_endpoint = workers
                .get(expert_id)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Worker not found for expert {}", expert_id))?
                .clone();

            // Store metadata for reconstruction
            expert_metadata.insert(expert_id.clone(), seq_positions.clone());

            let hidden_state_ref = hidden_state.shallow_clone();
            let expert_id_clone = expert_id.clone();
            let seq_positions_clone = seq_positions.clone();
            let transport = self.transport.clone();

            jobs.spawn(async move {
                let sub_task_t = std::time::Instant::now();

                debug!(
                    "[Client-Time] ⚡ Task SPAWNED for expert {} at {:?} μs from task_create_t",
                    expert_id_clone,
                    task_create_t.elapsed().as_micros()
                );

                // Tensor operations run directly
                let prep_start = std::time::Instant::now();

                // Extract sequence indices
                let seq_indices: Vec<i64> = seq_positions_clone
                    .iter()
                    .map(|(seq_idx, _)| *seq_idx as i64)
                    .collect();

                // Create index tensor and slice
                let index_tensor = Tensor::from_slice(&seq_indices);
                let expert_input = hidden_state_ref.index_select(0, &index_tensor);

                // Serialize for network transfer
                let tensor_bytes = serialize_tch_tensor_2_safetensor(&expert_input)?;
                let num_sequences = seq_indices.len();

                debug!(
                    "[Client-Time] 🛸 BEFORE SEND for expert {} with {} sequences, prep took {:?} μs, cost from task create time {:?} μs",
                    expert_id_clone,
                    num_sequences,
                    prep_start.elapsed().as_micros(),
                    task_create_t.elapsed().as_micros()
                );

                // Create request
                let request =
                    ExpertRequest::new(expert_id_clone.clone(), tensor_bytes, num_sequences);

                let send_t = std::time::Instant::now();
                let responses = transport
                    .send_batch(&worker_endpoint, vec![request])
                    .await?;

                debug!(
                    "[Client-Time] 🔚 Sub-task for expert {} completed in {:?} μs, send_batch took {:?} μs, cost from task create time {:?} μs",
                    expert_id_clone,
                    sub_task_t.elapsed().as_micros(),
                    send_t.elapsed().as_micros(),
                    task_create_t.elapsed().as_micros()
                );

                Ok((
                    expert_id_clone,
                    deserialize_safetensor_2_tch_tensor(&responses.into_iter().next().unwrap().tensor_data).unwrap(),
                ))
            });
        }

        // Execute all requests in parallel and collect results
        info!("[Client] Spawned {} parallel tasks", jobs.len());
        let t = std::time::Instant::now();
        let results: Vec<(String, Tensor)> = jobs
            .join_all()
            .await
            .into_iter()
            .map(|res: Result<(String, Tensor), _>| res.unwrap())
            .collect();
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
        for (expert_id_resp, resp_tensor) in results.iter() {
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

    /// Refresh routing table
    pub async fn refresh_routing(&self) -> Result<()> {
        self.routing.fetch_routing(None).await?;

        // Warm up connections to any new workers
        let routing_table = self.routing.get_all_routing().await;
        let endpoints: Vec<_> = routing_table.values().cloned().collect();
        self.transport.warmup_connections(&endpoints).await?;

        Ok(())
    }
}
