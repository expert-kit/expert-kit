use anyhow::Result;
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

        eprintln!(
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

        eprintln!(
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
        let mut handles = Vec::new();
        let mut expert_metadata = Vec::new();

        let task_create_t = std::time::Instant::now();
        for (expert_id, seq_positions) in expert_to_sequences.iter() {
            let worker_endpoint = workers
                .get(expert_id)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Worker not found for expert {}", expert_id))?
                .clone();

            // Store metadata for reconstruction
            expert_metadata.push((expert_id.clone(), seq_positions.clone()));

            let hidden_state_ref = hidden_state.shallow_clone();
            let expert_id_clone = expert_id.clone();
            let seq_positions_clone = seq_positions.clone();
            let transport = Arc::clone(&self.transport);

            let handle = tokio::spawn(async move {
                let sub_task_t = std::time::Instant::now();

                eprintln!(
                    "[Client-Time] ⚡ Task SPAWNED for expert {} at {:?} μs from task_create_t",
                    expert_id_clone,
                    task_create_t.elapsed().as_micros()
                );

                // Tensor operations run directly (they're already parallelized by being in separate tasks)
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

                eprintln!(
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

                eprintln!(
                    "[Client-Time] 🔚 Sub-task for expert {} completed in {:?} μs, send_batch took {:?} μs, cost from task create time {:?} μs",
                    expert_id_clone,
                    sub_task_t.elapsed().as_micros(),
                    send_t.elapsed().as_micros(),
                    task_create_t.elapsed().as_micros()
                );

                Ok::<(String, ExpertResponse), anyhow::Error>((
                    expert_id_clone,
                    responses.into_iter().next().unwrap(),
                ))
            });

            handles.push(handle);
        }

        // Execute all requests in parallel and collect results
        eprintln!("[Client] Spawned {} parallel tasks", handles.len());
        let results = futures::future::try_join_all(handles.into_iter().map(|h| async move {
            h.await
                .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
        }))
        .await?;

        // Reconstruct output: place expert outputs back in original positions
        let n_experts_per_seq = expert_ids[0].len();
        let mut output_tensors: Vec<Vec<Option<Tensor>>> = Vec::new();
        for _ in 0..batch_size {
            let mut row = Vec::new();
            for _ in 0..n_experts_per_seq {
                row.push(None);
            }
            output_tensors.push(row);
        }

        for ((expert_id, seq_positions), (_expert_id_resp, response)) in
            expert_metadata.iter().zip(results.iter())
        {
            // Deserialize response to tensor
            let expert_output = deserialize_safetensor_2_tch_tensor(&response.tensor_data)?;

            // Place each sequence's output in the correct position
            for (output_idx, (seq_idx, expert_pos)) in seq_positions.iter().enumerate() {
                let seq_output = expert_output.get(output_idx as i64);
                output_tensors[*seq_idx][*expert_pos] = Some(seq_output);
            }
        }

        // Stack tensors to create final output [batch_size, n_experts, expert_dim]
        let mut final_output_rows = Vec::new();

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

        eprintln!(
            "[Client] Reconstructed output shape: {:?}, device: {:?}",
            final_output.size(),
            final_output.device()
        );

        Ok(final_output)
    }

    /// Refresh routing table
    pub async fn refresh_routing(&self) -> Result<()> {
        self.routing.fetch_routing(None).await
    }
}
