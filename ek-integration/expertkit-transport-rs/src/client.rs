use anyhow::Result;
use std::collections::HashMap;

use crate::routing::RoutingClient;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;
use crate::transport::{ExpertRequest, ExpertResponse, Transport, auto::AutoTransport};

use ndarray::{Array2, Array3, Axis};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

/// High-level client with worker-level batching and routing
pub struct ExpertKitClient {
    routing: RoutingClient,
    transport: AutoTransport,
}

impl ExpertKitClient {
    pub fn new(controller_addr: String, timeout_sec: f64) -> Self {
        Self {
            routing: RoutingClient::new(controller_addr),
            transport: AutoTransport::new(timeout_sec),
        }
    }

    /// Connect to controller and fetch initial routing
    pub async fn connect(&mut self) -> Result<()> {
        self.routing.connect().await?;
        self.routing.fetch_routing(None).await?;
        Ok(())
    }

    /// Send expert requests with automatic batching by expert
    pub async fn send_expert_batch(
        &self,
        expert_ids: Vec<String>,
        tensor_data: Vec<Vec<u8>>,
    ) -> Result<Vec<Vec<u8>>> {
        if expert_ids.len() != tensor_data.len() {
            return Err(anyhow::anyhow!(
                "expert_ids and tensor_data length mismatch: {} vs {}",
                expert_ids.len(),
                tensor_data.len()
            ));
        }

        // Look up workers for all experts
        let workers = self.routing.get_workers(&expert_ids).await;

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

        // Group by expert (expert_id -> [(worker_endpoint, tensor_data, original_idx)])
        let mut expert_batches: HashMap<String, (WorkerEndpoint, Vec<(Vec<u8>, usize)>)> =
            HashMap::new();

        for (idx, expert_id) in expert_ids.iter().enumerate() {
            let worker_endpoint = workers
                .get(expert_id)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Worker not found for expert {}", expert_id))?
                .clone();

            expert_batches
                .entry(expert_id.clone())
                .or_insert_with(|| (worker_endpoint, Vec::new()))
                .1
                .push((tensor_data[idx].clone(), idx));
        }

        eprintln!(
            "[Client] Grouped {} sequences into {} expert batches",
            expert_ids.len(),
            expert_batches.len()
        );

        // Send one request per expert (in parallel)
        let mut tasks = Vec::new();

        for (expert_id, (worker_endpoint, batch)) in expert_batches {
            let transport = &self.transport;

            // Combine all sequence tensors into ONE batched safetensors blob
            let batched_tensor = if batch.len() == 1 {
                // Single sequence - use as-is
                batch[0].0.clone()
            } else {
                // Multiple sequences - need to stack them
                // TODO: Implement proper safetensors stacking
                // For now, this is a limitation - we can only handle single sequences per expert
                eprintln!(
                    "[Warning] Expert {} has {} sequences - stacking not yet implemented, using first only",
                    expert_id,
                    batch.len()
                );
                batch[0].0.clone()
            };

            let request = ExpertRequest::new(expert_id.clone(), batched_tensor, batch.len());

            // Create async task
            let task = async move {
                let responses = transport
                    .send_batch(&worker_endpoint, vec![request])
                    .await?;
                Ok::<(String, ExpertResponse, Vec<usize>), anyhow::Error>((
                    expert_id,
                    responses.into_iter().next().unwrap(),
                    batch.iter().map(|(_, idx)| *idx).collect(),
                ))
            };

            tasks.push(task);
        }

        // Wait for all experts to respond
        let results = futures::future::try_join_all(tasks).await?;

        // Reconstruct responses in original order
        let mut output = vec![Vec::new(); expert_ids.len()];

        for (_expert_id, response, indices) in results {
            // For now, since we don't stack properly, we just replicate the response
            // TODO: Properly split the batched response
            for orig_idx in indices {
                output[orig_idx] = response.tensor_data.clone();
            }
        }

        Ok(output)
    }

    /// Forward expert computation with automatic decomposition and reconstruction
    pub async fn forward_expert(
        &self,
        expert_ids: Vec<Vec<String>>, // [batch_size, n_routed_experts]
        hidden_state_bytes: Vec<u8>,  // Serialized safetensors
    ) -> Result<Vec<u8>> {
        // Deserialize the input tensor once
        let tensors = SafeTensors::deserialize(&hidden_state_bytes)?;
        let tensor_view = tensors.tensor("data")?;

        // Convert to ndarray for slicing
        let shape = tensor_view.shape();
        if shape.len() != 2 {
            return Err(anyhow::anyhow!("Expected 2D tensor, got shape {:?}", shape));
        }

        let batch_size = shape[0];
        let hidden_dim = shape[1];

        // Parse tensor data based on dtype
        let hidden_state: Array2<f32> = match tensor_view.dtype() {
            Dtype::F32 => {
                let data: &[f32] = tensor_view
                    .data()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<_>>()
                    .leak();
                Array2::from_shape_vec((batch_size, hidden_dim), data.to_vec())?
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported dtype: {:?}",
                    tensor_view.dtype()
                ));
            }
        };

        // Decompose by expert (figure out which sequences go to which expert)
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

        // Build requests: slice tensor and serialize per expert
        let mut tasks = Vec::new();
        let mut expert_metadata = Vec::new();

        for (expert_id, seq_positions) in expert_to_sequences.iter() {
            let worker_endpoint = workers
                .get(expert_id)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Worker not found for expert {}", expert_id))?
                .clone();

            // Extract sequence indices
            let seq_indices: Vec<usize> =
                seq_positions.iter().map(|(seq_idx, _)| *seq_idx).collect();

            // Slice the tensor to get inputs for this expert
            let expert_input = hidden_state.select(Axis(0), &seq_indices);

            // Serialize to safetensors
            let tensor_bytes = serialize_tensor(&expert_input)?;

            // Create request
            let request = ExpertRequest::new(expert_id.clone(), tensor_bytes, seq_indices.len());

            // Store metadata for reconstruction
            expert_metadata.push((expert_id.clone(), seq_positions.clone()));

            // Create async task
            let transport = &self.transport;
            let task = async move {
                let responses = transport
                    .send_batch(&worker_endpoint, vec![request])
                    .await?;
                Ok::<(String, ExpertResponse), anyhow::Error>((
                    expert_id.clone(),
                    responses.into_iter().next().unwrap(),
                ))
            };

            tasks.push(task);
        }

        // Execute all requests in parallel
        eprintln!("[Client] Sending {} parallel requests", tasks.len());
        let results = futures::future::try_join_all(tasks).await?;

        // Reconstruct output: place expert outputs back in original positions
        let n_experts_per_seq = expert_ids[0].len();
        let mut output_tensors: Vec<Vec<Option<Array2<f32>>>> =
            vec![vec![None; n_experts_per_seq]; batch_size];

        for ((expert_id, seq_positions), (_expert_id_resp, response)) in
            expert_metadata.iter().zip(results.iter())
        {
            // Deserialize response
            let response_tensors = SafeTensors::deserialize(&response.tensor_data)?;
            let response_view = response_tensors.tensor("data")?;

            let response_shape = response_view.shape();
            if response_shape.len() != 2 {
                return Err(anyhow::anyhow!(
                    "Expected 2D response tensor, got shape {:?}",
                    response_shape
                ));
            }

            let n_seqs = response_shape[0];
            let expert_dim = response_shape[1];

            // Parse response tensor
            let response_data: Array2<f32> = match response_view.dtype() {
                Dtype::F32 => {
                    let data: &[f32] = response_view
                        .data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect::<Vec<_>>()
                        .leak();
                    Array2::from_shape_vec((n_seqs, expert_dim), data.to_vec())?
                }
                _ => return Err(anyhow::anyhow!("Unsupported response dtype")),
            };

            // Place each sequence's output in the correct position
            for (output_idx, (seq_idx, expert_pos)) in seq_positions.iter().enumerate() {
                let expert_output = response_data
                    .slice(ndarray::s![output_idx..output_idx + 1, ..])
                    .to_owned();
                output_tensors[*seq_idx][*expert_pos] = Some(expert_output);
            }
        }

        // Stack tensors and serialize once
        let mut final_output_rows = Vec::new();

        for seq_outputs in output_tensors {
            let outputs: Vec<Array2<f32>> = seq_outputs
                .into_iter()
                .map(|opt| opt.ok_or_else(|| anyhow::anyhow!("Missing expert output")))
                .collect::<Result<Vec<_>>>()?;

            // Concatenate along expert dimension
            let seq_output = ndarray::concatenate(
                Axis(0),
                &outputs.iter().map(|a| a.view()).collect::<Vec<_>>(),
            )?;

            final_output_rows.push(seq_output);
        }

        // Stack all sequences
        let final_output = ndarray::stack(
            Axis(0),
            &final_output_rows
                .iter()
                .map(|a| a.view())
                .collect::<Vec<_>>(),
        )?;

        eprintln!(
            "[Client] Reconstructed output shape: {:?}",
            final_output.shape()
        );

        // Serialize final output
        serialize_tensor_3d(&final_output)
    }

    /// Refresh routing table
    pub async fn refresh_routing(&self) -> Result<()> {
        self.routing.fetch_routing(None).await
    }
}

/// Helper function to serialize 2D ndarray tensor to safetensors format
fn serialize_tensor(tensor: &Array2<f32>) -> Result<Vec<u8>> {
    let shape = tensor.shape();
    let data: Vec<u8> = tensor.iter().flat_map(|&f| f.to_le_bytes()).collect();

    // Create safetensors format
    let mut tensors = HashMap::new();
    tensors.insert(
        "data".to_string(),
        TensorView::new(Dtype::F32, vec![shape[0], shape[1]], &data)?,
    );

    safetensors::serialize(&tensors, &None)
        .map_err(|e| anyhow::anyhow!("Failed to serialize tensor: {}", e))
}

/// Helper function to serialize 3D ndarray tensor to safetensors format
fn serialize_tensor_3d(tensor: &Array3<f32>) -> Result<Vec<u8>> {
    let shape = tensor.shape();
    let data: Vec<u8> = tensor.iter().flat_map(|&f| f.to_le_bytes()).collect();

    // Create safetensors format
    let mut tensors = HashMap::new();
    tensors.insert(
        "data".to_string(),
        TensorView::new(Dtype::F32, vec![shape[0], shape[1], shape[2]], &data)?,
    );

    safetensors::serialize(&tensors, &None)
        .map_err(|e| anyhow::anyhow!("Failed to serialize tensor: {}", e))
}
