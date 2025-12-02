use anyhow::Result;
use std::collections::HashMap;

use crate::routing::RoutingClient;
use crate::transport::{ExpertRequest, ExpertResponse, Transport, auto::AutoTransport};

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

        // Group by expert (expert_id -> [(tensor_data, original_idx)])
        let mut expert_batches: HashMap<String, (String, Vec<(Vec<u8>, usize)>)> = HashMap::new();

        for (idx, expert_id) in expert_ids.iter().enumerate() {
            let worker_addr = workers
                .get(expert_id)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| anyhow::anyhow!("Worker not found for expert {}", expert_id))?
                .clone();

            expert_batches
                .entry(expert_id.clone())
                .or_insert_with(|| (worker_addr, Vec::new()))
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

        for (expert_id, (worker_addr, batch)) in expert_batches {
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
                let responses = transport.send_batch(&worker_addr, vec![request]).await?;
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

    /// Refresh routing table
    pub async fn refresh_routing(&self) -> Result<()> {
        self.routing.fetch_routing(None).await
    }

    /// Get routing table version
    pub async fn get_routing_version(&self) -> u64 {
        self.routing.get_version().await
    }
}
