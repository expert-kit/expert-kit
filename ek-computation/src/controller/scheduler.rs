use std::sync::Arc;

use crate::state::{io::StateReader, models::Node};
use ek_base::error::{EKError, EKResult};

use super::load_tracker::LoadTracker;

/// WorkerScheduler selects the best worker for each expert based on device tier and load
/// This implements the controller-side scheduling logic (hidden from frontends)
pub struct WorkerScheduler {
    state_reader: Arc<dyn StateReader + Send + Sync>,
    load_tracker: Arc<LoadTracker>,
}

impl WorkerScheduler {
    pub fn new(
        state_reader: Arc<dyn StateReader + Send + Sync>,
        load_tracker: Arc<LoadTracker>,
    ) -> Self {
        Self {
            state_reader,
            load_tracker,
        }
    }

    /// Select the best worker for a given expert
    /// Returns worker address in format "host:port"
    pub async fn select_worker_for_expert(&self, expert_id: &str) -> EKResult<String> {
        // Get all nodes hosting this expert
        let nodes = self.state_reader.node_by_expert(expert_id).await?;

        if nodes.is_empty() {
            return Err(EKError::ExpertNotFound(expert_id.to_string()));
        }

        // If only one node, return it immediately
        if nodes.len() == 1 {
            return Ok(self.node_to_addr(&nodes[0]));
        }

        // Score each replica based on device tier, load, etc.
        let selected = nodes
            .into_iter()
            .map(|n| {
                let score = self.score_node(&n);
                (n, score)
            })
            .max_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(n, _)| n)
            .ok_or_else(|| EKError::ExpertNotFound(expert_id.to_string()))?;

        Ok(self.node_to_addr(&selected))
    }

    /// Select workers for multiple experts in batch
    /// Returns map of expert_id → worker_addr
    pub async fn select_workers_batch(
        &self,
        expert_ids: &[String],
    ) -> EKResult<std::collections::HashMap<String, String>> {
        let mut result = std::collections::HashMap::new();

        for expert_id in expert_ids {
            match self.select_worker_for_expert(expert_id).await {
                Ok(addr) => {
                    result.insert(expert_id.clone(), addr);
                }
                Err(e) => {
                    log::warn!("Failed to select worker for expert {}: {:?}", expert_id, e);
                    // Continue with other experts even if one fails
                }
            }
        }

        Ok(result)
    }

    /// Score a node based on device tier and load
    /// Higher score = better choice
    fn score_node(&self, node: &Node) -> f64 {
        // Device tier scoring (tier 1 = 100, tier 5 = 20)
        // Default to tier 3 if not specified
        let tier = node
            .config
            .get("device")
            .and_then(|d| d.get("tier"))
            .and_then(|t| t.as_i64())
            .unwrap_or(3);

        let tier_score = (6 - tier.clamp(1, 5)) as f64 * 20.0;

        // Load penalty (higher load = lower score)
        let load = self.load_tracker.get_load(&node.hostname) as f64;
        let load_penalty = load * 2.0;

        // Total score
        let score = tier_score - load_penalty;

        log::debug!(
            "Node {} (tier={}, load={}) score: {}",
            node.hostname,
            tier,
            load,
            score
        );

        score
    }

    /// Extract worker address from node config
    fn node_to_addr(&self, node: &Node) -> String {
        node.config
            .get("addr")
            .and_then(|a| a.as_str())
            .unwrap_or("unknown")
            .to_string()
    }
}
