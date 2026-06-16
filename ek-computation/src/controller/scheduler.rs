use std::sync::Arc;

use crate::{
    controller::elastic::progressive,
    state::{
        io::{StateReader, StateReaderImpl},
        models::Node,
    },
};
use ek_base::{config::get_ek_settings, error::{EKError, EKResult}};

use super::load_tracker::LoadTracker;

/// Estimated size of one expert in MB (3 BF16 weight matrices: up/gate/down).
pub fn expert_size_mb() -> u64 {
    let s = get_ek_settings();
    let bytes = 3usize * s.inference.hidden_dim * s.inference.intermediate_dim * 2;
    (bytes / (1024 * 1024)).max(1) as u64
}

/// Remaining memory capacity for a node: total - (assigned_experts × expert_size_mb).
pub async fn remaining_capacity_mb(node: &Node, reader: &StateReaderImpl) -> u64 {
    let total = node
        .config
        .get("mem_capacity_mb")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096);
    let assigned = reader
        .experts_by_node(node.id)
        .await
        .map(|e| e.len() as u64)
        .unwrap_or(0);
    total.saturating_sub(assigned * expert_size_mb())
}

/// Select the best available worker to host a new replica of `expert_id`.
///
/// Excludes the dead/preempted node (`exclude_hostname`), any node already
/// hosting the expert, and any node currently under progressive loading.
/// Among remaining active nodes, picks the one with the most remaining capacity.
pub async fn select_worker_for_new_replica(
    expert_id: &str,
    exclude_hostname: &str,
) -> EKResult<Node> {
    let reader = StateReaderImpl::new();

    // Nodes already hosting this expert (DB, regardless of state)
    let existing = reader.node_by_expert(expert_id).await.unwrap_or_default();
    let existing_hostnames: std::collections::HashSet<&str> =
        existing.iter().map(|n| n.hostname.as_str()).collect();

    // All nodes with a recent heartbeat, filtered by exclusions
    let mut candidates: Vec<Node> = Vec::new();
    for n in reader.active_nodes().await? {
        if n.hostname == exclude_hostname { continue; }
        if existing_hostnames.contains(n.hostname.as_str()) { continue; }
        if progressive::is_progressive_loading(&n.hostname).await { continue; }
        candidates.push(n);
    }

    if candidates.is_empty() {
        return Err(EKError::NotFound(format!(
            "no recovery target for expert {expert_id}"
        )));
    }

    // Prefer the node with the most remaining capacity
    let mut with_remaining: Vec<(Node, u64)> = Vec::new();
    for n in candidates {
        let rem = remaining_capacity_mb(&n, &reader).await;
        with_remaining.push((n, rem));
    }
    with_remaining.sort_by(|a, b| b.1.cmp(&a.1));

    Ok(with_remaining.into_iter().next().map(|(n, _)| n).unwrap())
}

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
