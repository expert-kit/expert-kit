use crate::{
    controller::elastic::progressive,
    state::{
        io::{StateReader, StateReaderImpl},
        models::Node,
    },
};
use ek_base::error::{EKError, EKResult};

/// Return the capacity derived and reported by the Python Worker.
pub fn max_experts(node: &Node) -> u64 {
    node.config
        .get("max_experts")
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
}

/// Return unassigned expert slots, including loading assignments as occupied.
pub async fn remaining_expert_capacity(node: &Node, reader: &StateReaderImpl) -> u64 {
    let assigned = reader
        .experts_by_node(node.id)
        .await
        .map(|e| e.len() as u64)
        .unwrap_or(0);
    remaining_slots(max_experts(node), assigned)
}

fn remaining_slots(maximum: u64, assigned: u64) -> u64 {
    maximum.saturating_sub(assigned)
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
        if n.hostname == exclude_hostname {
            continue;
        }
        if existing_hostnames.contains(n.hostname.as_str()) {
            continue;
        }
        if progressive::is_progressive_loading(&n.hostname).await {
            continue;
        }
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
        let remaining = remaining_expert_capacity(&n, &reader).await;
        if remaining > 0 {
            with_remaining.push((n, remaining));
        }
    }
    with_remaining.sort_by(|a, b| b.1.cmp(&a.1));

    with_remaining
        .into_iter()
        .next()
        .map(|(node, _remaining)| node)
        .ok_or_else(|| EKError::NotFound(format!("no worker has capacity for expert {expert_id}")))
}

#[cfg(test)]
mod capacity_tests {
    use super::remaining_slots;

    #[test]
    fn assigned_experts_consume_one_reported_slot_each() {
        assert_eq!(remaining_slots(10, 3), 7);
        assert_eq!(remaining_slots(10, 10), 0);
        assert_eq!(remaining_slots(10, 11), 0);
    }
}
