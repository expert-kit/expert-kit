use std::{
    collections::{HashMap, HashSet},
    time::Instant,
};

use crate::{
    controller::{
        dispatcher::DISPATCHER,
        elastic::frequency::get_freq_tracker,
        scheduler::remaining_expert_capacity,
    },
    state::{
        io::{StateReader, StateReaderImpl},
        models::{Expert, NewExpert, Node},
        writer::StateWriterImpl,
    },
};
use ek_base::config::get_ek_settings;

/// Identify experts that were ONLY hosted on `dead_hostname` and reassign them
/// to available surviving workers using batch operations.
///
/// Called as a background task immediately after a node is deactivated, for both
/// abrupt failure (kill -9, heartbeat timeout) and graceful preemption (SIGTERM).
///
/// Performance: uses batch DB operations to reduce recovery time from O(n) DB
/// round-trips to O(1), enabling recovery of thousands of experts in seconds
/// instead of minutes.
pub async fn recover_unique_experts(dead_hostname: &str) {
    let recovery_start = Instant::now();
    let settings = get_ek_settings();
    let reader = StateReaderImpl::new();

    // 1. Look up the dead node
    let dead_node = match reader.node_by_hostname(dead_hostname).await {
        Ok(Some(n)) => n,
        Ok(None) => {
            log::warn!("recover: node {dead_hostname} not found in DB");
            return;
        }
        Err(e) => {
            log::error!("recover: DB error looking up {dead_hostname}: {e}");
            return;
        }
    };

    // 2. All experts assigned to the dead node
    let dead_experts = match reader.experts_by_node(dead_node.id).await {
        Ok(e) => e,
        Err(e) => {
            log::error!("recover: failed to query experts for {dead_hostname}: {e}");
            return;
        }
    };

    if dead_experts.is_empty() {
        log::info!("recover: no experts on node {dead_hostname}");
        return;
    }

    // 3. Identify unique experts (no active replica elsewhere)
    //
    // Fetch active nodes once, then check each expert.  node_by_expert()
    // returns ALL nodes (including deactivated ones), so we cross-check
    // against the active set to avoid phantom replicas.
    let active_nodes = reader.active_nodes().await.unwrap_or_default();
    let active_hostnames: std::collections::HashSet<String> =
        active_nodes.iter().map(|n| n.hostname.clone()).collect();

    // Surviving active nodes (exclude the dead one and any under progressive loading)
    let mut target_nodes: Vec<_> = Vec::new();
    for n in &active_nodes {
        if n.hostname == dead_hostname {
            continue;
        }
        if super::progressive::is_progressive_loading(&n.hostname).await {
            continue;
        }
        target_nodes.push(n);
    }

    if target_nodes.is_empty() {
        log::error!("recover: no surviving active nodes to receive experts from {dead_hostname}");
        return;
    }

    let mut unique_expert_ids: Vec<String> = Vec::new();
    for expert in &dead_experts {
        match reader.node_by_expert(&expert.expert_id).await {
            Ok(nodes) => {
                let other_active = nodes
                    .iter()
                    .filter(|n| n.hostname != dead_hostname)
                    .filter(|n| active_hostnames.contains(&n.hostname))
                    .count();
                if other_active == 0 {
                    unique_expert_ids.push(expert.expert_id.clone());
                }
            }
            Err(e) => {
                log::warn!(
                    "recover: can't check replicas for {}: {e}",
                    expert.expert_id
                );
            }
        }
    }

    if unique_expert_ids.is_empty() {
        log::info!(
            "recover_unique_experts: all experts on {dead_hostname} are replicated elsewhere"
        );
        return;
    }

    let identification_ms = recovery_start.elapsed().as_millis();
    log::info!(
        "recover_unique_experts: recovering {} unique experts from {} \
         (identification: {}ms, {} target nodes: [{}])",
        unique_expert_ids.len(),
        dead_hostname,
        identification_ms,
        target_nodes.len(),
        target_nodes
            .iter()
            .map(|n| n.hostname.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    );

    // ── 4. Resolve instance_id ────────────────────────────────────────────
    let instance = match reader
        .instance_by_name(&settings.inference.instance_name)
        .await
    {
        Ok(Some(i)) => i,
        Ok(None) => {
            log::error!(
                "recover: instance '{}' not found",
                settings.inference.instance_name
            );
            return;
        }
        Err(e) => {
            log::error!("recover: failed to fetch instance: {e}");
            return;
        }
    };

    // ── 5. Capacity-aware assignment: best-fit across target nodes ──────
    //
    // Build a capacity map and greedily assign each expert to the node with
    // the most remaining reported expert slots.
    let mut capacity: Vec<(&_, u64)> = Vec::new();
    for n in &target_nodes {
        let rem = remaining_expert_capacity(n, &reader).await;
        capacity.push((n, rem));
    }

    let mut assignments: HashMap<String, Vec<NewExpert>> = HashMap::new(); // hostname → experts
    let mut unplaced: Vec<String> = Vec::new();

    for expert_id in &unique_expert_ids {
        // Pick the node with the most remaining capacity
        capacity.sort_by(|a, b| b.1.cmp(&a.1));

        if let Some((target, rem)) = capacity.first_mut() {
            if *rem > 0 {
                assignments
                    .entry(target.hostname.clone())
                    .or_default()
                    .push(NewExpert {
                        instance_id: instance.id,
                        node_id: target.id,
                        expert_id: expert_id.clone(),
                        replica: 0,
                        state: serde_json::Value::Null,
                    });
                *rem -= 1;
            } else {
                unplaced.push(expert_id.clone());
            }
        }
    }

    if !unplaced.is_empty() {
        log::warn!(
            "recover: {} experts could not be placed — trying evict-then-place",
            unplaced.len(),
        );
    }

    // ── 5b. Evict-then-place for remaining unplaced experts ──────────────
    let writer = StateWriterImpl::new();

    let eviction_targets = if !unplaced.is_empty() {
        let (still_unplaced, affected) = evict_and_place(
            unplaced,
            &target_nodes,
            instance.id,
            dead_hostname,
            &reader,
            &writer,
        )
        .await;

        if !still_unplaced.is_empty() {
            log::error!(
                "recover: {} experts remain unplaced after eviction — \
                 no evictable replicas on any target",
                still_unplaced.len(),
            );
        }
        affected
    } else {
        HashSet::new()
    };

    // Log distribution
    for (hostname, experts) in &assignments {
        log::info!(
            "recover: assigning {} experts to {} (batch upsert)",
            experts.len(),
            hostname,
        );
    }

    // ── 6. Batch upsert all direct assignments ────────────────────────────
    let all_new_experts: Vec<NewExpert> = assignments.values().flatten().cloned().collect();
    let batch_count = all_new_experts.len();

    match writer.expert_upsert_batch(all_new_experts).await {
        Ok(n) => {
            log::info!(
                "recover: batch upsert complete — {} rows in {}ms",
                n,
                recovery_start.elapsed().as_millis(),
            );
        }
        Err(e) => {
            log::error!("recover: batch upsert failed: {e}");
            return;
        }
    }

    // ── 7. Trigger each affected worker to start loading ──────────────────
    //
    // Include both direct-assignment targets and eviction targets so workers
    // that received evict-then-place changes also get notified.
    let mut trigger_hostnames: HashSet<&str> =
        assignments.keys().map(|s| s.as_str()).collect();
    trigger_hostnames.extend(eviction_targets.iter().map(|s| s.as_str()));

    for hostname in &trigger_hostnames {
        let target = match target_nodes.iter().find(|n| n.hostname == *hostname) {
            Some(n) => n,
            None => continue,
        };
        match reader.experts_by_node(target.id).await {
            Ok(experts) => {
                let dispatchable: Vec<_> = experts
                    .into_iter()
                    .filter(|e| {
                        e.state.get("status").and_then(|s| s.as_str()) != Some("scheduled")
                    })
                    .collect();
                DISPATCHER
                    .lock()
                    .await
                    .trigger_worker(hostname, dispatchable)
                    .await;
            }
            Err(e) => {
                log::error!("recover: can't fetch experts for {hostname}: {e}");
            }
        }
    }

    log::info!(
        "recover_unique_experts: {} placed + {} evict-placed in {}ms \
         (node: {}, targets: {})",
        batch_count,
        eviction_targets.len(),
        recovery_start.elapsed().as_millis(),
        dead_hostname,
        trigger_hostnames
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", "),
    );
}

/// Evict replicated (non-unique) experts from full target nodes to make room
/// for unplaced unique experts.  An expert is safe to evict if it has at least
/// one other active replica.  Among candidates, the least-frequently-accessed
/// expert is evicted first.
///
/// Returns `(still_unplaced, affected_hostnames)`.
async fn evict_and_place(
    unplaced: Vec<String>,
    target_nodes: &[&Node],
    instance_id: i32,
    dead_hostname: &str,
    reader: &StateReaderImpl,
    writer: &StateWriterImpl,
) -> (Vec<String>, HashSet<String>) {
    // ── Phase 1: Build evictability index ──────────────────────────────────
    //
    // For every expert on every target node, count how many OTHER active
    // nodes also host it.  An expert with active_replica_count >= 2 is
    // safe to evict from one node without losing coverage.

    let active_nodes = reader.active_nodes().await.unwrap_or_default();
    let active_hostnames: HashSet<String> =
        active_nodes.iter().map(|n| n.hostname.clone()).collect();

    let mut node_experts: HashMap<i32, Vec<Expert>> = HashMap::new();
    let mut replica_counts: HashMap<String, usize> = HashMap::new();

    for n in target_nodes {
        let experts = match reader.experts_by_node(n.id).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        for expert in &experts {
            if replica_counts.contains_key(&expert.expert_id) {
                continue; // already counted
            }
            if let Ok(nodes) = reader.node_by_expert(&expert.expert_id).await {
                let count = nodes
                    .iter()
                    .filter(|n| n.hostname != dead_hostname)
                    .filter(|n| active_hostnames.contains(&n.hostname))
                    .count();
                replica_counts.insert(expert.expert_id.clone(), count);
            }
        }
        node_experts.insert(n.id, experts);
    }

    // ── Phase 2: Match unplaced experts to eviction candidates ────────────

    let freq = get_freq_tracker();
    let has_freq = freq.has_data();

    let mut evictions: Vec<(i32, String)> = Vec::new(); // (node_id, victim_expert_id)
    let mut placements: Vec<NewExpert> = Vec::new();
    let mut still_unplaced: Vec<String> = Vec::new();
    let mut evicted_set: HashSet<(i32, String)> = HashSet::new();

    for expert_id in &unplaced {
        let mut placed = false;

        for n in target_nodes {
            let experts = match node_experts.get(&n.id) {
                Some(e) => e,
                None => continue,
            };

            // Find evictable candidates on this node
            let mut candidates: Vec<&Expert> = experts
                .iter()
                .filter(|e| {
                    // Must be loaded (not scheduled/pending)
                    let status = e.state.get("status").and_then(|s| s.as_str());
                    status == Some("loaded") || e.state.is_null()
                })
                .filter(|e| {
                    // Must have at least one other active replica
                    let count = replica_counts.get(&e.expert_id).copied().unwrap_or(0);
                    count >= 2
                })
                .filter(|e| {
                    // Not already evicted in this batch
                    !evicted_set.contains(&(n.id, e.expert_id.clone()))
                })
                .collect();

            // Sort: evict least valuable first
            if has_freq {
                // Lowest frequency first
                candidates.sort_by_key(|e| freq.rate_in_window(&e.expert_id));
            } else {
                // Most replicas first (safest to lose one copy)
                candidates.sort_by(|a, b| {
                    let ra = replica_counts.get(&a.expert_id).copied().unwrap_or(0);
                    let rb = replica_counts.get(&b.expert_id).copied().unwrap_or(0);
                    rb.cmp(&ra)
                });
            }

            if let Some(victim) = candidates.first() {
                log::info!(
                    "evict_and_place: evicting {} from {} to place {}",
                    victim.expert_id,
                    n.hostname,
                    expert_id,
                );

                evictions.push((n.id, victim.expert_id.clone()));
                evicted_set.insert((n.id, victim.expert_id.clone()));

                // Decrement replica count so we don't over-evict
                if let Some(count) = replica_counts.get_mut(&victim.expert_id) {
                    *count = count.saturating_sub(1);
                }

                placements.push(NewExpert {
                    instance_id,
                    node_id: n.id,
                    expert_id: expert_id.clone(),
                    replica: 0,
                    state: serde_json::Value::Null,
                });

                placed = true;
                break;
            }
        }

        if !placed {
            still_unplaced.push(expert_id.clone());
        }
    }

    if evictions.is_empty() {
        return (still_unplaced, HashSet::new());
    }

    // ── Phase 3: Execute DB mutations ─────────────────────────────────────

    for (node_id, victim_id) in &evictions {
        if let Err(e) = writer.delete_expert_on_node(*node_id, victim_id).await {
            log::error!("evict_and_place: failed to delete {victim_id}: {e}");
        }
    }

    match writer.expert_upsert_batch(placements.clone()).await {
        Ok(n) => {
            log::info!(
                "evict_and_place: {} evictions + {} placements committed",
                evictions.len(),
                n,
            );
        }
        Err(e) => {
            log::error!("evict_and_place: batch upsert failed: {e}");
        }
    }

    // Collect affected hostnames for worker triggering
    let affected: HashSet<String> = evictions
        .iter()
        .filter_map(|(node_id, _)| {
            target_nodes
                .iter()
                .find(|n| n.id == *node_id)
                .map(|n| n.hostname.clone())
        })
        .collect();

    (still_unplaced, affected)
}
