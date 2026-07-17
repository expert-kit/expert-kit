use std::{collections::HashSet, sync::LazyLock, time::Duration};

use ek_base::config::get_ek_settings;
use tokio::sync::Mutex;

/// Hostnames currently undergoing progressive expert assignment.
/// `select_worker_for_new_replica` excludes these to prevent over-assignment races.
static PROGRESSIVE_LOADING: LazyLock<Mutex<HashSet<String>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Serializes the coverage-check + assignment phase of progressive_assign.
/// Without this, two workers starting simultaneously both see "128/128 uncovered"
/// and assign the same expert subset, leaving a coverage gap.
static ASSIGN_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

/// Returns true if the given hostname is currently mid-progressive-load.
pub async fn is_progressive_loading(hostname: &str) -> bool {
    PROGRESSIVE_LOADING.lock().await.contains(hostname)
}
use ek_db::{safetensor::ExpertKey, weight_srv::client::WeightSrvClient};

use crate::{
    controller::dispatcher::DISPATCHER,
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewExpert,
        writer::StateWriterImpl,
    },
};

/// Automatically assign experts to a newly registered worker in bounded stripes.
///
/// Called as a background task when a worker sends its first heartbeat.
/// Exits immediately if `scaling.auto_assign` is false (default).
pub async fn progressive_assign(new_hostname: &str) {
    let settings = get_ek_settings();
    if !settings.controller.scaling.auto_assign {
        return;
    }

    log::info!("progressive_assign: starting for new worker {new_hostname}");

    // --- Load model vital metadata from weight server ---
    let ws_addr = match settings.weight.server.as_ref() {
        Some(s) => s.addr.clone(),
        None => {
            log::warn!("progressive_assign: no weight server configured, skipping");
            return;
        }
    };
    let model_name = settings.inference.model_name.clone();
    let cli = WeightSrvClient::new(ws_addr);
    let vital = match cli.load_meta_vital(&model_name).await {
        Ok(v) => v,
        Err(e) => {
            log::error!("progressive_assign: failed to load vital meta for {model_name}: {e}");
            return;
        }
    };

    // --- Look up the new node in DB ---
    let reader = StateReaderImpl::new();
    let node = match reader.node_by_hostname(new_hostname).await {
        Ok(Some(n)) => n,
        Ok(None) => {
            log::warn!("progressive_assign: node {new_hostname} not found in DB yet");
            return;
        }
        Err(e) => {
            log::error!("progressive_assign: DB error looking up {new_hostname}: {e}");
            return;
        }
    };

    // --- Clean stale expert assignments from previous runs ---
    let writer = StateWriterImpl::new();
    match reader.experts_by_node(node.id).await {
        Ok(old) if !old.is_empty() => {
            log::info!(
                "progressive_assign: clearing {} stale expert assignments for returning node {new_hostname}",
                old.len()
            );
            if let Err(e) = writer.delete_experts_by_node(node.id).await {
                log::error!("progressive_assign: failed to clear stale experts: {e}");
                return;
            }
        }
        _ => {}
    }

    // --- Resolve instance (auto-create if missing) ---
    let instance = match reader
        .instance_by_name(&settings.inference.instance_name)
        .await
    {
        Ok(Some(i)) => i,
        Ok(None) => {
            log::info!(
                "progressive_assign: instance '{}' not found, creating from config",
                settings.inference.instance_name
            );
            let model = match reader.model_by_name(&model_name).await {
                Ok(Some(m)) => m,
                _ => {
                    log::error!("progressive_assign: model '{model_name}' not found in DB");
                    return;
                }
            };
            let writer = StateWriterImpl::new();
            match writer
                .instance_upsert(crate::state::models::NewInstance {
                    model_id: model.id,
                    name: settings.inference.instance_name.clone(),
                })
                .await
            {
                Ok(i) => {
                    log::info!("progressive_assign: created instance '{}'", i.name);
                    i
                }
                Err(e) => {
                    log::error!("progressive_assign: failed to create instance: {e}");
                    return;
                }
            }
        }
        Err(e) => {
            log::error!("progressive_assign: failed to fetch instance: {e}");
            return;
        }
    };

    // --- Compute target experts per layer ---
    let n_layers = vital.moe_layers.1.saturating_sub(vital.moe_layers.0);
    if n_layers == 0 {
        log::warn!("progressive_assign: no MoE layers found for model {model_name}");
        return;
    }

    let max_experts = node
        .config
        .get("max_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let target_per_layer = target_per_layer(max_experts, n_layers, vital.routed_experts);
    if target_per_layer == 0 {
        log::error!(
            "progressive_assign: {new_hostname} reports {max_experts} expert slots, \
             which cannot cover one expert across {n_layers} layers"
        );
        return;
    }

    log::info!(
        "progressive_assign: {new_hostname} capacity={max_experts} experts, \
         target={target_per_layer}/{} experts/layer across {n_layers} layers",
        vital.routed_experts
    );

    // --- Serialize coverage check + DB insertion ---
    //
    // Hold ASSIGN_LOCK while reading coverage and writing ALL expert
    // assignments to the DB.  Without this, two workers starting
    // simultaneously both read "128/128 uncovered", assign the same
    // expert subset, and leave a coverage gap. The lock ensures
    // the second worker sees the first worker's DB rows and fills gaps.
    //
    // The lock is released BEFORE progressive dispatching (stripe-by-stripe
    // loading with delays), since the DB rows are already visible.

    let sorted_indices;
    let experts_per_step;

    {
        let _assign_guard = ASSIGN_LOCK.lock().await;

        // --- Build coverage-aware expert index list ---
        //
        // First priority: assign experts with ZERO replicas (coverage gaps).
        // Second priority: fill remaining capacity in stable expert-index order.
        //
        // This ensures two partial-capacity nodes together cover the full model
        // before duplicating anything.
        let expert_indices: Vec<usize> = (0..vital.routed_experts).collect();

        // Count active replicas per expert index (aggregated across all layers).
        // An expert index with zero replicas on any layer is "uncovered".
        let active_nodes = reader.active_nodes().await.unwrap_or_default();
        let active_node_ids: std::collections::HashSet<i32> =
            active_nodes.iter().map(|n| n.id).collect();

        let mut replica_counts: Vec<usize> = vec![0; vital.routed_experts];
        // Sample a single layer to determine per-index replica counts (they're
        // symmetric across layers since progressive_assign assigns the same
        // indices to every layer).
        let sample_layer = vital.moe_layers.0;
        for (idx, replica_count) in replica_counts.iter_mut().enumerate() {
            let key = ExpertKey::new(model_name.clone(), sample_layer, idx);
            let eid = key.as_object_key();
            if let Ok(nodes) = reader.node_by_expert(&eid).await {
                *replica_count = nodes
                    .iter()
                    .filter(|n| active_node_ids.contains(&n.id))
                    .count();
            }
        }

        // Sort zero-replica experts first, then preserve stable expert-index order.
        let mut prioritized: Vec<usize> = expert_indices;
        prioritized.sort_by(|&a, &b| {
            replica_counts[a]
                .cmp(&replica_counts[b])
                .then_with(|| a.cmp(&b))
        });

        let zero_replica_count = replica_counts.iter().filter(|&&c| c == 0).count();
        log::info!(
            "progressive_assign: {new_hostname} coverage: {}/{} experts uncovered, assigning {target_per_layer} per layer",
            zero_replica_count,
            vital.routed_experts
        );

        sorted_indices = prioritized;

        experts_per_step = {
            let cfg = settings.controller.scaling.experts_per_step;
            if cfg == 0 {
                target_per_layer // bulk load
            } else {
                cfg.min(target_per_layer)
            }
        };

        // Insert ALL expert assignments as "scheduled" under the lock.
        // "scheduled" experts are visible for coverage queries but NOT
        // dispatched to workers or published in Frontend topology.
        let scheduled_state = serde_json::json!({"status": "scheduled"});
        for &expert_idx in &sorted_indices[..target_per_layer] {
            for layer in vital.moe_layers.0..vital.moe_layers.1 {
                let key = ExpertKey::new(model_name.clone(), layer, expert_idx);
                if let Err(e) = writer
                    .expert_upsert(NewExpert {
                        instance_id: instance.id,
                        node_id: node.id,
                        expert_id: key.as_object_key(),
                        replica: 1,
                        state: scheduled_state.clone(),
                    })
                    .await
                {
                    log::error!(
                        "progressive_assign: upsert {} layer {} expert {} failed: {e}",
                        new_hostname,
                        layer,
                        expert_idx
                    );
                }
            }
        }

        log::info!(
            "progressive_assign: {new_hostname} all {} experts/layer inserted as scheduled",
            target_per_layer
        );
        // _assign_guard dropped here — next worker can now read our rows
    }

    // --- Progressive dispatch (stripe-by-stripe loading) ---
    //
    // All expert DB rows exist as "scheduled".  Each stripe promotes a
    // batch from "scheduled" → "pending", then triggers the worker.
    // The worker loads pending experts and reports them as "loaded" via
    // heartbeat. This lets the worker start serving the first ready stripe
    // without waiting for the full assigned set.

    // Exclude this node from concurrent recovery placement until loading ends.
    PROGRESSIVE_LOADING
        .lock()
        .await
        .insert(new_hostname.to_string());

    let mut offset = 0usize;

    while offset < target_per_layer {
        let batch_end = (offset + experts_per_step).min(target_per_layer);
        let batch = &sorted_indices[offset..batch_end];

        // Promote this stripe: scheduled → pending (across all layers)
        let mut promote_ids: Vec<String> = Vec::with_capacity(batch.len() * n_layers);
        for &expert_idx in batch {
            for layer in vital.moe_layers.0..vital.moe_layers.1 {
                let key = ExpertKey::new(model_name.clone(), layer, expert_idx);
                promote_ids.push(key.as_object_key());
            }
        }
        if let Err(e) = writer
            .promote_experts_to_pending(node.id, &promote_ids)
            .await
        {
            log::error!("progressive_assign: promote stripe failed: {e}");
        }

        // Dispatch non-scheduled experts to the worker
        match reader.experts_by_node(node.id).await {
            Ok(experts) => {
                let dispatchable: Vec<_> = experts
                    .into_iter()
                    .filter(|e| e.state.get("status").and_then(|s| s.as_str()) != Some("scheduled"))
                    .collect();
                DISPATCHER
                    .lock()
                    .await
                    .trigger_worker(new_hostname, dispatchable)
                    .await;
            }
            Err(e) => {
                log::error!("progressive_assign: can't fetch experts for {new_hostname}: {e}");
            }
        }

        log::info!(
            "progressive_assign: {new_hostname} stripe {}-{} dispatched ({} experts/layer)",
            offset,
            batch_end,
            batch_end - offset
        );

        offset = batch_end;

        if settings.controller.scaling.step_delay_ms > 0 && offset < target_per_layer {
            tokio::time::sleep(Duration::from_millis(
                settings.controller.scaling.step_delay_ms,
            ))
            .await;
        }
    }

    PROGRESSIVE_LOADING.lock().await.remove(new_hostname);
    log::info!(
        "progressive_assign: {new_hostname} complete — {target_per_layer} experts/layer assigned"
    );
}

fn target_per_layer(max_experts: usize, layer_count: usize, routed_experts: usize) -> usize {
    if layer_count == 0 {
        return 0;
    }
    routed_experts.min(max_experts / layer_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reported_capacity_is_shared_across_layers() {
        assert_eq!(target_per_layer(40, 4, 16), 10);
        assert_eq!(target_per_layer(100, 4, 16), 16);
        assert_eq!(target_per_layer(3, 4, 16), 0);
    }
}
