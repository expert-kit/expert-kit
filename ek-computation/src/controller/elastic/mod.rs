pub mod frequency;
pub mod progressive;
pub mod provisioner;
pub mod recovery;

use std::{sync::LazyLock, time::Instant};

use tokio::sync::Mutex;

use ek_base::{config::get_ek_settings, error::EKResult};

use crate::{
    controller::{
        dispatcher::DISPATCHER,
        routing_broadcaster::get_broadcaster,
        scheduler::{remaining_expert_capacity, select_worker_for_new_replica},
    },
    state::{
        io::{StateReader, StateReaderImpl},
        models::NewExpert,
        writer::StateWriterImpl,
    },
};

use self::{frequency::get_freq_tracker, provisioner::provision_nodes};

pub static ELASTIC_MANAGER: LazyLock<Mutex<ElasticManager>> =
    LazyLock::new(|| Mutex::new(ElasticManager::new()));

/// Drives reactive replication of hot experts and capacity-shortfall provisioning.
/// Called once per poller tick (every 5 s) after `FREQ_TRACKER.commit_tick()`.
pub struct ElasticManager {
    last_provision_time: Option<Instant>,
}

impl ElasticManager {
    fn new() -> Self {
        Self {
            last_provision_time: None,
        }
    }

    /// Main entry point: replicate hot experts and optionally provision new nodes.
    ///
    /// Implements "Load-Triggered Adaptation" (paper §4.2):
    /// - Stage 1 (hotspot): if an expert's request rate in the sliding window exceeds
    ///   `replication.hotspot_threshold`, replicate it onto an additional worker.
    ///   Desired replicas = ceil(rate / target_rate_per_worker). The threshold encodes
    ///   the per-worker throughput SLO (see `ReplicationSettings` in ek-base).
    /// - Stage 2 (capacity shortfall): if >50% of all routed experts are hot,
    ///   the cluster is undersized; trigger node provisioning via external script.
    pub async fn run_tick(&mut self) {
        let freq = get_freq_tracker();
        if !freq.has_data() {
            return; // no history yet — skip
        }

        let settings = get_ek_settings();
        let replication = &settings.controller.replication;
        let total_experts = get_broadcaster().routed_expert_count().await;

        let hot_count = if replication.enabled {
            let threshold = replication.hotspot_threshold;
            let target_rate = replication.target_rate_per_worker.max(1);
            let max_replicas = replication.max_replicas as usize;

            let hot_experts: Vec<(String, u64)> = freq
                .all_sorted()
                .into_iter()
                .filter(|(_, rate)| *rate > threshold)
                .collect();

            let count = hot_experts.len();
            for (expert_id, rate) in &hot_experts {
                if let Err(e) = self
                    .maybe_replicate(expert_id, *rate, target_rate, max_replicas)
                    .await
                {
                    log::warn!("ElasticManager: replication failed for {expert_id}: {e}");
                }
            }
            count
        } else {
            0
        };

        self.check_capacity_shortfall(hot_count, total_experts)
            .await;
    }

    async fn maybe_replicate(
        &self,
        expert_id: &str,
        rate: u64,
        target_rate: u64,
        max_replicas: usize,
    ) -> EKResult<()> {
        let current = get_broadcaster().replica_count(expert_id).await;
        let desired = ((rate as f64 / target_rate as f64).ceil() as usize).min(max_replicas);

        if desired <= current {
            return Ok(());
        }

        let replicas_to_add = desired - current;

        // Look up instance once per expert
        let reader = StateReaderImpl::new();
        let settings = get_ek_settings();
        let instance = match reader
            .instance_by_name(&settings.inference.instance_name)
            .await
        {
            Ok(Some(i)) => i,
            Ok(None) => {
                log::error!(
                    "ElasticManager: instance '{}' not found",
                    settings.inference.instance_name
                );
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        let writer = StateWriterImpl::new();

        for _ in 0..replicas_to_add {
            let target = match select_worker_for_new_replica(expert_id, "").await {
                Ok(n) => n,
                Err(e) => {
                    log::warn!("ElasticManager: no target for {expert_id}: {e}");
                    break;
                }
            };

            // Refuse to assign if the best candidate has no room left
            let remaining = remaining_expert_capacity(&target, &reader).await;
            if remaining == 0 {
                log::warn!(
                    "ElasticManager: skipping replication of {expert_id} → {} \
                     — no reported expert slots remain",
                    target.hostname
                );
                break;
            }

            writer
                .expert_upsert(NewExpert {
                    instance_id: instance.id,
                    node_id: target.id,
                    expert_id: expert_id.to_owned(),
                    replica: 0,
                    state: serde_json::Value::Null,
                })
                .await?;

            log::info!(
                "Replicating {} → {} (rate={}/window)",
                expert_id,
                target.hostname,
                rate
            );

            if let Ok(experts) = reader.experts_by_node(target.id).await {
                let dispatchable: Vec<_> = experts
                    .into_iter()
                    .filter(|e| {
                        e.state.get("status").and_then(|s| s.as_str()) != Some("scheduled")
                    })
                    .collect();
                DISPATCHER
                    .lock()
                    .await
                    .trigger_worker(&target.hostname, dispatchable)
                    .await;
            }
        }

        Ok(())
    }

    async fn check_capacity_shortfall(&mut self, hot_count: usize, total_experts: usize) {
        if total_experts == 0 {
            return;
        }
        let hot_fraction = hot_count as f64 / total_experts as f64;
        if hot_fraction <= 0.5 {
            return;
        }

        log::info!(
            "Capacity shortfall: {:.0}% of experts are hot ({}/{})",
            hot_fraction * 100.0,
            hot_count,
            total_experts,
        );

        let settings = get_ek_settings();
        let provisioning = &settings.controller.provisioning;
        if !provisioning.enabled {
            return;
        }
        let script = match &provisioning.script {
            Some(s) => s.clone(),
            None => {
                log::warn!("ElasticManager: provisioning.enabled=true but no script configured");
                return;
            }
        };

        let cooldown = std::time::Duration::from_secs(provisioning.cooldown_secs);
        let elapsed = self.last_provision_time.map(|t| t.elapsed());
        if elapsed.is_some_and(|e| e < cooldown) {
            log::debug!("ElasticManager: provisioner cooldown active, skipping");
            return;
        }

        self.last_provision_time = Some(Instant::now());
        let model = settings.inference.model_name.clone();
        let instance = settings.inference.instance_name.clone();
        tokio::spawn(async move {
            if let Err(e) = provision_nodes(&script, 1, &model, &instance).await {
                log::error!("ElasticManager: provisioner error: {e}");
            }
        });
    }
}
