use std::{sync::{Arc, LazyLock}, time::Duration};

use diesel::{
    BelongingToDsl, ExpressionMethods, GroupedBy, SelectableHelper,
    query_dsl::methods::{FilterDsl, SelectDsl},
};
use diesel_async::RunQueryDsl;
use ek_base::{config::get_ek_settings, error::EKResult};
use std::time::SystemTime;
use tokio::{sync::Notify, time::{self}};
use tonic::async_trait;


use crate::{
    schema,
    state::{
        models::{self, NodeWithExperts},
        pool,
    },
};

use super::{
    dispatcher::{DISPATCHER, Dispatcher},
    elastic::{ELASTIC_MANAGER, frequency::get_freq_tracker},
    routing_broadcaster::RoutingBroadcaster,
};

/// Notify handle for forcing an immediate poller tick.
///
/// When a node is removed (stream close / kill), the routing table must be
/// refreshed immediately so that the subscription stream delivers a post-removal
/// update to frontends.  Without this, a stale in-flight poller update can
/// re-introduce the dead node's endpoints via the subscription.
static POLL_NOW: LazyLock<Notify> = LazyLock::new(Notify::new);

/// Request an immediate poller tick.  Called from the heartbeat stream-close
/// handler after `remove_node` + `deactivate_node`.
pub fn request_immediate_poll() {
    POLL_NOW.notify_one();
}

#[async_trait]
pub trait StatePoller {
    async fn run(&mut self) -> EKResult<()>;
}

pub struct StatePollerImpl {
    broadcaster: Arc<RoutingBroadcaster>,
}

#[async_trait]
impl StatePoller for StatePollerImpl {
    async fn run(&mut self) -> EKResult<()> {
        let poller_secs = get_ek_settings().controller.fault_detection.poller_interval_secs;
        log::info!("state poller started (interval={}s)", poller_secs);
        let mut interval = time::interval(Duration::from_secs(poller_secs));
        loop {
            // Wait for either the regular interval OR an explicit request
            tokio::select! {
                _ = interval.tick() => {},
                _ = POLL_NOW.notified() => {
                    log::info!("state poller: immediate tick requested (node removal)");
                    interval.reset(); // avoid double-tick shortly after
                },
            }
            log::info!("state poller tick");
            let r = self.poll_state().await;
            if let Err(e) = r {
                log::error!("state poller error: {e}");
            }
        }
    }
}

impl StatePollerImpl {
    pub fn new(broadcaster: Arc<RoutingBroadcaster>) -> Self {
        StatePollerImpl {
            broadcaster,
        }
    }

    /// Polls the state of the system, fetching nodes and their associated experts,
    async fn poll_state(&mut self) -> EKResult<()> {
        let mut conn = pool::POOL.get().await?;
        let settings = get_ek_settings();

        // Fetch instance by name in settings.
        // If the instance doesn't exist yet (no worker has joined to trigger
        // auto-creation via progressive_assign), skip this tick gracefully.
        let instance = match schema::instance::table
            .filter(schema::instance::name.eq(settings.inference.instance_name.clone()))
            .first::<models::Instance>(&mut conn)
            .await
        {
            Ok(i) => i,
            Err(diesel::result::Error::NotFound) => {
                log::debug!(
                    "state poller: instance '{}' not found yet, skipping tick",
                    settings.inference.instance_name
                );
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        // Calculate threshold time for active nodes
        let threshold_secs = settings.controller.fault_detection.node_active_threshold_secs;
        let threshold_time = SystemTime::now()
            .checked_sub(std::time::Duration::from_secs(threshold_secs))
            .unwrap_or(SystemTime::UNIX_EPOCH);

        // Fetch only active nodes (last_seen_at within threshold)
        let nodes = schema::node::table
            .filter(schema::node::last_seen_at.gt(threshold_time))
            .select(models::Node::as_select())
            .load(&mut conn)
            .await?;

        // Fetch experts associated with the instance
        let experts = models::Expert::belonging_to(&nodes)
            .select(models::Expert::as_select())
            .filter(schema::expert::instance_id.eq(instance.id))
            .load(&mut conn)
            .await?;

        // Group experts by nodes
        let node_with_expert = experts
            .grouped_by(&nodes)
            .into_iter()
            .zip(nodes)
            .map(|(e, n)| NodeWithExperts {
                experts: e,
                node: n,
            })
            .collect::<Vec<NodeWithExperts>>();

        // update the dispatcher with the new state
        let mut lg = DISPATCHER.lock().await;
        let nodes_count = node_with_expert.len();
        log::info!(nodes_count; "polling nodes");
        lg.update(node_with_expert.clone()).await;
        drop(lg); // Release dispatcher lock before updating routing

        // Update routing table
        self.update_routing(node_with_expert).await?;

        // Close the current frequency window and run hotspot/replication checks
        get_freq_tracker().commit_tick();
        ELASTIC_MANAGER.lock().await.run_tick().await;

        Ok(())
    }

    /// Check if an expert is ready for routing (state is "loaded" or null for backwards compatibility)
    fn is_expert_loaded(expert: &models::Expert) -> bool {
        // Null state means expert was registered before progressive startup support - treat as loaded
        if expert.state.is_null() {
            return true;
        }

        // Check for {"status": "loaded"} in the state JSON
        expert
            .state
            .get("status")
            .and_then(|s| s.as_str())
            .map(|s| s == "loaded")
            .unwrap_or(false)
    }

    /// Update routing table based on current state
    /// Only includes experts that are marked as "loaded" or have null state (backwards compatibility)
    async fn update_routing(&self, node_with_experts: Vec<NodeWithExperts>) -> EKResult<()> {
        use std::collections::HashMap;
        use crate::proto::ek::control::v1::WorkerEndpoint;

        // Build map of expert_id → Vec<WorkerEndpoint> (collect ALL workers per expert)
        let mut routing_updates: HashMap<String, Vec<WorkerEndpoint>> = HashMap::new();
        let mut skipped_pending = 0;

        for nwe in node_with_experts {
            // Extract worker info from node config
            let node_addr = nwe
                .node
                .config
                .get("addr")
                .and_then(|a| a.as_str())
                .unwrap_or("unknown")
                .to_string();

            let channel = nwe
                .node
                .config
                .get("channel")
                .and_then(|c| c.as_str())
                .unwrap_or("grpc")
                .to_string();

            let rdma_tcp_port = nwe
                .node
                .config
                .get("rdma_tcp_port")
                .and_then(|p| p.as_u64())
                .unwrap_or(0) as u32;

            let device = nwe.node.device.clone();

            let wm_addr = nwe
                .node
                .config
                .get("wm_addr")
                .and_then(|a| a.as_str())
                .unwrap_or("")
                .to_string();

            // Create WorkerEndpoint
            let endpoint = WorkerEndpoint {
                grpc_addr: node_addr.clone(),
                channel: channel.clone(),
                rdma_tcp_port,
                shm_queue_prefix: nwe.node.hostname.clone(), // Use hostname as queue prefix
                device,
                wm_addr,
            };

            // For each expert on this node, add endpoint to list (only if loaded)
            for expert in nwe.experts {
                // Skip experts that are not yet loaded (progressive startup support)
                if !Self::is_expert_loaded(&expert) {
                    skipped_pending += 1;
                    continue;
                }

                let expert_id = expert.expert_id;

                // Collect all workers hosting this expert (multi-replica support)
                routing_updates
                    .entry(expert_id.clone())
                    .or_default()
                    .push(endpoint.clone());
            }
        }

        // Log multi-replica experts and pending experts for debugging
        let multi_replica_count = routing_updates.values().filter(|v| v.len() > 1).count();
        if skipped_pending > 0 {
            log::info!(
                "Updating routing table with {} loaded experts ({} with multiple replicas), {} pending experts skipped",
                routing_updates.len(),
                multi_replica_count,
                skipped_pending
            );
        } else if multi_replica_count > 0 {
            log::info!(
                "Updating routing table with {} experts ({} with multiple replicas)",
                routing_updates.len(),
                multi_replica_count
            );
        } else {
            log::info!("Updating routing table with {} experts", routing_updates.len());
        }

        self.broadcaster.batch_update(routing_updates).await;

        Ok(())
    }
}

pub fn start_poll(broadcaster: Arc<RoutingBroadcaster>) {
    let mut poller = StatePollerImpl::new(broadcaster);
    tokio::spawn(async move {
        if let Err(e) = poller.run().await {
            log::error!("state poller error {e}");
        }
    });
}
