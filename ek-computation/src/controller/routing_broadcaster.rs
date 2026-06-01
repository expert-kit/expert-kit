use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use tokio::sync::{RwLock, broadcast};

use crate::proto::ek::control::v1::{GetRoutingResp, RoutingUpdate, WorkerEndpoint, WorkerEndpointList, routing_update::ChangeType};

pub static BROADCASTER: LazyLock<Arc<RoutingBroadcaster>> =
    LazyLock::new(|| Arc::new(RoutingBroadcaster::new(1000)));

pub fn get_broadcaster() -> Arc<RoutingBroadcaster> {
    BROADCASTER.clone()
}

/// RoutingBroadcaster manages routing table and broadcasts updates to subscribed frontends
/// This is the central pub/sub system for routing metadata distribution
#[derive(Clone)]
pub struct RoutingBroadcaster {
    inner: Arc<RoutingBroadcasterInner>,
}

struct RoutingBroadcasterInner {
    /// Current routing table: expert_id → list of WorkerEndpoints (multi-replica support)
    routing: RwLock<HashMap<String, Vec<WorkerEndpoint>>>,

    /// Current version number (incremented on every change)
    version: RwLock<u64>,

    /// Broadcast channel for routing updates (unbounded, drop slow subscribers)
    update_tx: broadcast::Sender<RoutingUpdate>,
}

impl RoutingBroadcaster {
    /// Create a new RoutingBroadcaster with specified channel capacity
    pub fn new(channel_capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(channel_capacity);

        Self {
            inner: Arc::new(RoutingBroadcasterInner {
                routing: RwLock::new(HashMap::new()),
                version: RwLock::new(0),
                update_tx: tx,
            }),
        }
    }

    /// Get the current routing table snapshot
    pub async fn get_routing(&self, expert_ids: Option<Vec<String>>) -> GetRoutingResp {
        let routing_map = self.inner.routing.read().await;
        let version = *self.inner.version.read().await;

        let routing = if let Some(filter) = expert_ids {
            // Return only requested experts
            filter
                .into_iter()
                .filter_map(|id| {
                    routing_map.get(&id).map(|endpoints| {
                        (id, WorkerEndpointList { endpoints: endpoints.clone() })
                    })
                })
                .collect()
        } else {
            // Return all experts
            routing_map
                .iter()
                .map(|(id, endpoints)| {
                    (id.clone(), WorkerEndpointList { endpoints: endpoints.clone() })
                })
                .collect()
        };

        GetRoutingResp { routing, version }
    }

    /// Subscribe to routing updates
    /// Returns a receiver that will receive all future updates
    pub fn subscribe(&self) -> broadcast::Receiver<RoutingUpdate> {
        self.inner.update_tx.subscribe()
    }

    /// Get current version number
    pub async fn get_version(&self) -> u64 {
        *self.inner.version.read().await
    }

    /// Add or update an expert mapping with a single endpoint
    /// For multi-endpoint updates, use upsert_expert_endpoints instead
    pub async fn upsert_expert(&self, expert_id: String, endpoint: WorkerEndpoint) {
        self.upsert_expert_endpoints(expert_id, vec![endpoint]).await;
    }

    /// Add or update an expert mapping with multiple endpoints
    pub async fn upsert_expert_endpoints(&self, expert_id: String, endpoints: Vec<WorkerEndpoint>) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        let change_type = if routing.contains_key(&expert_id) {
            ChangeType::Modified
        } else {
            ChangeType::Added
        };

        routing.insert(expert_id.clone(), endpoints.clone());
        *version += 1;

        let update = RoutingUpdate {
            r#type: change_type as i32,
            expert_id,
            endpoints: Some(WorkerEndpointList { endpoints }),
            version: *version,
        };

        // Broadcast update (ignore if no subscribers)
        let _ = self.inner.update_tx.send(update);
    }

    /// Remove an expert mapping (e.g., when worker goes offline)
    pub async fn remove_expert(&self, expert_id: String) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        if routing.remove(&expert_id).is_some() {
            *version += 1;

            let update = RoutingUpdate {
                r#type: ChangeType::Removed as i32,
                expert_id,
                endpoints: None, // No endpoints for removals
                version: *version,
            };

            // Broadcast update (ignore if no subscribers)
            let _ = self.inner.update_tx.send(update);
        }
    }

    /// Batch update multiple expert mappings atomically (multi-endpoint version)
    pub async fn batch_update(&self, updates: HashMap<String, Vec<WorkerEndpoint>>) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        for (expert_id, endpoints) in updates {
            let change_type = if routing.contains_key(&expert_id) {
                ChangeType::Modified
            } else {
                ChangeType::Added
            };

            routing.insert(expert_id.clone(), endpoints.clone());
            *version += 1;

            let update = RoutingUpdate {
                r#type: change_type as i32,
                expert_id,
                endpoints: Some(WorkerEndpointList { endpoints }),
                version: *version,
            };

            // Broadcast each update
            let _ = self.inner.update_tx.send(update);
        }
    }

    /// Batch remove multiple expert mappings atomically
    pub async fn batch_remove(&self, expert_ids: Vec<String>) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        for expert_id in expert_ids {
            if routing.remove(&expert_id).is_some() {
                *version += 1;

                let update = RoutingUpdate {
                    r#type: ChangeType::Removed as i32,
                    expert_id,
                    endpoints: None,
                    version: *version,
                };

                // Broadcast each update
                let _ = self.inner.update_tx.send(update);
            }
        }
    }

    /// Number of `WorkerEndpoint` entries currently routing for `expert_id`.
    pub async fn replica_count(&self, expert_id: &str) -> usize {
        self.inner
            .routing
            .read()
            .await
            .get(expert_id)
            .map(|eps| eps.len())
            .unwrap_or(0)
    }

    /// Total number of distinct expert_ids currently in the routing table.
    pub async fn routed_expert_count(&self) -> usize {
        self.inner.routing.read().await.len()
    }

    /// Remove all endpoints belonging to a dead worker node immediately.
    ///
    /// For each expert: if the node was the sole replica, the expert is removed
    /// (ChangeType::Removed). If other replicas remain, the entry is updated
    /// (ChangeType::Modified). Broadcasts an update for every affected expert.
    pub async fn remove_node(&self, hostname: &str) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        let mut to_remove = Vec::new();
        let mut to_update: Vec<(String, Vec<WorkerEndpoint>)> = Vec::new();

        for (expert_id, endpoints) in routing.iter() {
            let remaining: Vec<WorkerEndpoint> = endpoints
                .iter()
                .filter(|ep| ep.shm_queue_prefix != hostname)
                .cloned()
                .collect();

            if remaining.len() < endpoints.len() {
                if remaining.is_empty() {
                    to_remove.push(expert_id.clone());
                } else {
                    to_update.push((expert_id.clone(), remaining));
                }
            }
        }

        let affected = to_remove.len() + to_update.len();
        log::info!("Removing node {} from routing ({} experts affected)", hostname, affected);

        for expert_id in to_remove {
            routing.remove(&expert_id);
            *version += 1;
            let update = RoutingUpdate {
                r#type: ChangeType::Removed as i32,
                expert_id,
                endpoints: None,
                version: *version,
            };
            let _ = self.inner.update_tx.send(update);
        }

        for (expert_id, endpoints) in to_update {
            routing.insert(expert_id.clone(), endpoints.clone());
            *version += 1;
            let update = RoutingUpdate {
                r#type: ChangeType::Modified as i32,
                expert_id,
                endpoints: Some(WorkerEndpointList { endpoints }),
                version: *version,
            };
            let _ = self.inner.update_tx.send(update);
        }
    }
}
