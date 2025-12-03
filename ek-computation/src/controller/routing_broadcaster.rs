use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};

use crate::proto::ek::control::v1::{GetRoutingResp, RoutingUpdate, WorkerEndpoint, routing_update::ChangeType};

/// RoutingBroadcaster manages routing table and broadcasts updates to subscribed frontends
/// This is the central pub/sub system for routing metadata distribution
#[derive(Clone)]
pub struct RoutingBroadcaster {
    inner: Arc<RoutingBroadcasterInner>,
}

struct RoutingBroadcasterInner {
    /// Current routing table: expert_id → WorkerEndpoint
    routing: RwLock<HashMap<String, WorkerEndpoint>>,

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
                .filter_map(|id| routing_map.get(&id).map(|endpoint| (id, endpoint.clone())))
                .collect()
        } else {
            // Return all experts
            routing_map.clone()
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

    /// Add or update an expert mapping
    pub async fn upsert_expert(&self, expert_id: String, endpoint: WorkerEndpoint) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        let change_type = if routing.contains_key(&expert_id) {
            ChangeType::Modified
        } else {
            ChangeType::Added
        };

        routing.insert(expert_id.clone(), endpoint.clone());
        *version += 1;

        let update = RoutingUpdate {
            r#type: change_type as i32,
            expert_id,
            endpoint: Some(endpoint),
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
                endpoint: None, // No endpoint for removals
                version: *version,
            };

            // Broadcast update (ignore if no subscribers)
            let _ = self.inner.update_tx.send(update);
        }
    }

    /// Batch update multiple expert mappings atomically
    pub async fn batch_update(&self, updates: HashMap<String, WorkerEndpoint>) {
        let mut routing = self.inner.routing.write().await;
        let mut version = self.inner.version.write().await;

        for (expert_id, endpoint) in updates {
            let change_type = if routing.contains_key(&expert_id) {
                ChangeType::Modified
            } else {
                ChangeType::Added
            };

            routing.insert(expert_id.clone(), endpoint.clone());
            *version += 1;

            let update = RoutingUpdate {
                r#type: change_type as i32,
                expert_id,
                endpoint: Some(endpoint),
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
                    endpoint: None,
                    version: *version,
                };

                // Broadcast each update
                let _ = self.inner.update_tx.send(update);
            }
        }
    }
}
