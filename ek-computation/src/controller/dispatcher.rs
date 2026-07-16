use std::{
    collections::{BTreeMap, HashMap},
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, LazyLock},
};

use tokio::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender},
};
use tonic::async_trait;

use crate::{
    controller::elastic::frequency::get_freq_tracker,
    state::models::{Expert, NodeWithExperts},
};

#[async_trait]
/// Dispatcher trait for propagate the latest mapping between nodes and their experts
pub trait Dispatcher {
    /// Update the latest expert mapping for some nodes
    async fn update(&mut self, state: Vec<NodeWithExperts>);

    /// Subscribe to updates for a specific node, returning a receiver for expert updates
    async fn subscribe(&mut self, hostname: &str) -> Receiver<Vec<Expert>>;

    /// Unsubscribe from updates for a specific node
    async fn unsubscribe(&mut self, hostname: &str);
}

pub struct DispatcherImpl {
    ch_store: BTreeMap<String, (u64, Sender<Vec<Expert>>)>,
    next_subscription_id: u64,
    /// Per-worker fingerprint of the last-sent expert list.
    /// If the fingerprint is unchanged the send is skipped.
    last_fingerprint: HashMap<String, u64>,
}

pub static DISPATCHER: LazyLock<Arc<Mutex<DispatcherImpl>>> = LazyLock::new(|| {
    let inner = DispatcherImpl::new();
    Arc::new(Mutex::new(inner))
});

/// Hash the sorted expert IDs of a list to produce a stable fingerprint.
fn fingerprint(experts: &[Expert]) -> u64 {
    let mut ids: Vec<&str> = experts.iter().map(|e| e.expert_id.as_str()).collect();
    ids.sort_unstable();
    let mut h = DefaultHasher::new();
    for id in ids {
        id.hash(&mut h);
    }
    h.finish()
}

impl DispatcherImpl {
    fn new() -> Self {
        Self {
            ch_store: BTreeMap::new(),
            next_subscription_id: 0,
            last_fingerprint: HashMap::new(),
        }
    }
}

#[async_trait]
impl Dispatcher for DispatcherImpl {
    async fn update(&mut self, state: Vec<NodeWithExperts>) {
        // Build replica count map: expert_id → number of nodes currently hosting it.
        // Used as a popularity proxy: more replicas → more activation frequency.
        let mut replica_count: HashMap<&str, usize> = HashMap::new();
        for nwe in &state {
            for expert in &nwe.experts {
                *replica_count.entry(expert.expert_id.as_str()).or_default() += 1;
            }
        }

        for data in &state {
            let node = &data.node;
            // Sort experts for frequency-ordered progressive start.
            // If the frequency tracker has history, use actual request rates.
            // Otherwise fall back to replica-count as a popularity proxy.
            //
            // Filter out "scheduled" experts — they have DB rows for coverage
            // tracking but have not been promoted to "pending" yet and must
            // not be sent to the worker.
            let mut experts: Vec<Expert> = data
                .experts
                .iter()
                .filter(|e| {
                    e.state.get("status").and_then(|s| s.as_str()) != Some("scheduled")
                })
                .cloned()
                .collect();
            let freq = get_freq_tracker();
            if freq.has_data() {
                experts.sort_by(|a, b| {
                    let ra = freq.rate_in_window(&a.expert_id);
                    let rb = freq.rate_in_window(&b.expert_id);
                    if ra == 0 && rb == 0 {
                        // Both cold: use replica count as tiebreaker
                        let ca = replica_count.get(a.expert_id.as_str()).copied().unwrap_or(0);
                        let cb = replica_count.get(b.expert_id.as_str()).copied().unwrap_or(0);
                        cb.cmp(&ca)
                    } else {
                        rb.cmp(&ra)
                    }
                });
            } else {
                experts.sort_by(|a, b| {
                    let ca = replica_count.get(a.expert_id.as_str()).copied().unwrap_or(0);
                    let cb = replica_count.get(b.expert_id.as_str()).copied().unwrap_or(0);
                    cb.cmp(&ca)
                });
            }

            let fp = fingerprint(&experts);
            if self.last_fingerprint.get(&node.hostname) == Some(&fp) {
                continue; // expert set unchanged — skip send
            }
            self.last_fingerprint.insert(node.hostname.clone(), fp);

            if let Some((_subscription_id, ch)) = self.ch_store.get(&node.hostname)
                && let Err(e) = ch.send(experts).await
            {
                log::error!(
                    "Failed to send expert update to channel for hostname: {} err: {}",
                    node.hostname,
                    e
                );
            }
        }
    }
    async fn subscribe(&mut self, hostname: &str) -> Receiver<Vec<Expert>> {
        self.subscribe_with_lease(hostname).1
    }
    async fn unsubscribe(&mut self, hostname: &str) {
        self.ch_store.remove(hostname);
    }
}

impl DispatcherImpl {
    /// Subscribe while retaining an identifier that cannot remove a newer stream.
    pub fn subscribe_with_lease(&mut self, hostname: &str) -> (u64, Receiver<Vec<Expert>>) {
        let (tx, rx) = mpsc::channel(10);
        self.next_subscription_id = self.next_subscription_id.wrapping_add(1).max(1);
        let subscription_id = self.next_subscription_id;
        self.ch_store
            .insert(hostname.to_string(), (subscription_id, tx));
        // Clear stale fingerprint so the next poller tick always sends the full list
        self.last_fingerprint.remove(hostname);
        (subscription_id, rx)
    }

    /// Remove a subscription only when it is still the current stream.
    pub fn unsubscribe_with_lease(&mut self, hostname: &str, subscription_id: u64) -> bool {
        let is_current = self
            .ch_store
            .get(hostname)
            .is_some_and(|(current, _sender)| *current == subscription_id);
        if is_current {
            self.ch_store.remove(hostname);
        }
        is_current
    }
}

impl DispatcherImpl {
    /// Immediately send `experts` to a specific worker without waiting for the next
    /// poller tick. Used by recovery to trigger loading on the target node right away.
    pub async fn trigger_worker(&self, hostname: &str, experts: Vec<Expert>) {
        if let Some((_subscription_id, ch)) = self.ch_store.get(hostname) {
            if let Err(e) = ch.send(experts).await {
                log::warn!("trigger_worker: failed to send to {hostname}: {e}");
            }
        } else {
            log::warn!("trigger_worker: no active channel for {hostname}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::models::Expert;

    fn make_expert(id: &str) -> Expert {
        Expert {
            id: 0,
            instance_id: 0,
            node_id: 0,
            expert_id: id.to_string(),
            replica: 0,
            state: serde_json::Value::Null,
        }
    }

    #[test]
    fn fingerprint_order_independent() {
        let a = vec![make_expert("m/l0-e1"), make_expert("m/l0-e0")];
        let b = vec![make_expert("m/l0-e0"), make_expert("m/l0-e1")];
        assert_eq!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn fingerprint_differs_on_change() {
        let a = vec![make_expert("m/l0-e0"), make_expert("m/l0-e1")];
        let b = vec![make_expert("m/l0-e0"), make_expert("m/l0-e2")];
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn fingerprint_empty() {
        assert_eq!(fingerprint(&[]), fingerprint(&[]));
    }

    #[test]
    fn an_old_subscription_cannot_remove_its_replacement() {
        let mut dispatcher = DispatcherImpl::new();
        let (first, _first_receiver) = dispatcher.subscribe_with_lease("worker-0");
        let (second, _second_receiver) = dispatcher.subscribe_with_lease("worker-0");
        assert!(!dispatcher.unsubscribe_with_lease("worker-0", first));
        assert!(dispatcher.ch_store.contains_key("worker-0"));
        assert!(dispatcher.unsubscribe_with_lease("worker-0", second));
        assert!(!dispatcher.ch_store.contains_key("worker-0"));
    }
}
