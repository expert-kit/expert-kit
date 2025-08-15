use std::{
    collections::BTreeMap,
    sync::{Arc, LazyLock},
};

use tokio::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender},
};
use tonic::async_trait;

use crate::state::models::{Expert, NodeWithExperts};

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
    ch_store: BTreeMap<String, Sender<Vec<Expert>>>,
}

pub static DISPATCHER: LazyLock<Arc<Mutex<DispatcherImpl>>> = LazyLock::new(|| {
    let inner = DispatcherImpl::new();
    Arc::new(Mutex::new(inner))
});

impl DispatcherImpl {
    fn new() -> Self {
        Self {
            ch_store: BTreeMap::new(),
        }
    }
}

#[async_trait]
impl Dispatcher for DispatcherImpl {
    async fn update(&mut self, state: Vec<NodeWithExperts>) {
        for data in &state {
            let node = &data.node;
            let experts = &data.experts;
            // Find the channel for the host
            // If it exists, send the experts updates to the channel
            if let Some(ch) = self.ch_store.get(&node.hostname)
                && let Err(e) = ch.send(experts.clone()).await
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
        let (tx, rx) = mpsc::channel(10);
        self.ch_store.insert(hostname.to_string(), tx);
        rx
    }
    async fn unsubscribe(&mut self, hostname: &str) {
        self.ch_store.remove(hostname);
    }
}
