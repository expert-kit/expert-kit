use std::{collections::BTreeMap, sync::Arc};

use once_cell::sync::Lazy;
use tokio::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender},
};
use tonic::async_trait;

use crate::state::models::{self, Expert, NodeWithExperts};

#[async_trait]
pub trait Dispatcher {
    async fn update(&mut self, state: Vec<NodeWithExperts>);
    async fn subscribe(&mut self, hostname: &str) -> Receiver<Vec<Expert>>;
    async fn unsubscribe(&mut self, hostname: &str);
}

pub struct DispatcherImpl {
    ch_store: BTreeMap<String, Sender<Vec<Expert>>>,
}

pub static DISPATCHER: Lazy<Arc<Mutex<DispatcherImpl>>> = Lazy::new(|| {
    todo!();
});

#[async_trait]
impl Dispatcher for DispatcherImpl {
    async fn update(&mut self, state: Vec<NodeWithExperts>) {
        for data in &state {
            let node = &data.node;
            let experts = &data.experts;
            if let Some(ch) = self.ch_store.get(&node.hostname) {
                if let Err(e) = ch.send(experts.clone()).await {
                    log::error!(
                        "Failed to send expert update to channel for hostname: {} err: {}",
                        node.hostname,
                        e
                    );
                }
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
