use std::{collections::HashMap, sync::Arc};

use ek_base::error::{EKError, EKResult};
use ndarray_rand::rand;
use once_cell::sync::OnceCell;
use tokio::sync::Mutex;
use tonic::transport::Channel;

use crate::state::io::{StateReader, StateReaderImpl};

#[async_trait::async_trait]
pub trait ExpertRegistry {
    async fn select(&mut self, eid: String) -> EKResult<Channel>;
    async fn reset(&mut self) -> EKResult<()>;
    async fn deregister(&mut self, host_id: &str);
}

type ExpertId = String;

struct ChannelMeta {
    host_id: String,
    ch: Channel,
}

pub struct ExpertRegistryImpl {
    channels: HashMap<ExpertId, Vec<ChannelMeta>>,
    reader: Box<dyn StateReader + Send + Sync>,
}

#[async_trait::async_trait]
impl ExpertRegistry for ExpertRegistryImpl {
    async fn reset(&mut self) -> EKResult<()> {
        self.inner_reset().await
    }
    async fn select(&mut self, eid: String) -> EKResult<Channel> {
        self.inner_select(eid).await
    }
    async fn deregister(&mut self, host_id: &str) {
        self.inner_deregister(host_id).await;
    }
}

impl ExpertRegistryImpl {
    async fn inner_reset(&mut self) -> EKResult<()> {
        self.channels.clear();
        Ok(())
    }

    async fn inner_select(&mut self, eid: String) -> EKResult<Channel> {
        let channels = self.channels.get(&eid);
        if let Some(channels) = channels {
            if channels.is_empty() {
                return self.create_then_select_channel(eid).await;
            }
            self.select_random(eid).await
        } else {
            self.create_then_select_channel(eid).await
        }
    }

    async fn select_random(&mut self, eid: String) -> EKResult<Channel> {
        let channels = self.channels.get_mut(&eid);
        if let Some(channels) = channels {
            if channels.is_empty() {
                return self.create_then_select_channel(eid).await;
            }
            let idx = rand::random::<usize>() % channels.len();
            Ok(channels[idx].ch.clone())
        } else {
            self.create_then_select_channel(eid).await
        }
    }

    async fn create_then_select_channel(&mut self, eid: String) -> EKResult<Channel> {
        let nodes = self.reader.node_by_expert(&eid).await?;
        for node in nodes {
            let addr = node.config["addr"].as_str().unwrap().to_owned();
            let end = Channel::from_shared(addr)
                .map_err(|e| EKError::InvalidInput(format!("invalid url for gRPC: {}", e)))?;
            let ch = end.connect().await?;
            let meta = ChannelMeta {
                ch,
                host_id: node.hostname.clone(),
            };
            self.channels.insert(eid.clone(), vec![meta]);
        }
        let res = self.channels.get(&eid).ok_or(EKError::NotFound(format!(
            "no channel found for expert {}",
            eid
        )))?;
        if res.is_empty() {
            return Err(EKError::NotFound(format!(
                "no channel found for expert {}",
                eid
            )));
        }
        Ok(res[0].ch.clone())
    }

    pub async fn inner_deregister(&mut self, host_id: &str) {
        log::info!("deregister host_id {}", host_id);
        for (_, channels) in self.channels.iter_mut() {
            channels.retain(|meta| meta.host_id != host_id);
        }
    }
}

impl Default for ExpertRegistryImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertRegistryImpl {
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            reader: Box::new(StateReaderImpl::new()),
        }
    }
}

pub fn get_registry() -> Arc<Mutex<dyn ExpertRegistry + Send + Sync>> {
    static INSTANCE: OnceCell<Arc<Mutex<ExpertRegistryImpl>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = ExpertRegistryImpl::new();
        Arc::new(Mutex::new(inner))
    });
    (res.clone()) as _
}
