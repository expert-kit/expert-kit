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
}

type ExpertId = String;

struct ChannelMeta {
    ch: Channel,
}

pub struct ExpertRegistryImpl {
    channels: HashMap<ExpertId, Vec<ChannelMeta>>,
    reader: Box<dyn StateReader + Send + Sync>,
}

#[async_trait::async_trait]
impl ExpertRegistry for ExpertRegistryImpl {
    async fn select(&mut self, eid: String) -> EKResult<Channel> {
        self.inner_select(eid).await
    }
}

impl ExpertRegistryImpl {
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
            let hostname = node.hostname.clone();
            // TODO: hard code port
            let port = 51234;
            let end = Channel::from_shared(format!("http://{}:{}", hostname, port))
                .map_err(|e| EKError::InvalidInput(format!("invalid url for gRPC: {}", e)))?;
            let ch = end.connect().await?;
            let meta = ChannelMeta { ch };
            self.channels.insert(eid.clone(), vec![meta]);
        }
        let res = self.channels.get(&eid).ok_or(EKError::NotFound(format!(
            "No channel found for expert {}",
            eid
        )))?;
        if res.is_empty() {
            return Err(EKError::NotFound(format!(
                "No channel found for expert {}",
                eid
            )));
        }
        Ok(res[0].ch.clone())
    }
    pub fn update() {}
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
