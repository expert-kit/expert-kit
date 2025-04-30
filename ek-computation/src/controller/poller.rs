use std::time::Duration;

use diesel::{BelongingToDsl, GroupedBy, SelectableHelper, query_dsl::methods::SelectDsl};
use diesel_async::RunQueryDsl;
use ek_base::error::{EKError, EKResult};
use tokio::time::{self};
use tonic::async_trait;

use crate::{
    schema,
    state::{
        models::{self, NodeWithExperts},
        pool,
    },
};

use super::dispatcher::{DISPATCHER, Dispatcher};

#[async_trait]
pub trait StatePoller {
    async fn run(&mut self) -> EKResult<()>;
}

pub struct StatePollerImpl {}

#[async_trait]
impl StatePoller for StatePollerImpl {
    async fn run(&mut self) -> EKResult<()> {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            let r = self.poll_state().await;
            if let Err(e) = r {
                log::error!("state poller error {}", e);
            }
            interval.tick().await;
        }
    }
}

impl Default for StatePollerImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl StatePollerImpl {
    pub fn new() -> Self {
        StatePollerImpl {}
    }
    async fn poll_state(&mut self) -> EKResult<()> {
        let mut conn = pool::POOL.get().await.map_err(|_| EKError::DBError())?;
        let nodes = schema::node::table
            .select(models::Node::as_select())
            .load(&mut conn)
            .await?;
        let experts = models::Expert::belonging_to(&nodes)
            .select(models::Expert::as_select())
            .load(&mut conn)
            .await?;
        let node_with_expert = experts
            .grouped_by(&nodes)
            .into_iter()
            .zip(nodes)
            .map(|(e, n)| NodeWithExperts {
                experts: e,
                node: n,
            })
            .collect::<Vec<NodeWithExperts>>();
        let mut lg = DISPATCHER.lock().await;
        lg.update(node_with_expert).await;

        Ok(())
    }
}
