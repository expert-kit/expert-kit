use std::time::Duration;

use diesel::{
    BelongingToDsl, ExpressionMethods, GroupedBy, SelectableHelper,
    query_dsl::methods::{FilterDsl, SelectDsl},
};
use diesel_async::RunQueryDsl;
use ek_base::{config::get_ek_settings, error::EKResult};
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
        log::info!("state poller started");
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            log::info!("state poller tick");
            let r = self.poll_state().await;
            if let Err(e) = r {
                log::error!("state poller error: {}", e);
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
        let mut conn = pool::POOL.get().await?;
        let settings = get_ek_settings();

        let instance = schema::instance::table
            .filter(schema::instance::name.eq(settings.inference.instance_name.clone()))
            .first::<models::Instance>(&mut conn)
            .await?;

        let nodes = schema::node::table
            .select(models::Node::as_select())
            .load(&mut conn)
            .await?;
        let experts = models::Expert::belonging_to(&nodes)
            .select(models::Expert::as_select())
            .filter(schema::expert::instance_id.eq(instance.id))
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
        log::info!("state poller got {} nodes", node_with_expert.len());
        lg.update(node_with_expert).await;

        Ok(())
    }
}

pub fn start_poll() {
    let mut poller = StatePollerImpl::new();
    tokio::spawn(async move {
        if let Err(e) = poller.run().await {
            log::error!("state poller error {}", e);
        }
    });
}
