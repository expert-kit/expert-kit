use std::{sync::LazyLock, time::Duration};

use diesel::{
    BelongingToDsl, ExpressionMethods, GroupedBy, SelectableHelper,
    query_dsl::methods::{FilterDsl, SelectDsl},
};
use diesel_async::RunQueryDsl;
use ek_base::{config::get_ek_settings, error::EKResult};
use std::time::SystemTime;
use tokio::{
    sync::Notify,
    time::{self},
};
use tonic::async_trait;

use crate::{
    schema,
    state::{
        models::{self, NodeWithExperts},
        pool,
    },
};

use super::dispatcher::{DISPATCHER, Dispatcher};

/// Notify handle for forcing an immediate poller tick.
///
/// When a node is removed, placement state must be refreshed immediately rather
/// than waiting for the next regular interval.
static POLL_NOW: LazyLock<Notify> = LazyLock::new(Notify::new);

/// Request an immediate poller tick after a lifecycle or placement change.
pub fn request_immediate_poll() {
    POLL_NOW.notify_one();
}

#[async_trait]
pub trait StatePoller {
    async fn run(&mut self) -> EKResult<()>;
}

pub struct StatePollerImpl;

#[async_trait]
impl StatePoller for StatePollerImpl {
    async fn run(&mut self) -> EKResult<()> {
        let poller_secs = get_ek_settings()
            .controller
            .fault_detection
            .poller_interval_secs;
        log::info!("state poller started (interval={}s)", poller_secs);
        let mut interval = time::interval(Duration::from_secs(poller_secs));
        loop {
            // Wait for either the regular interval OR an explicit request
            tokio::select! {
                _ = interval.tick() => {},
                _ = POLL_NOW.notified() => {
                    log::debug!("state poller: immediate tick requested");
                    interval.reset(); // avoid double-tick shortly after
                },
            }
            log::debug!("state poller tick");
            let r = self.poll_state().await;
            if let Err(e) = r {
                log::error!("state poller error: {e}");
            }
        }
    }
}

impl StatePollerImpl {
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
        let threshold_secs = settings
            .controller
            .fault_detection
            .node_active_threshold_secs;
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
        log::debug!(nodes_count; "polling nodes");
        lg.update(node_with_expert.clone()).await;
        drop(lg);

        Ok(())
    }
}

pub fn start_poll() {
    let mut poller = StatePollerImpl;
    tokio::spawn(async move {
        if let Err(e) = poller.run().await {
            log::error!("state poller error {e}");
        }
    });
}
