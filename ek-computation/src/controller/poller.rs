use std::{sync::Arc, time::Duration};

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

use super::{
    dispatcher::{DISPATCHER, Dispatcher},
    routing_broadcaster::RoutingBroadcaster,
    scheduler::WorkerScheduler,
};

#[async_trait]
pub trait StatePoller {
    async fn run(&mut self) -> EKResult<()>;
}

pub struct StatePollerImpl {
    broadcaster: Arc<RoutingBroadcaster>,
    scheduler: Arc<WorkerScheduler>,
}

#[async_trait]
impl StatePoller for StatePollerImpl {
    async fn run(&mut self) -> EKResult<()> {
        log::info!("state poller started");
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            log::info!("state poller tick");
            let r = self.poll_state().await;
            if let Err(e) = r {
                log::error!("state poller error: {e}");
            }
            interval.tick().await;
        }
    }
}

impl StatePollerImpl {
    pub fn new(broadcaster: Arc<RoutingBroadcaster>, scheduler: Arc<WorkerScheduler>) -> Self {
        StatePollerImpl {
            broadcaster,
            scheduler,
        }
    }

    /// Polls the state of the system, fetching nodes and their associated experts,
    async fn poll_state(&mut self) -> EKResult<()> {
        let mut conn = pool::POOL.get().await?;
        let settings = get_ek_settings();

        // Fetch instance by name in settings
        let instance = schema::instance::table
            .filter(schema::instance::name.eq(settings.inference.instance_name.clone()))
            .first::<models::Instance>(&mut conn)
            .await?;

        // Fetch all nodes
        let nodes = schema::node::table
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
        log::info!(nodes_count; "polling nodes");
        lg.update(node_with_expert.clone()).await;
        drop(lg); // Release dispatcher lock before updating routing

        // Update routing table
        self.update_routing(node_with_expert).await?;

        Ok(())
    }

    /// Update routing table based on current state
    async fn update_routing(&self, node_with_experts: Vec<NodeWithExperts>) -> EKResult<()> {
        use std::collections::HashMap;
        use crate::proto::ek::control::v1::WorkerEndpoint;

        // Build map of expert_id → WorkerEndpoint
        let mut routing_updates: HashMap<String, WorkerEndpoint> = HashMap::new();

        for nwe in node_with_experts {
            // Extract worker info from node config
            let node_addr = nwe
                .node
                .config
                .get("addr")
                .and_then(|a| a.as_str())
                .unwrap_or("unknown")
                .to_string();

            let channel = nwe
                .node
                .config
                .get("channel")
                .and_then(|c| c.as_str())
                .unwrap_or("grpc")
                .to_string();

            let rdma_tcp_port = nwe
                .node
                .config
                .get("rdma_tcp_port")
                .and_then(|p| p.as_u64())
                .unwrap_or(0) as u32;

            let device = nwe.node.device.clone();

            // Create WorkerEndpoint
            let endpoint = WorkerEndpoint {
                grpc_addr: node_addr.clone(),
                channel: channel.clone(),
                rdma_tcp_port,
                shm_queue_prefix: nwe.node.hostname.clone(), // Use hostname as queue prefix
                device,
            };

            // For each expert on this node
            for expert in nwe.experts {
                let expert_id = expert.expert_id;

                // Use scheduler to select best worker if there are multiple replicas
                // For now, we'll just use the first one we encounter
                // The scheduler will handle replica selection when frontends request routing
                if !routing_updates.contains_key(&expert_id) {
                    routing_updates.insert(expert_id.clone(), endpoint.clone());
                } else {
                    // Expert exists on multiple nodes - use scheduler to pick best one
                    // For now, just keep the first one (scheduler logic can be enhanced later)
                    log::debug!("Expert {} found on multiple nodes, keeping first assignment", expert_id);
                }
            }
        }

        log::info!("Updating routing table with {} experts", routing_updates.len());
        self.broadcaster.batch_update(routing_updates).await;

        Ok(())
    }
}

pub fn start_poll(broadcaster: Arc<RoutingBroadcaster>, scheduler: Arc<WorkerScheduler>) {
    let mut poller = StatePollerImpl::new(broadcaster, scheduler);
    tokio::spawn(async move {
        if let Err(e) = poller.run().await {
            log::error!("state poller error {e}");
        }
    });
}
