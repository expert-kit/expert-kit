use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::Channel;

// Import generated routing proto
use crate::transport::grpc::proto::ek::control::v1::{
    routing_service_client::RoutingServiceClient, GetRoutingReq,
};

/// Routing table client for fetching expert → worker mappings
pub struct RoutingClient {
    controller_addr: String,
    routing_table: Arc<RwLock<HashMap<String, String>>>, // expert_id → worker_addr
    routing_version: Arc<RwLock<u64>>,
    channel: Option<Channel>,
}

impl RoutingClient {
    pub fn new(controller_addr: String) -> Self {
        Self {
            controller_addr,
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            routing_version: Arc::new(RwLock::new(0)),
            channel: None,
        }
    }

    /// Connect to controller
    pub async fn connect(&mut self) -> Result<()> {
        let uri = if self.controller_addr.starts_with("http://")
            || self.controller_addr.starts_with("https://")
        {
            self.controller_addr.clone()
        } else {
            format!("http://{}", self.controller_addr)
        };

        self.channel = Some(Channel::from_shared(uri)?.connect().await?);
        Ok(())
    }

    /// Fetch routing table from controller
    pub async fn fetch_routing(&self, expert_ids: Option<Vec<String>>) -> Result<()> {
        let channel = self
            .channel
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not connected to controller"))?;

        let mut client = RoutingServiceClient::new(channel.clone());

        let is_partial = expert_ids.is_some();
        let req = GetRoutingReq {
            expert_ids: expert_ids.unwrap_or_default(),
        };

        let response = client.get_routing(req).await?;
        let routing_response = response.into_inner();

        let mut table = self.routing_table.write().await;
        let mut version = self.routing_version.write().await;

        if is_partial {
            // Partial update
            table.extend(routing_response.routing);
        } else {
            // Full replace
            *table = routing_response.routing;
        }

        *version = routing_response.version;

        eprintln!(
            "Routing table updated: {} experts, version={}",
            table.len(),
            *version
        );

        Ok(())
    }

    /// Get worker address for an expert
    pub async fn get_worker(&self, expert_id: &str) -> Option<String> {
        let table = self.routing_table.read().await;
        table.get(expert_id).cloned()
    }

    /// Get workers for multiple experts
    pub async fn get_workers(&self, expert_ids: &[String]) -> HashMap<String, Option<String>> {
        let table = self.routing_table.read().await;
        expert_ids
            .iter()
            .map(|id| (id.clone(), table.get(id).cloned()))
            .collect()
    }

    /// Get all routing entries
    pub async fn get_all_routing(&self) -> HashMap<String, String> {
        let table = self.routing_table.read().await;
        table.clone()
    }

    /// Get routing version
    pub async fn get_version(&self) -> u64 {
        let version = self.routing_version.read().await;
        *version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_routing_client_creation() {
        let client = RoutingClient::new("localhost:5002".to_string());
        assert_eq!(client.get_version().await, 0);
        assert!(client.get_all_routing().await.is_empty());
    }

    #[tokio::test]
    async fn test_routing_table_operations() {
        let client = RoutingClient::new("localhost:5002".to_string());

        // Manually populate for testing
        {
            let mut table = client.routing_table.write().await;
            table.insert("expert_1".to_string(), "worker1:50051".to_string());
            table.insert("expert_2".to_string(), "worker2:50051".to_string());
        }

        assert_eq!(
            client.get_worker("expert_1").await,
            Some("worker1:50051".to_string())
        );
        assert_eq!(
            client.get_worker("expert_2").await,
            Some("worker2:50051".to_string())
        );
        assert_eq!(client.get_worker("expert_3").await, None);
    }
}
