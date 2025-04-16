use crate::proto::ek::object::v1::ExpertSlice;

use super::models;
use ek_base::error::EKResult;
use models::{Expert, Instance, Model, Node};
use tonic::async_trait;

#[async_trait]
pub trait StateReader {
    async fn node_by_hostname(&self, hostname: String) -> EKResult<Option<Node>>;
    async fn instance_by_id(&self, id: i32) -> EKResult<Option<Instance>>;
    async fn experts_by_node(&self, node_id: i32) -> EKResult<Vec<Expert>>;
}

#[async_trait]
pub trait StateWriter {
    async fn add_instance(&mut self, instance: &Instance) -> EKResult<()>;
    async fn add_model(&mut self, instance: &Model) -> EKResult<()>;
    async fn add_expert(&mut self, instance: &Expert) -> EKResult<()>;
    async fn add_node(&mut self, instance: &Node) -> EKResult<()>;

    async fn del_instance(&mut self, id: i32) -> EKResult<()>;
    async fn del_model(&mut self, id: i32) -> EKResult<()>;
    async fn del_expert(&mut self, id: i32) -> EKResult<()>;
    async fn del_node(&mut self, id: i32) -> EKResult<()>;

    async fn upd_expert_state(&mut self, hostname: &String, state: ExpertSlice) -> EKResult<()>;
}
