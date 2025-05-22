use crate::{proto::ek::object::v1::ExpertSlice, schema, state::pool::POOL};

use super::models::{self, NewExpert, NewInstance, NewModel, NewNode};
use diesel::{ExpressionMethods, QueryDsl, SelectableHelper};
use diesel_async::RunQueryDsl;
use ek_base::error::EKResult;
use models::{Expert, Instance, Model, Node};
use tonic::async_trait;

#[allow(dead_code)]
#[async_trait]
pub trait StateReader {
    async fn node_by_hostname(&self, hostname: &str) -> EKResult<Option<Node>>;
    async fn instance_by_id(&self, id: i32) -> EKResult<Option<Instance>>;
    async fn experts_by_node(&self, node_id: i32) -> EKResult<Vec<Expert>>;
    async fn node_by_expert(&self, expert_id: &str) -> EKResult<Vec<Node>>;
}

#[allow(dead_code)]
#[async_trait]
pub trait StateWriter {
    async fn add_instance(&mut self, instance: &NewInstance) -> EKResult<Instance>;
    async fn add_model(&mut self, instance: &NewModel) -> EKResult<Model>;
    async fn add_expert(&mut self, instance: &NewExpert) -> EKResult<Expert>;
    async fn add_node(&mut self, instance: &NewNode) -> EKResult<Node>;

    async fn del_instance(&mut self, id: i32) -> EKResult<()>;
    async fn del_model(&mut self, id: i32) -> EKResult<()>;
    async fn del_expert(&mut self, id: i32) -> EKResult<()>;
    async fn del_node(&mut self, id: i32) -> EKResult<()>;

    async fn upd_expert_state(&mut self, hostname: &str, state: ExpertSlice) -> EKResult<()>;
}

pub struct StateReaderImpl {}

impl Default for StateReaderImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl StateReaderImpl {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn active_nodes(&self) -> EKResult<Vec<Node>> {
        let mut conn = POOL.get().await?;
        use schema::node::dsl;
        let th = std::time::SystemTime::now() - std::time::Duration::from_secs(20);
        let res = schema::node::table
            .filter(dsl::last_seen_at.gt(th))
            .select(models::Node::as_select())
            .load(&mut conn)
            .await?;
        Ok(res)
    }

    pub async fn model_by_name(&self, name: &str) -> EKResult<Option<Model>> {
        let mut conn = POOL.get().await?;
        use schema::model::dsl;
        let res = schema::model::table
            .filter(dsl::name.eq(name))
            .select(models::Model::as_select())
            .first(&mut conn)
            .await?;
        Ok(Some(res))
    }

    pub async fn instance_by_name(&self, name: &str) -> EKResult<Option<Instance>> {
        let mut conn = POOL.get().await?;
        use schema::instance::dsl;
        let res = schema::instance::table
            .filter(dsl::name.eq(name))
            .select(models::Instance::as_select())
            .first(&mut conn)
            .await?;
        Ok(Some(res))
    }
    async fn _node_by_expert(&self, expert_id: &str) -> EKResult<Vec<Node>> {
        let mut conn = POOL.get().await?;

        let res = schema::node::table
            .inner_join(schema::expert::table)
            .filter(schema::expert::dsl::expert_id.eq(expert_id))
            .select(Node::as_select())
            .distinct()
            .load(&mut conn)
            .await?;
        Ok(res)
    }
}

#[async_trait]
impl StateReader for StateReaderImpl {
    async fn node_by_expert(&self, expert_id: &str) -> EKResult<Vec<Node>> {
        self._node_by_expert(expert_id).await
    }
    async fn node_by_hostname(&self, hostname: &str) -> EKResult<Option<Node>> {
        let mut conn = POOL.get().await?;
        use schema::node::dsl;
        let res = schema::node::table
            .filter(dsl::hostname.eq(hostname))
            .select(models::Node::as_select())
            .get_result(&mut conn)
            .await?;
        Ok(Some(res))
    }
    async fn instance_by_id(&self, id: i32) -> EKResult<Option<Instance>> {
        let mut conn = POOL.get().await?;
        use schema::instance::dsl;
        let res = schema::instance::table
            .filter(dsl::id.eq(id))
            .select(models::Instance::as_select())
            .first(&mut conn)
            .await?;
        Ok(Some(res))
    }
    async fn experts_by_node(&self, node_id: i32) -> EKResult<Vec<Expert>> {
        let mut conn = POOL.get().await?;
        use schema::expert::dsl;
        let res = schema::expert::table
            .filter(dsl::node_id.eq(node_id))
            .select(models::Expert::as_select())
            .load(&mut conn)
            .await?;
        Ok(res)
    }
}
