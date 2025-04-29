use crate::{proto::ek::object::v1::ExpertSlice, schema, state::pool::POOL};

use super::models::{self, NewExpert, NewInstance, NewModel, NewNode};
use diesel::{ExpressionMethods, QueryDsl, SelectableHelper};
use diesel_async::{AsyncConnection, RunQueryDsl};
use ek_base::error::{EKError, EKResult};
use models::{Expert, Instance, Model, Node};
use tonic::async_trait;

#[async_trait]
pub trait StateReader {
    async fn node_by_hostname(&self, hostname: String) -> EKResult<Option<Node>>;
    async fn instance_by_id(&self, id: i32) -> EKResult<Option<Instance>>;
    async fn experts_by_node(&self, node_id: i32) -> EKResult<Vec<Expert>>;
    async fn node_by_expert(&self, expert_id: &String) -> EKResult<Vec<Node>>;
}

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

    async fn upd_expert_state(&mut self, hostname: &String, state: ExpertSlice) -> EKResult<()>;
}

pub struct StateReaderImpl {}

impl StateReaderImpl {
    async fn _node_by_expert(&self, expert_id: &String) -> EKResult<Vec<Node>> {
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
    async fn node_by_expert(&self, expert_id: &String) -> EKResult<Vec<Node>> {
        self._node_by_expert(expert_id).await
    }
    async fn node_by_hostname(&self, hostname: String) -> EKResult<Option<Node>> {
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

pub struct StateWriterImpl {}

#[async_trait]
impl StateWriter for StateWriterImpl {
    async fn add_instance(&mut self, instance: &NewInstance) -> EKResult<Instance> {
        let mut conn = POOL.get().await?;
        let res = diesel::insert_into(schema::instance::table)
            .values(instance)
            .returning(models::Instance::as_returning())
            .get_result(&mut conn)
            .await?;
        Ok(res)
    }
    async fn add_model(&mut self, instance: &NewModel) -> EKResult<Model> {
        let mut conn = POOL.get().await?;
        let res = diesel::insert_into(schema::model::table)
            .values(instance)
            .returning(Model::as_returning())
            .get_result(&mut conn)
            .await?;
        Ok(res)
    }
    async fn add_expert(&mut self, instance: &NewExpert) -> EKResult<Expert> {
        let mut conn = POOL.get().await?;
        diesel::insert_into(schema::expert::table)
            .values(instance)
            .returning(models::Expert::as_returning())
            .get_result(&mut conn)
            .await
            .map_err(EKError::from)
    }
    async fn add_node(&mut self, instance: &NewNode) -> EKResult<Node> {
        let mut conn = POOL.get().await?;
        diesel::insert_into(schema::node::table)
            .values(instance)
            .returning(models::Node::as_returning())
            .get_result(&mut conn)
            .await
            .map_err(EKError::from)
    }
    async fn del_instance(&mut self, id: i32) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::instance::dsl;
        diesel::delete(schema::instance::table)
            .filter(dsl::id.eq(id))
            .execute(&mut conn)
            .await?;
        Ok(())
    }
    async fn del_model(&mut self, id: i32) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::model::dsl;
        diesel::delete(schema::model::table)
            .filter(dsl::id.eq(id))
            .execute(&mut conn)
            .await?;
        Ok(())
    }
    async fn del_expert(&mut self, id: i32) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::expert::dsl;
        diesel::delete(schema::expert::table)
            .filter(dsl::id.eq(id))
            .execute(&mut conn)
            .await?;
        Ok(())
    }
    async fn del_node(&mut self, id: i32) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::node::dsl;
        diesel::delete(schema::node::table)
            .filter(dsl::id.eq(id))
            .execute(&mut conn)
            .await?;
        Ok(())
    }
    async fn upd_expert_state(&mut self, hostname: &String, state: ExpertSlice) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        let reader = StateReaderImpl {};
        use schema::expert::dsl;
        conn.transaction::<_, EKError, _>(|conn| {
            Box::pin(async move {
                let node = reader
                    .node_by_hostname(hostname.clone())
                    .await?
                    .ok_or(EKError::NotFound(format!("node {} not found", hostname)))?;
                let updating_ids = self.expert_slice_to_ids(&state)?;
                let new_experts = self.expert_slice_to_new_expert(node.id, &state)?;
                // delete state of updating experts
                diesel::delete(schema::expert::table)
                    .filter(dsl::id.eq_any(updating_ids))
                    .execute(conn)
                    .await?;
                // insert state of updating experts
                diesel::insert_into(schema::expert::table)
                    .values(new_experts)
                    .execute(conn)
                    .await?;
                Ok(())
            })
        })
        .await?;
        Ok(())
    }
}

impl StateWriterImpl {
    fn expert_slice_to_ids(&self, slice: &ExpertSlice) -> EKResult<Vec<i32>> {
        Ok(vec![])
    }
    fn expert_slice_to_new_expert(
        &self,
        node_id: i32,
        slice: &ExpertSlice,
    ) -> EKResult<Vec<NewExpert>> {
        Ok(vec![])
    }
}
