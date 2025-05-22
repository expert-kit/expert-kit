use std::{sync::Arc, time};

use crate::{
    proto::ek::object::v1::ExpertSlice,
    schema,
    state::{
        io::{StateReader, StateReaderImpl},
        pool::POOL,
    },
};
use tonic::async_trait;

use super::models::{self, NewExpert, NewInstance, NewModel, NewNode};
use diesel::{ExpressionMethods, SelectableHelper, upsert::excluded};
use diesel_async::{AsyncConnection, RunQueryDsl};
use ek_base::error::{EKError, EKResult};
use models::{Expert, Instance, Model, Node};
use once_cell::sync::OnceCell;
use tokio::sync::RwLock;

use super::io::StateWriter;

pub struct StateWriterImpl {}

impl Default for StateWriterImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl StateWriterImpl {
    pub fn new() -> Self {
        Self {}
    }
}

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
    async fn upd_expert_state(&mut self, hostname: &str, state: ExpertSlice) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        let reader = StateReaderImpl {};
        use schema::expert::dsl;
        conn.transaction::<_, EKError, _>(|conn| {
            Box::pin(async move {
                let node = reader
                    .node_by_hostname(hostname)
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
    pub async fn expert_del_by_instance(&self, instance: i32) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::expert::dsl;
        diesel::delete(schema::expert::table)
            .filter(dsl::instance_id.eq(instance))
            .execute(&mut conn)
            .await?;
        Ok(())
    }
    pub async fn node_update_seen(&self, hostname: &str) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::node::dsl;
        diesel::update(schema::node::table)
            .filter(dsl::hostname.eq(hostname))
            .set(dsl::last_seen_at.eq(time::SystemTime::now()))
            .execute(&mut conn)
            .await?;
        Ok(())
    }

    pub async fn expert_upsert(&self, node: NewExpert) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        diesel::insert_into(schema::expert::table)
            .values(node)
            .on_conflict((
                schema::expert::node_id,
                schema::expert::instance_id,
                schema::expert::expert_id,
            ))
            .do_update()
            .set(schema::expert::expert_id.eq(excluded(schema::expert::expert_id)))
            .execute(&mut conn)
            .await?;
        Ok(())
    }

    pub async fn instance_upsert(&self, node: NewInstance) -> EKResult<Instance> {
        let mut conn = POOL.get().await?;
        let res = diesel::insert_into(schema::instance::table)
            .values(node)
            .on_conflict(schema::instance::name)
            .do_update()
            .set(schema::instance::name.eq(excluded(schema::instance::name)))
            .returning(models::Instance::as_returning())
            .get_result(&mut conn)
            .await?;
        Ok(res)
    }
    pub async fn node_upsert(&self, node: NewNode) -> EKResult<Node> {
        let mut conn = POOL.get().await?;
        let res = diesel::insert_into(schema::node::table)
            .values(node)
            .on_conflict(schema::node::hostname)
            .do_update()
            .set((
                schema::node::hostname.eq(excluded(schema::node::hostname)),
                schema::node::config.eq(excluded(schema::node::config)),
            ))
            .returning(models::Node::as_returning())
            .get_result(&mut conn)
            .await?;
        Ok(res)
    }
    pub async fn model_upsert(&self, weight_server: &str, model_name: &str) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        let new_model = NewModel {
            name: model_name.to_string(),
            config: serde_json::json!({
                "weight_server": weight_server,
            }),
        };
        diesel::insert_into(schema::model::table)
            .values(new_model)
            .on_conflict(schema::model::name)
            .do_update()
            .set(schema::model::config.eq(excluded(schema::model::config)))
            .execute(&mut conn)
            .await?;
        Ok(())
    }

    fn expert_slice_to_ids(&self, slice: &ExpertSlice) -> EKResult<Vec<i32>> {
        let mut ids = vec![];
        for x in slice.expert_meta.iter() {
            let id = x
                .tags
                .get("db_id")
                .ok_or(EKError::InvalidInput("db_id not found".into()))?
                .parse::<i32>()
                .map_err(|_| EKError::InvalidInput("db_id not invalid id".into()))?;
            ids.push(id);
        }
        Ok(ids)
    }
    fn expert_slice_to_new_expert(
        &self,
        node_id: i32,
        slice: &ExpertSlice,
    ) -> EKResult<Vec<NewExpert>> {
        let mut res = vec![];
        for x in slice.expert_meta.iter() {
            let new_expert = NewExpert {
                instance_id: 0,
                node_id,
                expert_id: x.id.clone(),
                replica: 0,
                state: serde_json::Value::Null,
            };
            res.push(new_expert);
        }
        Ok(res)
    }
}
pub fn get_state_writer() -> Arc<RwLock<dyn StateWriter + Send + Sync>> {
    static INSTANCE: OnceCell<Arc<RwLock<StateWriterImpl>>> = OnceCell::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = StateWriterImpl {};
        Arc::new(RwLock::new(inner))
    });

    (res.clone()) as _
}
