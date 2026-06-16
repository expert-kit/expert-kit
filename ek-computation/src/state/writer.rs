use std::{
    sync::{Arc, OnceLock},
    time,
};

use crate::{
    proto::ek::object::v1::ExpertSlice,
    schema,
    state::{
        io::{StateReader, StateReaderImpl},
        pool::POOL,
    },
};
use tonic::async_trait;

use super::{
    io::StateWriter,
    models::{self, NewExpert, NewInstance, NewModel, NewNode},
};
use diesel::{ExpressionMethods, QueryDsl, SelectableHelper, upsert::excluded};
use diesel_async::{AsyncConnection, RunQueryDsl};
use ek_base::error::{EKError, EKResult};
use models::{Expert, Instance, Model, Node};
use tokio::sync::RwLock;

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
                    .ok_or(EKError::NotFound(format!("node {hostname} not found")))?;
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

    pub async fn delete_experts_by_node(&self, node_id: i32) -> EKResult<usize> {
        let mut conn = POOL.get().await?;
        let count = diesel::delete(
            schema::expert::table.filter(schema::expert::node_id.eq(node_id)),
        )
        .execute(&mut conn)
        .await?;
        Ok(count)
    }

    /// Delete a single expert assignment identified by (node_id, expert_id).
    /// Used by evict-then-place recovery to free a slot for a unique expert.
    pub async fn delete_expert_on_node(&self, node_id: i32, expert_id: &str) -> EKResult<usize> {
        let mut conn = POOL.get().await?;
        use schema::expert::dsl;
        let count = diesel::delete(schema::expert::table)
            .filter(dsl::node_id.eq(node_id))
            .filter(dsl::expert_id.eq(expert_id))
            .execute(&mut conn)
            .await?;
        Ok(count)
    }

    pub async fn delete_experts_by_node_hostname(&self, hostname: &str) -> EKResult<usize> {
        let mut conn = POOL.get().await?;
        let node_ids: Vec<i32> = schema::node::table
            .filter(schema::node::hostname.eq(hostname))
            .select(schema::node::id)
            .load(&mut conn)
            .await?;
        if node_ids.is_empty() {
            return Ok(0);
        }
        let count = diesel::delete(
            schema::expert::table.filter(schema::expert::node_id.eq_any(&node_ids)),
        )
        .execute(&mut conn)
        .await?;
        log::info!("Cleaned {count} stale expert rows for node {hostname}");
        Ok(count)
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

    /// Batch upsert multiple expert assignments in a single query.
    /// Used by recovery to insert thousands of experts at once instead of
    /// one-by-one, reducing DB round-trips from O(n) to O(1).
    pub async fn expert_upsert_batch(&self, experts: Vec<NewExpert>) -> EKResult<usize> {
        if experts.is_empty() {
            return Ok(0);
        }
        let mut conn = POOL.get().await?;
        let count = diesel::insert_into(schema::expert::table)
            .values(&experts)
            .on_conflict((
                schema::expert::node_id,
                schema::expert::instance_id,
                schema::expert::expert_id,
            ))
            .do_update()
            .set(schema::expert::expert_id.eq(excluded(schema::expert::expert_id)))
            .execute(&mut conn)
            .await?;
        Ok(count)
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

    pub async fn del_experts_by_node(&self, node_id: i32, instance_id: i32) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::expert::dsl;
        diesel::delete(schema::expert::table)
            .filter(dsl::node_id.eq(node_id))
            .filter(dsl::instance_id.eq(instance_id))
            .execute(&mut conn)
            .await?;
        Ok(())
    }

    pub async fn deactivate_node(&self, hostname: &str) -> EKResult<()> {
        let mut conn = POOL.get().await?;

        // First, get the node_id for this hostname
        let node_id: Option<i32> = schema::node::table
            .filter(schema::node::hostname.eq(hostname))
            .select(schema::node::id)
            .first(&mut conn)
            .await
            .ok();

        // Reset all experts on this node to "pending" state
        // This ensures they're excluded from routing immediately
        if let Some(nid) = node_id {
            let pending_state = serde_json::json!({"status": "pending"});
            let updated = diesel::update(schema::expert::table)
                .filter(schema::expert::node_id.eq(nid))
                .set(schema::expert::state.eq(pending_state))
                .execute(&mut conn)
                .await?;
            log::info!(
                "Deactivated node {}: reset {} experts to pending state",
                hostname,
                updated
            );
        }

        // Set last seen to zero time and clear config
        use schema::node::dsl;
        diesel::update(schema::node::table)
            .filter(dsl::hostname.eq(hostname))
            .set((
                dsl::last_seen_at.eq(std::time::SystemTime::UNIX_EPOCH),
                dsl::config.eq(serde_json::json!({})),
            ))
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

    pub async fn clear_node_config_by_hostname(&self, hostname: &str) -> EKResult<()> {
        let mut conn = POOL.get().await?;
        use schema::node::dsl;
        diesel::update(schema::node::table)
            .filter(dsl::hostname.eq(hostname))
            .set(dsl::config.eq(serde_json::json!({})))
            .execute(&mut conn)
            .await?;
        Ok(())
    }

    /// Promote a batch of experts from "scheduled" to "pending" state.
    /// Used by progressive assignment to dispatch one stripe at a time.
    pub async fn promote_experts_to_pending(
        &self,
        node_id: i32,
        expert_ids: &[String],
    ) -> EKResult<usize> {
        if expert_ids.is_empty() {
            return Ok(0);
        }
        let mut conn = POOL.get().await?;
        let pending_state = serde_json::json!({"status": "pending"});

        use schema::expert::dsl;
        let count = diesel::update(schema::expert::table)
            .filter(dsl::node_id.eq(node_id))
            .filter(dsl::expert_id.eq_any(expert_ids))
            .set(dsl::state.eq(&pending_state))
            .execute(&mut conn)
            .await?;
        Ok(count)
    }

    /// Update expert load states based on worker heartbeat.
    /// Experts in loaded_experts list → "loaded".
    /// Other experts that are NOT "scheduled" → "pending".
    /// Experts in "scheduled" state are left untouched (not yet dispatched).
    pub async fn update_expert_load_states(
        &self,
        hostname: &str,
        loaded_experts: &[String],
    ) -> EKResult<usize> {
        let mut conn = POOL.get().await?;
        let reader = StateReaderImpl::new();

        // Get node by hostname
        let node = match reader.node_by_hostname(hostname).await? {
            Some(n) => n,
            None => {
                log::warn!("Cannot update expert load states: node {} not found", hostname);
                return Ok(0);
            }
        };

        let loaded_state = serde_json::json!({"status": "loaded"});
        let pending_state = serde_json::json!({"status": "pending"});
        let scheduled_state = serde_json::json!({"status": "scheduled"});

        use schema::expert::dsl;

        // Mark loaded experts
        let loaded_count = if !loaded_experts.is_empty() {
            diesel::update(schema::expert::table)
                .filter(dsl::node_id.eq(node.id))
                .filter(dsl::expert_id.eq_any(loaded_experts))
                .set(dsl::state.eq(&loaded_state))
                .execute(&mut conn)
                .await?
        } else {
            0
        };

        // Mark non-loaded, non-scheduled experts as pending.
        // "scheduled" experts are not yet dispatched and must not be
        // overwritten — they will be promoted to "pending" stripe by
        // stripe during progressive loading.
        let _pending_count = diesel::update(schema::expert::table)
            .filter(dsl::node_id.eq(node.id))
            .filter(diesel::dsl::not(dsl::expert_id.eq_any(loaded_experts)))
            .filter(dsl::state.ne(&scheduled_state))
            .set(dsl::state.eq(&pending_state))
            .execute(&mut conn)
            .await?;

        log::debug!(
            "Updated expert load states for node {}: {} loaded",
            hostname,
            loaded_count
        );

        Ok(loaded_count)
    }
}
pub fn get_state_writer() -> Arc<RwLock<dyn StateWriter + Send + Sync>> {
    static INSTANCE: OnceLock<Arc<RwLock<StateWriterImpl>>> = OnceLock::new();
    let res = INSTANCE.get_or_init(|| {
        let inner = StateWriterImpl {};
        Arc::new(RwLock::new(inner))
    });

    (res.clone()) as _
}
