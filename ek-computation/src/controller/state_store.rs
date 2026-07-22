//! Durable Controller placement generations and topology-version allocation.

use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use diesel::{ExpressionMethods, QueryDsl, upsert::excluded};
use diesel_async::RunQueryDsl;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::{proto::ek::control::v2::TargetExpert, schema, state::pool::POOL};

/// State required to restart without reusing a protocol counter.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ControllerStoreSnapshot {
    /// Last committed target list for each logical Worker ID.
    pub placements: Vec<PersistedPlacement>,
    /// Last allocated topology version for each model instance.
    pub topology_versions: BTreeMap<u64, u64>,
}

/// One Worker target list committed together with its generation.
#[derive(Clone, Debug, PartialEq)]
pub struct PersistedPlacement {
    /// Stable logical Worker ID.
    pub worker_id: String,
    /// Model instance served by this placement.
    pub instance_id: u64,
    /// Monotonic placement generation sent to the Worker.
    pub generation: u64,
    /// Complete target list for the generation.
    pub targets: Vec<TargetExpert>,
}

/// Failure to load or atomically update Controller protocol state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControllerStoreError(String);

impl ControllerStoreError {
    pub(crate) fn new(error: impl fmt::Display) -> Self {
        Self(error.to_string())
    }
}

impl fmt::Display for ControllerStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ControllerStoreError {}

/// Storage operations that must complete before protocol state becomes visible.
#[async_trait]
pub trait ControllerStateStore: Send + Sync + 'static {
    /// Load the last committed placements and version counters.
    async fn load(&self) -> Result<ControllerStoreSnapshot, ControllerStoreError>;

    /// Atomically replace one complete target list and its generation.
    async fn save_placement(
        &self,
        worker_id: &str,
        instance_id: u64,
        generation: u64,
        targets: &[TargetExpert],
    ) -> Result<(), ControllerStoreError>;

    /// Persist and return a version newer than both stored and current values.
    async fn advance_topology_version(
        &self,
        instance_id: u64,
        current_version: u64,
    ) -> Result<u64, ControllerStoreError>;
}

/// Process-local store used by deterministic Controller state tests.
#[derive(Default)]
pub struct TransientControllerStateStore {
    inner: Mutex<ControllerStoreSnapshot>,
}

impl TransientControllerStateStore {
    /// Create a shareable empty test store.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait]
impl ControllerStateStore for TransientControllerStateStore {
    async fn load(&self) -> Result<ControllerStoreSnapshot, ControllerStoreError> {
        Ok(self.inner.lock().await.clone())
    }

    async fn save_placement(
        &self,
        worker_id: &str,
        instance_id: u64,
        generation: u64,
        targets: &[TargetExpert],
    ) -> Result<(), ControllerStoreError> {
        let mut inner = self.inner.lock().await;
        let placement = PersistedPlacement {
            worker_id: worker_id.to_owned(),
            instance_id,
            generation,
            targets: targets.to_vec(),
        };
        if let Some(existing) = inner
            .placements
            .iter_mut()
            .find(|existing| existing.worker_id == worker_id)
        {
            *existing = placement;
        } else {
            inner.placements.push(placement);
        }
        Ok(())
    }

    async fn advance_topology_version(
        &self,
        instance_id: u64,
        current_version: u64,
    ) -> Result<u64, ControllerStoreError> {
        let mut inner = self.inner.lock().await;
        let stored = inner.topology_versions.entry(instance_id).or_default();
        let next = (*stored)
            .max(current_version)
            .checked_add(1)
            .ok_or_else(|| ControllerStoreError::new("topology version is exhausted"))?;
        *stored = next;
        Ok(next)
    }
}

/// PostgreSQL implementation used by the Controller process.
pub struct PostgresControllerStateStore;

impl PostgresControllerStateStore {
    /// Create the process store using the existing database connection pool.
    pub fn shared() -> Arc<Self> {
        Arc::new(Self)
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct StoredTarget {
    layer_id: u32,
    expert_id: u32,
    target_device: String,
    peer_weight_endpoints: Vec<String>,
}

impl From<&TargetExpert> for StoredTarget {
    fn from(target: &TargetExpert) -> Self {
        Self {
            layer_id: target.layer_id,
            expert_id: target.expert_id,
            target_device: target.target_device.clone(),
            peer_weight_endpoints: target.peer_weight_endpoints.clone(),
        }
    }
}

impl From<StoredTarget> for TargetExpert {
    fn from(target: StoredTarget) -> Self {
        Self {
            layer_id: target.layer_id,
            expert_id: target.expert_id,
            target_device: target.target_device,
            peer_weight_endpoints: target.peer_weight_endpoints,
        }
    }
}

fn encode_targets(targets: &[TargetExpert]) -> Result<serde_json::Value, ControllerStoreError> {
    let stored: Vec<StoredTarget> = targets.iter().map(StoredTarget::from).collect();
    serde_json::to_value(stored).map_err(ControllerStoreError::new)
}

fn decode_targets(value: serde_json::Value) -> Result<Vec<TargetExpert>, ControllerStoreError> {
    serde_json::from_value::<Vec<StoredTarget>>(value)
        .map(|targets| targets.into_iter().map(TargetExpert::from).collect())
        .map_err(ControllerStoreError::new)
}

fn to_database_id(name: &str, value: u64) -> Result<i64, ControllerStoreError> {
    i64::try_from(value)
        .map_err(|_| ControllerStoreError::new(format_args!("{name} exceeds PostgreSQL BIGINT")))
}

fn from_database_id(name: &str, value: i64) -> Result<u64, ControllerStoreError> {
    u64::try_from(value)
        .map_err(|_| ControllerStoreError::new(format_args!("stored {name} is negative")))
}

#[async_trait]
impl ControllerStateStore for PostgresControllerStateStore {
    async fn load(&self) -> Result<ControllerStoreSnapshot, ControllerStoreError> {
        let mut connection = POOL.get().await.map_err(ControllerStoreError::new)?;
        let placement_rows: Vec<(String, i64, i64, serde_json::Value)> =
            schema::controller_v2_placement::table
                .select((
                    schema::controller_v2_placement::worker_id,
                    schema::controller_v2_placement::instance_id,
                    schema::controller_v2_placement::generation,
                    schema::controller_v2_placement::targets,
                ))
                .load(&mut connection)
                .await
                .map_err(ControllerStoreError::new)?;
        let version_rows: Vec<(i64, i64)> = schema::controller_v2_topology::table
            .select((
                schema::controller_v2_topology::instance_id,
                schema::controller_v2_topology::version,
            ))
            .load(&mut connection)
            .await
            .map_err(ControllerStoreError::new)?;

        let placements = placement_rows
            .into_iter()
            .map(|(worker_id, instance_id, generation, targets)| {
                Ok(PersistedPlacement {
                    worker_id,
                    instance_id: from_database_id("instance_id", instance_id)?,
                    generation: from_database_id("placement generation", generation)?,
                    targets: decode_targets(targets)?,
                })
            })
            .collect::<Result<Vec<_>, ControllerStoreError>>()?;
        let topology_versions = version_rows
            .into_iter()
            .map(|(instance_id, version)| {
                Ok((
                    from_database_id("instance_id", instance_id)?,
                    from_database_id("topology version", version)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, ControllerStoreError>>()?;
        Ok(ControllerStoreSnapshot {
            placements,
            topology_versions,
        })
    }

    async fn save_placement(
        &self,
        worker_id: &str,
        instance_id: u64,
        generation: u64,
        targets: &[TargetExpert],
    ) -> Result<(), ControllerStoreError> {
        let instance_id = to_database_id("instance_id", instance_id)?;
        let generation = to_database_id("placement generation", generation)?;
        let targets = encode_targets(targets)?;
        let mut connection = POOL.get().await.map_err(ControllerStoreError::new)?;
        diesel::insert_into(schema::controller_v2_placement::table)
            .values((
                schema::controller_v2_placement::worker_id.eq(worker_id),
                schema::controller_v2_placement::instance_id.eq(instance_id),
                schema::controller_v2_placement::generation.eq(generation),
                schema::controller_v2_placement::targets.eq(&targets),
            ))
            .on_conflict(schema::controller_v2_placement::worker_id)
            .do_update()
            .set((
                schema::controller_v2_placement::instance_id
                    .eq(excluded(schema::controller_v2_placement::instance_id)),
                schema::controller_v2_placement::generation
                    .eq(excluded(schema::controller_v2_placement::generation)),
                schema::controller_v2_placement::targets
                    .eq(excluded(schema::controller_v2_placement::targets)),
            ))
            .execute(&mut connection)
            .await
            .map_err(ControllerStoreError::new)?;
        Ok(())
    }

    async fn advance_topology_version(
        &self,
        instance_id: u64,
        current_version: u64,
    ) -> Result<u64, ControllerStoreError> {
        use diesel::sql_types::BigInt;

        #[derive(diesel::QueryableByName)]
        struct VersionRow {
            #[diesel(sql_type = BigInt)]
            version: i64,
        }

        let instance_id = to_database_id("instance_id", instance_id)?;
        let minimum_next = current_version
            .checked_add(1)
            .ok_or_else(|| ControllerStoreError::new("topology version is exhausted"))?;
        let minimum_next = to_database_id("topology version", minimum_next)?;
        let mut connection = POOL.get().await.map_err(ControllerStoreError::new)?;
        let row: VersionRow = diesel::sql_query(
            "INSERT INTO controller_v2_topology (instance_id, version) \
             VALUES ($1, $2) \
             ON CONFLICT (instance_id) DO UPDATE SET version = \
             GREATEST(controller_v2_topology.version + 1, EXCLUDED.version) \
             RETURNING version",
        )
        .bind::<BigInt, _>(instance_id)
        .bind::<BigInt, _>(minimum_next)
        .get_result(&mut connection)
        .await
        .map_err(ControllerStoreError::new)?;
        from_database_id("topology version", row.version)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> TargetExpert {
        TargetExpert {
            layer_id: 2,
            expert_id: 7,
            target_device: "cuda:0".to_owned(),
            peer_weight_endpoints: vec!["http://peer:8000".to_owned()],
        }
    }

    #[test]
    fn stored_targets_round_trip_without_protobuf_serde() {
        let targets = vec![target()];
        assert_eq!(
            decode_targets(encode_targets(&targets).unwrap()).unwrap(),
            targets
        );
    }

    #[tokio::test]
    async fn transient_store_keeps_generations_and_advances_versions() {
        let store = TransientControllerStateStore::default();
        store
            .save_placement("worker-0", 4, 3, &[target()])
            .await
            .unwrap();
        assert_eq!(store.advance_topology_version(4, 0).await.unwrap(), 1);
        assert_eq!(store.advance_topology_version(4, 0).await.unwrap(), 2);

        let snapshot = ControllerStateStore::load(&store).await.unwrap();
        assert_eq!(snapshot.placements[0].generation, 3);
        assert_eq!(snapshot.topology_versions[&4], 2);
    }
}
