//! Resolution of the single model instance managed by one Controller.

use std::sync::Arc;

use async_trait::async_trait;
use ek_base::error::EKResult;
use tonic::{Request, Response, Status};

use crate::{
    proto::ek::control::v2::{
        ResolveDefaultInstanceRequest, ResolveDefaultInstanceResponse,
        instance_service_server::InstanceService,
    },
    state::{
        io::StateReaderImpl,
        models::{Instance, Model, NewInstance},
        writer::StateWriterImpl,
    },
};

#[derive(Clone, Debug, Eq, PartialEq)]
/// Identity of the one configured model instance returned to runtime clients.
pub struct ResolvedDefaultInstance {
    pub instance_id: u64,
    pub model_name: String,
    pub instance_name: String,
}

#[async_trait]
/// Resolves an omitted or explicit runtime ID against the Controller default.
pub trait DefaultInstanceResolver: Send + Sync + 'static {
    /// Return the default instance or reject a nonmatching explicit ID.
    async fn resolve(&self, requested_instance_id: u64) -> Result<ResolvedDefaultInstance, Status>;
}

#[async_trait]
trait DefaultInstanceStore: Send + Sync + 'static {
    async fn model_by_name(&self, name: &str) -> EKResult<Option<Model>>;

    async fn instance_upsert(&self, instance: NewInstance) -> EKResult<Instance>;
}

struct PostgresDefaultInstanceStore;

#[async_trait]
impl DefaultInstanceStore for PostgresDefaultInstanceStore {
    async fn model_by_name(&self, name: &str) -> EKResult<Option<Model>> {
        StateReaderImpl::new().model_by_name(name).await
    }

    async fn instance_upsert(&self, instance: NewInstance) -> EKResult<Instance> {
        StateWriterImpl::new().instance_upsert(instance).await
    }
}

/// PostgreSQL-backed resolver for the configured model and instance names.
pub struct DatabaseDefaultInstanceResolver {
    model_name: String,
    instance_name: String,
    store: Arc<dyn DefaultInstanceStore>,
}

impl DatabaseDefaultInstanceResolver {
    /// Create a resolver for one immutable Controller configuration.
    pub fn new(model_name: String, instance_name: String) -> Self {
        Self {
            model_name,
            instance_name,
            store: Arc::new(PostgresDefaultInstanceStore),
        }
    }

    #[cfg(test)]
    fn with_store(
        model_name: String,
        instance_name: String,
        store: Arc<dyn DefaultInstanceStore>,
    ) -> Self {
        Self {
            model_name,
            instance_name,
            store,
        }
    }
}

#[async_trait]
impl DefaultInstanceResolver for DatabaseDefaultInstanceResolver {
    async fn resolve(&self, requested_instance_id: u64) -> Result<ResolvedDefaultInstance, Status> {
        let model = self
            .store
            .model_by_name(&self.model_name)
            .await
            .map_err(internal_status)?
            .ok_or_else(|| {
                Status::not_found(format!(
                    "configured model '{}' is not registered; run model upsert first",
                    self.model_name
                ))
            })?;
        let instance = self
            .store
            .instance_upsert(NewInstance {
                model_id: model.id,
                name: self.instance_name.clone(),
            })
            .await
            .map_err(internal_status)?;
        if instance.model_id != model.id {
            return Err(Status::failed_precondition(format!(
                "configured instance '{}' belongs to a different model",
                self.instance_name
            )));
        }
        let instance_id = u64::try_from(instance.id)
            .map_err(|_| Status::internal("stored instance ID is not positive"))?;
        if instance_id == 0 {
            return Err(Status::internal("stored instance ID is not positive"));
        }
        if requested_instance_id != 0 && requested_instance_id != instance_id {
            return Err(Status::failed_precondition(format!(
                "requested instance ID {requested_instance_id} does not match Controller default {instance_id}"
            )));
        }
        Ok(ResolvedDefaultInstance {
            instance_id,
            model_name: self.model_name.clone(),
            instance_name: self.instance_name.clone(),
        })
    }
}

#[derive(Clone)]
/// gRPC adapter exposing resolution on both Controller ports.
pub struct InstanceServiceImpl {
    resolver: Arc<dyn DefaultInstanceResolver>,
}

impl InstanceServiceImpl {
    /// Create the gRPC adapter around a shared resolver.
    pub fn new(resolver: Arc<dyn DefaultInstanceResolver>) -> Self {
        Self { resolver }
    }
}

#[tonic::async_trait]
impl InstanceService for InstanceServiceImpl {
    async fn resolve_default_instance(
        &self,
        request: Request<ResolveDefaultInstanceRequest>,
    ) -> Result<Response<ResolveDefaultInstanceResponse>, Status> {
        let resolved = self
            .resolver
            .resolve(request.into_inner().requested_instance_id)
            .await?;
        Ok(Response::new(ResolveDefaultInstanceResponse {
            instance_id: resolved.instance_id,
            model_name: resolved.model_name,
            instance_name: resolved.instance_name,
        }))
    }
}

fn internal_status(error: impl std::fmt::Display) -> Status {
    Status::internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    use tokio::sync::Mutex;

    use super::*;

    struct FakeStore {
        model: Option<Model>,
        instance: Mutex<Option<Instance>>,
        next_instance: Instance,
        upsert_count: AtomicUsize,
    }

    #[async_trait]
    impl DefaultInstanceStore for FakeStore {
        async fn model_by_name(&self, _name: &str) -> EKResult<Option<Model>> {
            Ok(self.model.clone())
        }

        async fn instance_upsert(&self, instance: NewInstance) -> EKResult<Instance> {
            self.upsert_count.fetch_add(1, Ordering::Relaxed);
            tokio::task::yield_now().await;
            let mut stored = self.instance.lock().await;
            let result = stored.get_or_insert_with(|| Instance {
                id: self.next_instance.id,
                model_id: instance.model_id,
                name: instance.name,
            });
            Ok(result.clone())
        }
    }

    fn model(id: i32) -> Model {
        Model {
            id,
            name: "DeepSeek-V2-Lite-Chat".to_owned(),
            config: serde_json::json!({}),
        }
    }

    fn instance(id: i32, model_id: i32) -> Instance {
        Instance {
            id,
            model_id,
            name: "deepseek-v2-lite-demo".to_owned(),
        }
    }

    fn resolver(store: Arc<FakeStore>) -> DatabaseDefaultInstanceResolver {
        DatabaseDefaultInstanceResolver::with_store(
            "DeepSeek-V2-Lite-Chat".to_owned(),
            "deepseek-v2-lite-demo".to_owned(),
            store,
        )
    }

    #[tokio::test]
    async fn creates_and_reuses_one_default_instance_concurrently() {
        let store = Arc::new(FakeStore {
            model: Some(model(3)),
            instance: Mutex::new(None),
            next_instance: instance(7, 3),
            upsert_count: AtomicUsize::new(0),
        });
        let resolver = Arc::new(resolver(store.clone()));
        let first = {
            let resolver = resolver.clone();
            tokio::spawn(async move { resolver.resolve(0).await.unwrap() })
        };
        let second = {
            let resolver = resolver.clone();
            tokio::spawn(async move { resolver.resolve(0).await.unwrap() })
        };
        let (first, second) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(first, second)
        })
        .await
        .unwrap();

        assert_eq!(first.unwrap().instance_id, 7);
        assert_eq!(second.unwrap().instance_id, 7);
        assert_eq!(store.upsert_count.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn rejects_resolution_before_model_registration() {
        let store = Arc::new(FakeStore {
            model: None,
            instance: Mutex::new(None),
            next_instance: instance(7, 3),
            upsert_count: AtomicUsize::new(0),
        });

        let error = resolver(store).resolve(0).await.unwrap_err();

        assert_eq!(error.code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn rejects_an_instance_bound_to_another_model() {
        let store = Arc::new(FakeStore {
            model: Some(model(3)),
            instance: Mutex::new(Some(instance(7, 4))),
            next_instance: instance(7, 3),
            upsert_count: AtomicUsize::new(0),
        });

        let error = resolver(store).resolve(0).await.unwrap_err();

        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn rejects_an_explicit_nondefault_instance() {
        let store = Arc::new(FakeStore {
            model: Some(model(3)),
            instance: Mutex::new(Some(instance(7, 3))),
            next_instance: instance(7, 3),
            upsert_count: AtomicUsize::new(0),
        });

        let error = resolver(store).resolve(8).await.unwrap_err();

        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
        assert!(
            error
                .message()
                .contains("does not match Controller default 7")
        );
    }
}
