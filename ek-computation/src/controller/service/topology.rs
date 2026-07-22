//! Stream ready expert routes to Frontends.

use std::{pin::Pin, sync::Arc};

use tokio::sync::mpsc;
use tokio_stream::{Stream, wrappers::ReceiverStream};
use tonic::{Request, Response, Status};

use crate::{
    controller::{
        runtime_state::ControllerRuntimeState, service::instance::DefaultInstanceResolver,
    },
    proto::ek::control::v2::{
        TopologyMessage, WatchTopologyRequest, topology_service_server::TopologyService,
    },
};

use super::status::state_status;

const TOPOLOGY_STREAM_BUFFER: usize = 16;

#[derive(Clone)]
pub struct TopologyServiceImpl {
    state: ControllerRuntimeState,
    instance_resolver: Arc<dyn DefaultInstanceResolver>,
}

impl TopologyServiceImpl {
    pub fn new(
        state: ControllerRuntimeState,
        instance_resolver: Arc<dyn DefaultInstanceResolver>,
    ) -> Self {
        Self {
            state,
            instance_resolver,
        }
    }
}

#[tonic::async_trait]
impl TopologyService for TopologyServiceImpl {
    type WatchTopologyStream =
        Pin<Box<dyn Stream<Item = Result<TopologyMessage, Status>> + Send + 'static>>;

    async fn watch_topology(
        &self,
        request: Request<WatchTopologyRequest>,
    ) -> Result<Response<Self::WatchTopologyStream>, Status> {
        let request = request.into_inner();
        if request.instance_id == 0 {
            return Err(Status::invalid_argument("instance_id must be positive"));
        }
        self.instance_resolver.resolve(request.instance_id).await?;
        let newest = self.state.topology_version(request.instance_id).await;
        if request.current_version > newest {
            return Err(Status::invalid_argument(format!(
                "requested topology version {} is newer than current version {newest}",
                request.current_version
            )));
        }
        let state = self.state.clone();
        let mut changed = state.subscribe();
        let (sender, receiver) = mpsc::channel(TOPOLOGY_STREAM_BUFFER);
        tokio::spawn(async move {
            let mut installed_version = request.current_version;
            let mut sent_initial = false;
            loop {
                let newest = state.topology_version(request.instance_id).await;
                if !sent_initial || installed_version < newest {
                    let messages = match state
                        .topology_messages(request.instance_id, installed_version)
                        .await
                    {
                        Ok(messages) => messages,
                        Err(error) => {
                            let _ = sender.send(Err(state_status(error))).await;
                            return;
                        }
                    };
                    for message in messages {
                        if sender.send(Ok(message)).await.is_err() {
                            return;
                        }
                    }
                    installed_version = newest;
                    sent_initial = true;
                    continue;
                }
                if changed.changed().await.is_err() {
                    return;
                }
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use async_trait::async_trait;
    use tokio_stream::StreamExt;

    use super::*;
    use crate::{
        controller::service::instance::ResolvedDefaultInstance,
        proto::ek::control::v2::topology_message,
    };

    struct FakeInstanceResolver;

    #[async_trait]
    impl DefaultInstanceResolver for FakeInstanceResolver {
        async fn resolve(
            &self,
            requested_instance_id: u64,
        ) -> Result<ResolvedDefaultInstance, Status> {
            if requested_instance_id != 0 && requested_instance_id != 7 {
                return Err(Status::failed_precondition(
                    "requested instance does not match Controller default",
                ));
            }
            Ok(ResolvedDefaultInstance {
                instance_id: 7,
                model_name: "model".to_owned(),
                instance_name: "default".to_owned(),
            })
        }
    }

    fn instance_resolver() -> Arc<dyn DefaultInstanceResolver> {
        Arc::new(FakeInstanceResolver)
    }

    #[tokio::test]
    async fn topology_service_sends_an_initial_empty_snapshot() {
        let state = ControllerRuntimeState::new(8);
        let service = TopologyServiceImpl::new(state, instance_resolver());
        let response = service
            .watch_topology(Request::new(WatchTopologyRequest {
                instance_id: 7,
                current_version: 0,
            }))
            .await
            .unwrap();
        let mut stream = response.into_inner();
        let message = tokio::time::timeout(Duration::from_millis(100), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let Some(topology_message::Message::Snapshot(snapshot)) = message.message else {
            panic!("expected a snapshot");
        };
        assert_eq!(snapshot.topology_version, 0);
        assert_eq!(snapshot.part_count, 1);
        assert!(snapshot.routes.is_empty());
    }

    #[tokio::test]
    async fn topology_rejects_a_nondefault_instance() {
        let service = TopologyServiceImpl::new(ControllerRuntimeState::new(8), instance_resolver());

        let error = match service
            .watch_topology(Request::new(WatchTopologyRequest {
                instance_id: 8,
                current_version: 0,
            }))
            .await
        {
            Ok(_) => panic!("nondefault instance unexpectedly opened a topology stream"),
            Err(error) => error,
        };

        assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    }
}
