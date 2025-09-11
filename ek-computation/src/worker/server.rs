use std::time::Instant;

use crate::{
    metrics::{METRIC_WORKER_EXPERT_ACTIVATION, METRIC_WORKER_FORWARD},
    proto::ek::worker::v1::{
        ForwardReq, ForwardResp, computation_service_server::ComputationService,
    },
};
use ek_base::utils::Defers;
use tonic::{Request, Response, Status};
use tracing::instrument;

use super::core::{EKInstanceGateSync, get_instance_gate_sync};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[derive(Debug)]
pub struct BasicExpertImpl {
    gate_sync: &'static EKInstanceGateSync, // For compute operations
}

impl Default for BasicExpertImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicExpertImpl {
    pub fn new() -> Self {
        Self {
            gate_sync: get_instance_gate_sync(),
        }
    }
}

#[tonic::async_trait]
impl ComputationService for BasicExpertImpl {
    #[instrument(skip(self, request))]
    async fn forward(&self, request: Request<ForwardReq>) -> Result<Response<ForwardResp>, Status> {
        let now = Instant::now();
        let exp_id = request.get_ref().sequences[0].experts[0].clone();
        tracing::debug!("[L1 {:?}] exp received!", &exp_id);
        let res = self.inner_forward(request).await;
        tracing::debug!("[L1 {:?}] completed in {:?}", &exp_id, now.elapsed());
        res
    }
}

impl BasicExpertImpl {
    #[inline]
    async fn inner_forward(
        &self,
        request: Request<ForwardReq>,
    ) -> Result<Response<ForwardResp>, Status> {
        tracing::debug!(
            "[L2 {:?}] rpc.forward() request: seq={}",
            request.get_ref().sequences[0].experts[0],
            request.get_ref().sequences.len(),
        );
        let exp_id = request.get_ref().sequences[0].experts[0].clone();
        let start = Instant::now();

        let start_cloned = start;
        let settings = ek_base::config::get_ek_settings();

        // Record metrics
        METRIC_WORKER_EXPERT_ACTIVATION
            .with_label_values(&[
                settings.worker.id.as_str(),
                settings.inference.model_name.as_str(),
                request.get_ref().sequences[0].experts[0].as_str(),
            ])
            .inc_by(request.get_ref().sequences.len() as u64);

        {
            let worker_id = settings.worker.id.as_str();
            let model = settings.inference.model_name.as_str();
            let expert = request.get_ref().sequences[0].experts[0].as_str();
            let count = request.get_ref().sequences.len();
            log::info!(
                worker_id:%,
                model:%,
                expert:%,
                count:%
                ; "expert activation record",
            );
        }

        // Set up deferred metrics collection
        Defers::defer(Box::new(move || {
            let elapsed = start_cloned.elapsed();
            METRIC_WORKER_FORWARD
                .with_label_values(&[
                    settings.worker.id.as_str(),
                    settings.inference.model_name.as_str(),
                ])
                .observe(elapsed.as_micros() as f64);
        }));

        tracing::debug!("[L2 {:?}] sync_gate.forward() start", &exp_id,);

        let forward_now = Instant::now();
        let req_inner = request.into_inner();

        // Use sync gate for compute-intensive operations
        let gate_sync = self.gate_sync;

        // Capture current tracing context for the blocking task
        let cx = tracing::Span::current().context();

        let cx_clone = cx.clone();

        // Run synchronous computation in blocking task
        let res = tokio::task::spawn_blocking(move || {
            let _guard = cx_clone.attach();

            // Perform synchronous forward computation
            gate_sync.forward_sync(req_inner)
        })
        .await
        .map_err(|e| {
            log::error!("blocking task join error {e:?}");
            Status::internal("blocking task error")
        })?
        .map_err(|e| {
            log::error!("forward error {e:?}");
            Status::internal("forward error")
        })?;

        tracing::debug!(
            "[L2 {:?}] sync_gate.forward() end, elapsed {:?}",
            &exp_id,
            forward_now.elapsed(),
        );

        let res = Ok(Response::new(res));
        tracing::debug!(
            "[L2 {:?}] rpc.forward() end with {:?}",
            &exp_id,
            start.elapsed(),
        );

        res
    }
}
