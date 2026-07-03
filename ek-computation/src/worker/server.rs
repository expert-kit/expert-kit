use std::{
    collections::HashMap,
    sync::{
        LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use dashmap::DashMap;

use crate::{
    metrics::{METRIC_WORKER_EXPERT_ACTIVATION, METRIC_WORKER_FORWARD},
    observability::{StageTimer, TraceLabels, child_span_from_traceparent, extract_traceparent},
    proto::ek::worker::v1::{
        ForwardReq, ForwardResp, computation_service_server::ComputationService,
    },
};
use ek_base::utils::Defers;
use tonic::{Request, Response, Status};

/// Per-expert request counters, reset after each heartbeat snapshot.
static REQUEST_COUNTS: LazyLock<DashMap<String, AtomicU64>> = LazyLock::new(DashMap::new);

/// Atomically snapshot all per-expert request counts and reset them to zero.
/// Called once per heartbeat interval; returned map is included in the heartbeat.
pub fn snapshot_and_reset() -> HashMap<String, u64> {
    let mut result = HashMap::new();
    for entry in REQUEST_COUNTS.iter() {
        let count = entry.value().swap(0, Ordering::Relaxed);
        if count > 0 {
            result.insert(entry.key().clone(), count);
        }
    }
    result
}

use super::core::{EKInstanceGateSync, get_instance_gate_sync};

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
        let settings = ek_base::config::get_ek_settings();
        let num_tokens = request.get_ref().sequences.len();
        let traceparent = extract_traceparent(request.metadata());
        let forward_labels = TraceLabels::new("worker", "worker.forward")
            .path("worker")
            .worker(settings.worker.id.as_str())
            .expert_id(exp_id.as_str())
            .num_tokens(num_tokens);
        let forward_span = child_span_from_traceparent(&forward_labels, traceparent.as_str());
        let _forward_timer = StageTimer::start_with_span(forward_labels, forward_span.clone());

        // Increment per-expert counter (reset each heartbeat)
        REQUEST_COUNTS
            .entry(exp_id.clone())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        let start_cloned = start;

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

        let forward_span_for_task = forward_span.clone();
        let queue_wait_timer = {
            let _entered = forward_span.enter();
            StageTimer::start(
                TraceLabels::new("worker", "worker.queue_wait")
                    .path("worker")
                    .worker(settings.worker.id.as_str())
                    .expert_id(exp_id.as_str())
                    .num_tokens(num_tokens),
            )
        };

        // Run synchronous computation in blocking task
        let res = tokio::task::spawn_blocking(move || {
            drop(queue_wait_timer);
            let _entered = forward_span_for_task.enter();
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
