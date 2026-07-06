use std::{fmt::Debug, sync::Arc};

use ek_base::{config::get_ek_settings, utils::Defers};
use tokio::sync::{Mutex, mpsc};
use tracing::Instrument;

use crate::{
    controller::{
        executor::{Executor, get_executor},
        metrics::METRIC_CONTROLLER_LAYER,
    },
    observability::{StageTimer, TraceLabels, child_span_from_traceparent, extract_traceparent},
    proto::ek::worker::v1::{self, computation_service_server::ComputationService},
};

pub struct ComputationProxyServiceImpl {
    executor: Arc<Mutex<dyn Executor + Send>>,
}
impl Debug for ComputationProxyServiceImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationProxyServiceImpl").finish()
    }
}

#[async_trait::async_trait]
impl ComputationService for ComputationProxyServiceImpl {
    async fn forward(
        &self,
        request: tonic::Request<v1::ForwardReq>,
    ) -> Result<tonic::Response<v1::ForwardResp>, tonic::Status> {
        self.inner_controller_forward(request).await
    }
}

impl ComputationProxyServiceImpl {
    async fn inner_controller_forward(
        &self,
        request: tonic::Request<v1::ForwardReq>,
    ) -> Result<tonic::Response<v1::ForwardResp>, tonic::Status> {
        let seq_len = request.get_ref().sequences.len();
        log::info!(seq_len; "forward request in controller start");
        let start = std::time::Instant::now();
        let settings = get_ek_settings();
        let traceparent = extract_traceparent(request.metadata());
        let forward_labels = TraceLabels::new("controller", "controller.forward")
            .path("fallback_controller")
            .num_tokens(seq_len);
        let forward_span = child_span_from_traceparent(&forward_labels, traceparent.as_str());
        let _forward_timer = StageTimer::start_with_span(forward_labels, forward_span.clone());

        let cloned_start = start;
        let _d = Defers::defer(Box::new(move || {
            let elapsed = cloned_start.elapsed();
            // TODO: hardcode model name in metric
            METRIC_CONTROLLER_LAYER
                .with_label_values(&[settings.inference.model_name.as_str()])
                .observe(elapsed.as_micros() as f64);
        }));

        let submit_timer = {
            let _entered = forward_span.enter();
            StageTimer::start(
                TraceLabels::new("controller", "controller.submit")
                    .path("fallback_controller")
                    .num_tokens(seq_len),
            )
        };
        let submit_span = submit_timer.span();
        let mut rx = async {
            let mut lg = self.executor.lock().await;
            lg.submit(request.get_ref()).await
        }
        .instrument(submit_span)
        .await?;
        drop(submit_timer);

        let exec_bg = self.executor.clone();
        let (err_tx, mut err_rx) = mpsc::channel(1);

        tokio::spawn(
            async move {
                let mut lg = exec_bg.lock().await;
                let res = lg.exec().await;
                if let Err(err) = res {
                    log::error!("executor error: {err}");
                    err_tx.send(err).await.unwrap();
                }
            }
            .instrument(forward_span.clone()),
        );

        let wait_timer = {
            let _entered = forward_span.enter();
            StageTimer::start(
                TraceLabels::new("controller", "controller.wait_result")
                    .path("fallback_controller")
                    .num_tokens(seq_len),
            )
        };
        let wait_span = wait_timer.span();
        tokio::select! {
            err = err_rx.recv().instrument(wait_span.clone()) => {
                if let Some(err) = err {
                    log::error!("executor error: {err:?}");
                    return Err(tonic::Status::internal(format!("executor error: {err:?}")));
                }
                // err_tx dropped: exec task completed without error, wait for result
            }
            res = rx.recv().instrument(wait_span) => {
                drop(wait_timer);
                let elapsed_ms = start.elapsed().as_millis();
                log::info!(elapsed_ms; "forward request in controller done");
                if let Some(resp) = res {
                    return Ok(tonic::Response::new(resp.as_ref().clone()));
                } else {
                    return Err(tonic::Status::internal("forward error: no data"));
                }
            }
        }
        drop(wait_timer);
        let elapsed_ms = start.elapsed().as_millis();
        log::info!(elapsed_ms; "forward request in controller done");
        match rx.recv().await {
            Some(resp) => Ok(tonic::Response::new(resp.as_ref().clone())),
            None => Err(tonic::Status::internal("forward error: no data")),
        }
    }
}

impl Default for ComputationProxyServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputationProxyServiceImpl {
    pub fn new() -> Self {
        Self {
            executor: get_executor(),
        }
    }
}
