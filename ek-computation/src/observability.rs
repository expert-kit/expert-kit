use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use opentelemetry::propagation::{Extractor, Injector};
use tonic::metadata::MetadataMap;
use tracing::{Level, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static EXPERT_CALL_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct TraceLabels {
    pub component: &'static str,
    pub stage: &'static str,
    pub path: &'static str,
    pub request_id: u64,
    pub layer_id: u64,
    pub expert_call_id: u64,
    pub expert_id: String,
    pub worker: String,
    pub num_tokens: usize,
}

impl TraceLabels {
    pub fn new(component: &'static str, stage: &'static str) -> Self {
        Self {
            component,
            stage,
            path: "unknown",
            request_id: 0,
            layer_id: 0,
            expert_call_id: 0,
            expert_id: String::new(),
            worker: String::new(),
            num_tokens: 0,
        }
    }

    pub fn path(mut self, path: &'static str) -> Self {
        self.path = path;
        self
    }

    pub fn request_id(mut self, request_id: u64) -> Self {
        self.request_id = request_id;
        self
    }

    pub fn layer_id(mut self, layer_id: u64) -> Self {
        self.layer_id = layer_id;
        self
    }

    pub fn expert_call_id(mut self, expert_call_id: u64) -> Self {
        self.expert_call_id = expert_call_id;
        self
    }

    pub fn expert_id(mut self, expert_id: impl Into<String>) -> Self {
        self.expert_id = expert_id.into();
        self
    }

    pub fn worker(mut self, worker: impl Into<String>) -> Self {
        self.worker = worker.into();
        self
    }

    pub fn num_tokens(mut self, num_tokens: usize) -> Self {
        self.num_tokens = num_tokens;
        self
    }
}

pub struct StageTimer {
    labels: TraceLabels,
    start: Instant,
    span: Span,
}

impl StageTimer {
    pub fn start(labels: TraceLabels) -> Self {
        let span = trace_span(&labels);
        Self::start_with_span(labels, span)
    }

    pub fn start_with_span(labels: TraceLabels, span: Span) -> Self {
        Self {
            labels,
            start: Instant::now(),
            span,
        }
    }

    pub fn span(&self) -> Span {
        self.span.clone()
    }
}

impl Drop for StageTimer {
    fn drop(&mut self) {
        let elapsed_us = self.start.elapsed().as_micros() as u64;
        let labels = &self.labels;
        self.span.in_scope(|| {
            tracing::info!(
                elapsed_us,
                component = labels.component,
                stage = labels.stage,
                path = labels.path,
                request_id = labels.request_id,
                layer_id = labels.layer_id,
                expert_call_id = labels.expert_call_id,
                expert_id = labels.expert_id.as_str(),
                worker = labels.worker.as_str(),
                num_tokens = labels.num_tokens,
                "expert-kit trace stage done",
            );
        });
    }
}

pub fn next_request_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::SeqCst)
}

pub fn next_expert_call_id() -> u64 {
    EXPERT_CALL_ID.fetch_add(1, Ordering::SeqCst)
}

pub fn trace_span(labels: &TraceLabels) -> Span {
    tracing::span!(
        Level::INFO,
        "ek.trace",
        otel.name = labels.stage,
        component = labels.component,
        stage = labels.stage,
        path = labels.path,
        request_id = labels.request_id,
        layer_id = labels.layer_id,
        expert_call_id = labels.expert_call_id,
        expert_id = labels.expert_id.as_str(),
        worker = labels.worker.as_str(),
        num_tokens = labels.num_tokens,
    )
}

pub fn child_span_from_traceparent(labels: &TraceLabels, traceparent: &str) -> Span {
    let span = trace_span(labels);
    if !traceparent.is_empty() {
        let mut headers = HashMap::new();
        headers.insert("traceparent".to_string(), traceparent.to_string());
        let extractor = StringMapExtractor(headers);
        let ctx = opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.extract(&extractor)
        });
        span.set_parent(ctx);
    }
    span
}

pub fn extract_traceparent(metadata: &MetadataMap) -> String {
    metadata
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

pub fn inject_current_trace_context(metadata: &mut MetadataMap) {
    let ctx = Span::current().context();
    let mut injector = MetadataInjector(metadata);
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&ctx, &mut injector);
    });
}

struct MetadataInjector<'a>(&'a mut MetadataMap);

impl Injector for MetadataInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(key.as_bytes())
            && let Ok(value) = value.parse()
        {
            self.0.insert(key, value);
        }
    }
}

struct StringMapExtractor(HashMap<String, String>);

impl Extractor for StringMapExtractor {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }
}
