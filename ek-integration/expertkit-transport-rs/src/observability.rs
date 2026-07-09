use std::{
    sync::{
        Mutex, Once,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use opentelemetry::propagation::Injector;
use opentelemetry::{
    KeyValue, propagation::TextMapCompositePropagator, trace::TracerProvider as _,
};
use opentelemetry_sdk::{
    Resource,
    error::OTelSdkError,
    propagation::{BaggagePropagator, TraceContextPropagator},
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
};
use opentelemetry_semantic_conventions::{
    SCHEMA_URL,
    resource::{DEPLOYMENT_ENVIRONMENT_NAME, SERVICE_VERSION},
};
use tonic::metadata::MetadataMap;
use tracing::{Level, Span};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

static INIT: Once = Once::new();
static TRACER_PROVIDER: Mutex<Option<SdkTracerProvider>> = Mutex::new(None);
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);
static EXPERT_CALL_ID: AtomicU64 = AtomicU64::new(1);
static LAYER_ID: AtomicU64 = AtomicU64::new(1);

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

pub fn init_tracing() {
    INIT.call_once(|| {
        let exporter = match opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .build()
        {
            Ok(exporter) => exporter,
            Err(err) => {
                log::warn!("[Tracing] failed to initialize OTLP exporter: {err}");
                return;
            }
        };

        let resource = Resource::builder()
            .with_service_name("frontend")
            .with_schema_url(
                [
                    KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
                    KeyValue::new(DEPLOYMENT_ENVIRONMENT_NAME, "develop"),
                ],
                SCHEMA_URL,
            )
            .build();
        let provider = SdkTracerProvider::builder()
            .with_sampler(Sampler::AlwaysOn)
            .with_id_generator(RandomIdGenerator::default())
            .with_resource(resource)
            .with_batch_exporter(exporter)
            .build();
        let tracer = provider.tracer("expertkit-transport-rs");
        if let Ok(mut tracer_provider) = TRACER_PROVIDER.lock() {
            *tracer_provider = Some(provider.clone());
        } else {
            log::warn!("[Tracing] failed to store tracer provider for shutdown");
        }
        let propagator = TextMapCompositePropagator::new(vec![
            Box::new(BaggagePropagator::new()),
            Box::new(TraceContextPropagator::new()),
        ]);
        opentelemetry::global::set_text_map_propagator(propagator);

        if tracing_subscriber::registry()
            .with(tracing_subscriber::filter::LevelFilter::from_level(
                Level::INFO,
            ))
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .try_init()
            .is_err()
        {
            log::debug!("[Tracing] tracing subscriber already initialized");
        }
    });
}

pub fn flush_tracing() -> Result<(), String> {
    let provider = TRACER_PROVIDER
        .lock()
        .map_err(|err| format!("failed to lock tracer provider: {err}"))?
        .clone();

    if let Some(provider) = provider {
        match provider.force_flush() {
            Ok(()) | Err(OTelSdkError::AlreadyShutdown) => Ok(()),
            Err(err) => Err(format!("failed to flush tracing spans: {err}")),
        }
    } else {
        Ok(())
    }
}

pub fn shutdown_tracing() -> Result<(), String> {
    let provider = TRACER_PROVIDER
        .lock()
        .map_err(|err| format!("failed to lock tracer provider: {err}"))?
        .take();

    if let Some(provider) = provider {
        match provider.shutdown() {
            Ok(()) | Err(OTelSdkError::AlreadyShutdown) => Ok(()),
            Err(err) => Err(format!("failed to shutdown tracing: {err}")),
        }
    } else {
        Ok(())
    }
}

pub fn next_request_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::SeqCst)
}

pub fn next_expert_call_id() -> u64 {
    EXPERT_CALL_ID.fetch_add(1, Ordering::SeqCst)
}

pub fn next_layer_id() -> u64 {
    LAYER_ID.fetch_add(1, Ordering::SeqCst)
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

pub fn inject_current_trace_context(metadata: &mut MetadataMap) {
    let ctx = Span::current().context();
    let mut injector = MetadataInjector(metadata);
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&ctx, &mut injector);
    });
}

pub fn infer_layer_id(expert_ids: &[Vec<String>], fallback: u64) -> u64 {
    expert_ids
        .iter()
        .flatten()
        .find_map(|expert_id| {
            let marker = expert_id.rfind("/l")?;
            let after_layer = &expert_id[marker + 2..];
            let end = after_layer.find("-e").unwrap_or(after_layer.len());
            after_layer[..end].parse::<u64>().ok()
        })
        .unwrap_or(fallback)
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
