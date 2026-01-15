use std::{
    sync::{Arc, OnceLock},
    time::Instant,
};

use once_cell::sync::OnceCell;
use parking_lot::Once;
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use opentelemetry::{
    KeyValue, propagation::TextMapCompositePropagator, trace::TracerProvider as _,
};

use tokio::runtime::Runtime;
use tracing::Level;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use opentelemetry_sdk::{
    Resource,
    propagation::{BaggagePropagator, TraceContextPropagator},
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
};
use opentelemetry_semantic_conventions::{
    SCHEMA_URL,
    resource::{DEPLOYMENT_ENVIRONMENT_NAME, SERVICE_VERSION},
};

const DEFAULT_THREAD_NUM: usize = 16;
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new(); // for graceful shutdown
static TRACING_INIT: Once = Once::new();
static SVC_NAME: &str = "tracing client";
static TOKIO_RT: OnceCell<Runtime> = OnceCell::new();

/// Manually shutdown the tracer provider. Batch exporter starts a thread
/// which might be unintentionally closed after the main thread ends.
/// Thus, we need to manually shut it down first at the end of main function
#[pyfunction]
pub fn shutdown_tracer_provider() {
    if let Some(tp) = TRACER_PROVIDER.get() {
        let _ = tp.shutdown();
    }
}

/// PyExpertKitTracer defines a pythonic decorator that traces the whole process of a target function.
///
/// The rules are as follows:
/// * Except for `op_name`, any positional and keyword-only arguments are not allowed.
/// * if keyword-only argument `op_name` is provided, use it as the span name.
/// e.g. `@tracer(op_name = "xxx")`
/// * otherwise, use the target function name.
/// e.g. `@tracer` or `@tracer()`
///
/// Under the hood, the struct is imitating the behavior of this python decorator:
/// ```python
/// def tracer(func: Callable | None = None, *, op_name: str | None = None):
///     def outer(f: Callable) -> Callable:
///         span_name = op_name if op_name is not None else f.__name__
///         def inner(*args, **kwargs):
///             # Before calling the function, start a span with name `span_name`
///
///             # Call the function
///             res = f(*args, **kwargs)
///
///             return res
///         return inner
///             
///     return outer if func is None else outer(func)
/// ```
/// To implement the outer/inner function like what python does when defining the decorator,
/// we need to add both OuterWrapper and InnerWrapper class to acheive the same effect.
///
/// # Example
/// ```python
/// from expertkit_transport import ExpertKitTracer, shutdown_tracer_provider
/// tracer = ExpertKitTracer()   # initialize tracing subscriber for Jaeger WebUI
///
/// @tracer
/// def forward():
///     pass
///
/// @tracer(op_name="Attention Forward")  # override the operation name of Jaeger WebUI
/// def attn_forward():
///     pass
///
/// if __name__ == "__main__":
///     # At the end, manually call shutdown_tracer_provider
///     shutdown_tracer_provider()
/// ```
#[pyclass]
pub struct PyExpertKitTracer;

/// Outer wrapper of `PyExpertKitTracer` decorator. It is responsible for parsing the keyword-only case:
/// e.g. `@tracer(op_name = "xxx")` or `@tracer()`
#[pyclass]
struct TracerOuterWrapper {
    op_name: Option<Arc<str>>,
}

/// Inner wrapper of `PyExpertKitTracer` decorator. It is responsible for adding tracing span
/// and profiling the function under the decorator
#[pyclass]
struct TracerInnerWrapper {
    func: PyObject,
    op_name: Arc<str>,
}

#[pymethods]
impl PyExpertKitTracer {
    #[new]
    fn new() -> PyResult<Self> {
        let start_time = Instant::now();
        if env_logger::try_init().is_ok() {
            log::info!("Logger initialized");
        }
        // Tokio runtime singleton
        let runtime = TOKIO_RT.get_or_try_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(DEFAULT_THREAD_NUM)
                .enable_all()
                .build()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create runtime: {}",
                        e
                    ))
                })
        })?;

        runtime.block_on(async {
            // This will only be called once (like a singleton)
            TRACING_INIT.call_once(|| init_tracing_subscriber(SVC_NAME));
        });

        log::debug!("Init takes {:?}", start_time.elapsed());

        Ok(PyExpertKitTracer {})
    }

    #[pyo3(signature = (func = None, *, op_name = None))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        func: Option<PyObject>,
        op_name: Option<String>,
    ) -> PyResult<PyObject> {
        if func.is_none() {
            // Keyword-only parameter
            // Func not shown yet, deal with op_name first with outer wrapper
            // @tracer(op_name=None) or @tracer(op_name="xxx")
            let wrapper = TracerOuterWrapper {
                op_name: op_name.map(Arc::from),
            };

            return Ok(wrapper.into_py(py));
        }

        let func_obj = func.unwrap();
        // Positional arguments (value passed without keyword) is not allowed
        // e.g. tracer("foo")
        if func_obj.extract::<String>(py).is_ok() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Positional argument is not allowed",
            ));
        }

        // First argument is a function, use inner wrapper
        let func_name = func_obj.getattr(py, "__name__")?.extract::<String>(py)?;
        let wrapper = TracerInnerWrapper {
            func: func_obj,
            op_name: Arc::from(func_name),
        };

        Ok(wrapper.into_py(py))
    }
}
#[pymethods]
impl TracerOuterWrapper {
    fn __call__<'py>(&self, py: Python<'py>, func: PyObject) -> PyResult<TracerInnerWrapper> {
        // The operation name is the function name by default
        // Will be overrided by `maybe_op_name` if the user provides it
        let op_name = match &self.op_name {
            Some(name) => name.clone(),
            None => Arc::from(func.getattr(py, "__name__")?.extract::<String>(py)?),
        };

        Ok(TracerInnerWrapper { func, op_name })
    }
}

#[pymethods]
impl TracerInnerWrapper {
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &'py PyTuple,
        kwargs: Option<&'py PyDict>,
    ) -> PyResult<PyObject> {
        log::debug!("Start Tracing on {}", self.op_name);

        // Start a span to profile the process
        let start0 = Instant::now();
        // NOTE: Have to first define a static span name (`span_name`),
        // then use otel.name to override the span name
        // https://docs.rs/tracing-opentelemetry/latest/tracing_opentelemetry/#special-fields
        let span = tracing::span!(Level::INFO, "span_name", "otel.name" = %self.op_name);
        log::debug!("Calling span: {:?}", start0.elapsed());

        // Get the guard when entering the span
        let start1 = Instant::now();
        let _guard = span.enter();
        log::debug!("Getting guard: {:?}", start1.elapsed());

        // Call the function
        log::debug!("Before calling function: {:?}", start0.elapsed());
        let res = self.func.call(py, args, kwargs);

        log::debug!("End Tracing on {}", self.op_name);

        res
    }
}

// Copied from `ek-cli/src/main.rs`
fn init_tracer_provider(svc_name: &'static str) -> SdkTracerProvider {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()
        .unwrap();

    let provider = SdkTracerProvider::builder()
        // Customize sampling strategy
        .with_sampler(Sampler::AlwaysOn)
        // If export trace to AWS X-Ray, you can use XrayIdGenerator
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource(svc_name))
        // .with_batch_exporter(exporter)
        .with_batch_exporter(exporter)
        .build();

    // Set global tracer provider for graceful shuwdown at the end
    let _ = TRACER_PROVIDER.set(provider.clone());

    let baggage_propagator = BaggagePropagator::new();
    let trace_context_propagator = TraceContextPropagator::new();
    let composite_propagator = TextMapCompositePropagator::new(vec![
        Box::new(baggage_propagator),
        Box::new(trace_context_propagator),
    ]);
    opentelemetry::global::set_text_map_propagator(composite_propagator);
    provider
}

fn init_tracing_subscriber(svc_name: &'static str) {
    let tracer_provider = init_tracer_provider(svc_name);
    let tracer = tracer_provider.tracer("tracing-otel-subscriber");
    tracing_subscriber::registry()
        .with(tracing_subscriber::filter::LevelFilter::from_level(
            Level::INFO,
        ))
        // .with(
        //     tracing_subscriber::fmt::layer()
        //         .with_thread_ids(true)
        //         .with_span_events(FmtSpan::NONE),
        // )
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .init();
}

fn resource(cmd: &'static str) -> Resource {
    Resource::builder()
        .with_service_name(cmd)
        .with_schema_url(
            [
                KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
                KeyValue::new(DEPLOYMENT_ENVIRONMENT_NAME, "develop"),
            ],
            SCHEMA_URL,
        )
        .build()
}
