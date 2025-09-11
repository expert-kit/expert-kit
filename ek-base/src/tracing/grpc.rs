use http::{Request, Response, Uri};
use opentelemetry::propagation::{Extractor, Injector};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use tracing::{Instrument, Level};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use tonic::{body::Body, transport::Channel};
use tower::{Layer, Service};

#[derive(Default, Debug, Clone)]
pub struct GRPCClientLayer;

#[derive(Debug, Clone)]
pub struct OTelGrpcClientMiddleware {
    inner: Channel,
}

impl OTelGrpcClientMiddleware {
    pub fn new(inner: Channel) -> Self {
        OTelGrpcClientMiddleware { inner }
    }
}

impl Service<Request<Body>> for OTelGrpcClientMiddleware {
    type Response = Response<Body>;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx).map_err(Into::into)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let mut req = req;
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);

        let span = span_from_req_cli(&req);
        let ctx = span.context();
        let mut injector = HeaderInjector(req.headers_mut());
        opentelemetry::global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&ctx, &mut injector);
        });
        let _enter = span.enter();
        Box::pin(
            async move {
                let resp = inner.call(req).await?;
                Ok(resp)
            }
            .in_current_span(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct OTelGrpcServerMiddleware<S> {
    inner: S,
}
type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

impl<S> OTelGrpcServerMiddleware<S> {
    pub fn new(inner: S) -> Self {
        Self { inner }
    }
}

impl<S, ReqBody, ResBody> Service<http::Request<ReqBody>> for OTelGrpcServerMiddleware<S>
where
    S: Service<http::Request<ReqBody>, Response = http::Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ReqBody: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = BoxFuture<'static, Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: http::Request<ReqBody>) -> Self::Future {
        let clone = self.inner.clone();
        let mut inner = std::mem::replace(&mut self.inner, clone);
        let span = {
            let span = span_from_req_srv(&req);
            let extractor = HeaderExtractor(req.headers());
            let ctx = opentelemetry::global::get_text_map_propagator(|propagator| {
                propagator.extract(&extractor)
            });
            span.set_parent(ctx);
            span
        };

        Box::pin(async move {
            // Do extra async work here...
            let _enter = span.enter();
            let response = inner.call(req).await?;

            Ok(response)
        })
    }
}
#[derive(Debug, Clone, Default)]
pub struct OTelGrpcServerLayer {}

impl<S> Layer<S> for OTelGrpcServerLayer {
    type Service = OTelGrpcServerMiddleware<S>;

    fn layer(&self, service: S) -> Self::Service {
        OTelGrpcServerMiddleware { inner: service }
    }
}

pub fn span_from_req_cli<B>(req: &http::Request<B>) -> tracing::Span {
    let (service, method) = extract_service_method(req.uri());
    tracing::span!(
        Level::INFO,
        "grpc.request",
        otel.name = format!("{service}/{method}"),
        otel.kind = ?opentelemetry::trace::SpanKind::Client,
        // otel.status_code = Empty, // to set on response
        // trace_id = Empty, // to set on response
        // request_id = Empty, // to set
        // exception.message = Empty, // to set on response
    )
}
pub fn span_from_req_srv<B>(req: &http::Request<B>) -> tracing::Span {
    let (service, method) = extract_service_method(req.uri());
    tracing::span!(
        Level::INFO,
        "grpc.request",
        otel.name = format!("{service}/{method}"),
        otel.kind = ?opentelemetry::trace::SpanKind::Server,
        // otel.status_code = Empty, // to set on response
        // trace_id = Empty, // to set on response
        // request_id = Empty, // to set
        // exception.message = Empty, // to set on response
    )
}

pub fn extract_service_method(uri: &Uri) -> (&str, &str) {
    let path = uri.path();
    let mut parts = path.split('/').filter(|x| !x.is_empty());
    let service = parts.next().unwrap_or_default();
    let method = parts.next().unwrap_or_default();
    (service, method)
}

pub struct HeaderInjector<'a>(pub &'a mut http::HeaderMap);

impl Injector for HeaderInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        if let Ok(name) = http::header::HeaderName::from_bytes(key.as_bytes())
            && let Ok(val) = http::header::HeaderValue::from_str(&value)
        {
            self.0.insert(name, val);
        }
    }
}

pub struct HeaderExtractor<'a>(pub &'a http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .map(http::HeaderName::as_str)
            .collect::<Vec<_>>()
    }
}
