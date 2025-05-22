use std::net::ToSocketAddrs;
use std::thread;

use actix_web::http::header::ContentType;
use actix_web::{App, HttpRequest, HttpResponse, HttpServer, get, middleware, rt};
use ek_base::error::{EKError, EKResult};
use lazy_static::lazy_static;
use prometheus::{
    self, HistogramVec, IntCounterVec, IntGaugeVec, TextEncoder, histogram_opts, labels,
};
use prometheus::{register_histogram_vec, register_int_counter_vec, register_int_gauge_vec};

macro_rules! controller_const_opts {
    () => {
        labels! {"component".to_string() => "controller".to_string()}
    };
}

lazy_static! {

    // Histogram of end-to-end request duration
    pub static ref METRIC_CONTROLLER_LAYER: HistogramVec = register_histogram_vec!(
        histogram_opts!(
            "controller_layer",
            "elapsed time of inferring one layer",
            (1..10).map(|x| (x * 8 * 1000) as f64).collect::<Vec<_>>(),
            controller_const_opts!()
        ),
        &["model"]
    )
    .unwrap();

    // Histogram of intra-request (a.k.a controller to worker) duration.
    pub static ref METRIC_CONTROLLER_INTRA_REQ: HistogramVec = register_histogram_vec!(
        histogram_opts!(
            "controller_intra_request",
            "elapsed time of dispatching one request to worker",
            (0..10).map(|x| (x * 4 * 1000) as f64).collect::<Vec<_>>(),
            controller_const_opts!()
        ),
        &["model"]
    )
    .unwrap();


    // Histogram of expert forward duration.
    pub static ref METRIC_WORKER_FORWARD: HistogramVec = register_histogram_vec!(
        histogram_opts!(
            "worker_forward",
            "elapsed time of dispatching one request to worker",
            (0..10).map(|x| (x * 4 * 1000) as f64).collect::<Vec<_>>()
        ),
        &["worker" , "model"]
    )
    .unwrap();

    // Counter of expert forward activation.
    pub static ref METRIC_WORKER_EXPERT_ACTIVATION: IntCounterVec = register_int_counter_vec!(
        "worker_expert_activation",
        "activation count of expert",
        &["worker","model", "expert"]
    )
    .unwrap();

    // Gauge of expert loading
    pub static ref METRIC_WORKER_EXPERT_LOADING: IntGaugeVec = register_int_gauge_vec!(
        "worker_expert_loading",
        "loading count of expert",
        &["worker", "model", "state"]
    )
    .unwrap();


}

#[get("/metrics")]
async fn export_metrics(_req: HttpRequest) -> HttpResponse {
    let encoder = TextEncoder::new();
    log::info!("export metrics");
    let m = prometheus::gather();
    let body = encoder
        .encode_to_string(&m)
        .map_err(|e| EKError::InvalidInput(e.to_string()))
        .unwrap();

    HttpResponse::Ok()
        .insert_header(ContentType(mime::TEXT_PLAIN))
        .body(body)
}

async fn metrics_listen(addr: &str) -> EKResult<()> {
    let addr = addr.to_socket_addrs().unwrap().collect::<Vec<_>>();
    HttpServer::new(move || {
        App::new()
            .wrap(middleware::Compress::default())
            .service(export_metrics)
    })
    .disable_signals()
    .bind(addr.as_slice())?
    .run()
    .await
    .map_err(EKError::from)
}

pub fn spawn_metrics_server(addr: &str) {
    let addr = addr.to_owned();
    thread::spawn(move || {
        let server_future = metrics_listen(addr.as_str());
        rt::System::new().block_on(server_future)
    });
}
