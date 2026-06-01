use std::sync::Arc;

use actix_web::{HttpResponse, get, web};

use crate::safetensor::ExpertKey;

use super::LocalWeightManager;

/// Actix-web handler: GET /expert/{model}/{layer}/{expert}
/// Serves expert weight bytes from local caches (mem + disk) only,
/// avoiding cascading peer fetches.
#[get("/expert/{model}/{layer}/{expert}")]
async fn get_expert(
    path: web::Path<(String, usize, usize)>,
    wm: web::Data<Arc<LocalWeightManager>>,
) -> HttpResponse {
    let (model, layer, expert) = path.into_inner();
    let key = ExpertKey::new(model, layer, expert);

    match wm.get_expert_local(&key).await {
        Ok(bytes) => HttpResponse::Ok()
            .content_type("application/octet-stream")
            .body((*bytes).clone()),
        Err(_) => HttpResponse::NotFound().finish(),
    }
}

/// Start the peer weight HTTP server on the given address.
/// Returns a server handle that can be awaited.
pub async fn start_peer_server(
    wm: Arc<LocalWeightManager>,
    addr: impl std::net::ToSocketAddrs,
) -> std::io::Result<actix_web::dev::Server> {
    let addrs: Vec<std::net::SocketAddr> = addr.to_socket_addrs()?.collect();
    let wm = web::Data::new(wm);
    let server = actix_web::HttpServer::new(move || {
        actix_web::App::new()
            .app_data(wm.clone())
            .service(get_expert)
    })
    .bind(addrs.as_slice())?
    .run();
    Ok(server)
}
