use ek_base::{config::get_ek_settings, error::EKResult};
use ek_db::weight_manager::{LocalWeightManager, peer_server};

/// Start a standalone peer weight HTTP server.
/// This serves cached expert weights to peer workers without running any
/// expert computation stack — useful for dedicated cache-tier nodes in
/// multi-node experiments.
pub async fn wm_server_main() -> EKResult<()> {
    let settings = get_ek_settings();
    let wm = LocalWeightManager::new_shared();

    let addr: std::net::SocketAddr = settings
        .weight
        .wm_listen
        .parse()
        .map_err(|e| ek_base::error::EKError::InvalidInput(format!("invalid wm_listen: {e}")))?;

    log::info!("WmServer listening on {addr}");

    let server = peer_server::start_peer_server(wm, &addr)
        .await
        .map_err(ek_base::error::EKError::IoError)?;

    server
        .await
        .map_err(|e| ek_base::error::EKError::IoError(std::io::Error::other(e.to_string())))
}
