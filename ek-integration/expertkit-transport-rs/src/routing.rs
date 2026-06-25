use anyhow::Result;
use log::{debug, info, warn};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

// Import generated routing proto
use crate::transport::grpc::proto::ek::control::v1::{
    GetRoutingReq, SubscribeRoutingReq, WorkerEndpoint,
    routing_service_client::RoutingServiceClient, routing_update::ChangeType,
};

use crate::lb::{self, LoadBalancer};

// Configuration constants for WorkerStats EMA
const RTT_ALPHA: f64 = 0.2; // EMA smoothing factor (higher = more reactive)
pub(crate) const DEFAULT_RTT: f64 = 50.0; // Default RTT for new workers (ms)
pub(crate) const DEFAULT_THROUGHPUT_TPM: f64 = 0.5; // Default throughput: tokens per ms (self-corrects quickly)

/// Tracks per-worker statistics for load balancing
pub struct WorkerStats {
    /// Exponential moving average of response times (ms)
    avg_rtt: std::sync::RwLock<f64>,
    /// Currently pending (inflight) requests to this worker
    inflight: AtomicU32,
    /// Exponential moving average of throughput: tokens processed per ms
    /// Updated once per layer via finalize_layer_tpm(), not per-request.
    throughput_tpm: std::sync::RwLock<f64>,
    /// Tokens accumulated in the current layer (reset by finalize_layer_tpm)
    layer_tokens: std::sync::atomic::AtomicU64,
    /// Max RTT observed in the current layer, in microseconds (reset by finalize_layer_tpm)
    layer_max_rtt_us: std::sync::atomic::AtomicU64,
}

impl Default for WorkerStats {
    fn default() -> Self {
        Self {
            avg_rtt: std::sync::RwLock::new(DEFAULT_RTT),
            inflight: AtomicU32::new(0),
            throughput_tpm: std::sync::RwLock::new(DEFAULT_THROUGHPUT_TPM),
            layer_tokens: std::sync::atomic::AtomicU64::new(0),
            layer_max_rtt_us: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

impl WorkerStats {
    pub(crate) fn get_avg_rtt(&self) -> f64 {
        *self.avg_rtt.read().unwrap()
    }

    fn update_rtt(&self, rtt_ms: f64) {
        let mut avg = self.avg_rtt.write().unwrap();
        // Exponential moving average: new_avg = alpha * sample + (1 - alpha) * old_avg
        *avg = RTT_ALPHA * rtt_ms + (1.0 - RTT_ALPHA) * *avg;
    }

    pub(crate) fn get_throughput_tpm(&self) -> f64 {
        *self.throughput_tpm.read().unwrap()
    }

    /// True if this worker's tpm has been measured (not still at DEFAULT).
    pub(crate) fn has_measured_tpm(&self) -> bool {
        (*self.throughput_tpm.read().unwrap() - DEFAULT_THROUGHPUT_TPM).abs() > f64::EPSILON
    }

    /// True if this worker's RTT has been measured (not still at DEFAULT).
    pub(crate) fn has_measured_rtt(&self) -> bool {
        (*self.avg_rtt.read().unwrap() - DEFAULT_RTT).abs() > f64::EPSILON
    }

    #[allow(dead_code)]
    fn update_throughput(&self, token_count: usize, rtt_ms: f64) {
        if rtt_ms > 0.0 && token_count > 0 {
            let sample_tpm = token_count as f64 / rtt_ms;
            let mut tpm = self.throughput_tpm.write().unwrap();
            *tpm = RTT_ALPHA * sample_tpm + (1.0 - RTT_ALPHA) * *tpm;
        }
    }

    pub(crate) fn increment_inflight(&self) {
        self.inflight.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_inflight(&self) {
        self.inflight.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) fn get_inflight(&self) -> u32 {
        self.inflight.load(Ordering::Relaxed)
    }

    /// Accumulate per-request data for the current layer.
    /// tpm is NOT updated here — call finalize_layer_tpm() after the layer barrier.
    fn accumulate_layer_request(&self, token_count: usize, rtt_ms: f64) {
        self.layer_tokens
            .fetch_add(token_count as u64, Ordering::Relaxed);
        let rtt_us = (rtt_ms * 1000.0) as u64;
        // Atomic max update for layer_max_rtt_us
        let mut old = self.layer_max_rtt_us.load(Ordering::Relaxed);
        loop {
            if rtt_us <= old {
                break;
            }
            match self.layer_max_rtt_us.compare_exchange_weak(
                old,
                rtt_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => old = x,
            }
        }
    }

    /// Drain and return (total_tokens, max_rtt_ms) for this layer, resetting both accumulators.
    /// Returns None if no requests were accumulated.
    fn take_layer_stats(&self) -> Option<(u64, f64)> {
        let tokens = self.layer_tokens.swap(0, Ordering::Relaxed);
        let max_rtt_us = self.layer_max_rtt_us.swap(0, Ordering::Relaxed);
        if tokens > 0 && max_rtt_us > 0 {
            Some((tokens, max_rtt_us as f64 / 1000.0))
        } else {
            None
        }
    }

    /// Update throughput EMA with a pre-computed sample (tokens/ms).
    fn update_throughput_sample(&self, sample_tpm: f64) {
        let mut tpm = self.throughput_tpm.write().unwrap();
        *tpm = RTT_ALPHA * sample_tpm + (1.0 - RTT_ALPHA) * *tpm;
    }
}

/// Default polling interval for routing refresh fallback (seconds)
const DEFAULT_ROUTING_REFRESH_INTERVAL_SECS: u64 = 10;

/// Routing table client for fetching expert → worker mappings
/// with RTT-based adaptive load balancing for multi-replica scenarios
/// and background routing refresh for progressive startup support
pub struct RoutingClient {
    controller_addr: String,
    /// expert_id → list of WorkerEndpoints (multi-replica support)
    routing_table: Arc<RwLock<HashMap<String, Vec<WorkerEndpoint>>>>,
    /// Worker statistics keyed by grpc_addr for load balancing
    worker_stats: Arc<RwLock<HashMap<String, Arc<WorkerStats>>>>,
    routing_version: Arc<RwLock<u64>>,
    channel: Option<Channel>,
    /// Cancellation token for background tasks
    cancel_token: CancellationToken,
    /// Whether subscription is currently active
    subscription_active: Arc<AtomicBool>,
    /// Round-robin counter for breaking ties when workers have equal scores
    rr_counter: AtomicU64,
    /// Load balancing strategy (selected via EK_LB_ALGORITHM env var)
    lb_strategy: Box<dyn LoadBalancer>,
}

impl RoutingClient {
    pub fn new(controller_addr: String) -> Self {
        let (lb_strategy, _algo) = lb::balancer_from_env();
        Self {
            controller_addr,
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            worker_stats: Arc::new(RwLock::new(HashMap::new())),
            routing_version: Arc::new(RwLock::new(0)),
            channel: None,
            cancel_token: CancellationToken::new(),
            subscription_active: Arc::new(AtomicBool::new(false)),
            rr_counter: AtomicU64::new(0),
            lb_strategy,
        }
    }

    /// Create with a specific algorithm (for testing)
    #[cfg(test)]
    pub fn new_with_algorithm(controller_addr: String, algo: lb::LbAlgorithm) -> Self {
        Self {
            controller_addr,
            routing_table: Arc::new(RwLock::new(HashMap::new())),
            worker_stats: Arc::new(RwLock::new(HashMap::new())),
            routing_version: Arc::new(RwLock::new(0)),
            channel: None,
            cancel_token: CancellationToken::new(),
            subscription_active: Arc::new(AtomicBool::new(false)),
            rr_counter: AtomicU64::new(0),
            lb_strategy: lb::create_balancer(algo),
        }
    }

    /// Connect to controller
    pub async fn connect(&mut self) -> Result<()> {
        let uri = if self.controller_addr.starts_with("http://")
            || self.controller_addr.starts_with("https://")
        {
            self.controller_addr.clone()
        } else {
            format!("http://{}", self.controller_addr)
        };

        self.channel = Some(Channel::from_shared(uri)?.connect().await?);
        Ok(())
    }

    /// Fetch routing table from controller
    pub async fn fetch_routing(&self, expert_ids: Option<Vec<String>>) -> Result<()> {
        let channel = self
            .channel
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not connected to controller"))?;

        let mut client = RoutingServiceClient::new(channel.clone());

        let is_partial = expert_ids.is_some();
        let req = GetRoutingReq {
            expert_ids: expert_ids.unwrap_or_default(),
        };

        let response = client.get_routing(req).await?;
        let routing_response = response.into_inner();

        let mut table = self.routing_table.write().await;
        let mut version = self.routing_version.write().await;
        let mut stats = self.worker_stats.write().await;

        // Convert WorkerEndpointList to Vec<WorkerEndpoint>
        let new_routing: HashMap<String, Vec<WorkerEndpoint>> = routing_response
            .routing
            .into_iter()
            .map(|(id, list)| (id, list.endpoints))
            .collect();

        if is_partial {
            // Partial update
            table.extend(new_routing.clone());
        } else {
            // Full replace
            *table = new_routing.clone();
        }

        // Initialize stats for any new workers
        for endpoints in new_routing.values() {
            for endpoint in endpoints {
                if !stats.contains_key(&endpoint.grpc_addr) {
                    stats.insert(endpoint.grpc_addr.clone(), Arc::new(WorkerStats::default()));
                }
            }
        }

        *version = routing_response.version;

        // Count multi-replica experts for logging
        let multi_replica_count = table.values().filter(|v| v.len() > 1).count();
        if multi_replica_count > 0 {
            info!(
                "Routing table updated: {} experts ({} with multiple replicas), version={}",
                table.len(),
                multi_replica_count,
                *version
            );
        } else {
            info!(
                "Routing table updated: {} experts, version={}",
                table.len(),
                *version
            );
            if !table.is_empty() {
                warn!(
                    "All experts have single replicas — load balancing has no effect. \
                     Deploy experts on multiple workers to enable LB differentiation."
                );
            }
        }

        Ok(())
    }

    /// Select the best worker index from a list of endpoints using the
    /// configured load balancing strategy.
    fn pick_best(&self, expert_id: &str, endpoints: &[WorkerEndpoint], stats: &HashMap<String, Arc<WorkerStats>>) -> Option<usize> {
        self.lb_strategy.pick(expert_id, endpoints, stats, &self.rr_counter)
    }

    /// Select the best worker for an expert using the configured load balancing strategy.
    /// Returns the selected worker endpoint.
    pub async fn select_worker(&self, expert_id: &str) -> Option<WorkerEndpoint> {
        let table = self.routing_table.read().await;
        let endpoints = table.get(expert_id)?;

        if endpoints.is_empty() {
            return None;
        }

        let stats = self.worker_stats.read().await;
        let idx = self.pick_best(expert_id, endpoints, &stats)?;
        let worker = endpoints[idx].clone();

        debug!(
            "Selected worker {} for expert {} (among {} replicas)",
            worker.grpc_addr,
            expert_id,
            endpoints.len()
        );

        Some(worker)
    }

    /// **Reactive fallback selector**
    ///
    /// Used only in two narrow cases:
    /// 1. **Retry**: an expert task failed; we re-route to a (possibly different) worker after
    ///    routing has been refreshed.
    /// 2. **Missing-expert fallback**: an expert was absent from the LPT batch assignments
    ///    (e.g., routing table was updated mid-pass).
    ///
    /// For all normal forward passes, use `assign_layer_batch()`.
    pub async fn select_worker_and_mark_inflight(&self, expert_id: &str) -> Option<WorkerEndpoint> {
        let table = self.routing_table.read().await;
        let endpoints = table.get(expert_id)?;

        if endpoints.is_empty() {
            return None;
        }

        // Get write lock on stats to ensure atomic select + increment
        let mut stats = self.worker_stats.write().await;

        // Ensure all workers have stats entries
        for endpoint in endpoints {
            if !stats.contains_key(&endpoint.grpc_addr) {
                stats.insert(endpoint.grpc_addr.clone(), Arc::new(WorkerStats::default()));
            }
        }

        let idx = self.pick_best(expert_id, endpoints, &stats)?;
        let worker = endpoints[idx].clone();

        // Increment inflight counter for selected worker
        if let Some(s) = stats.get(&worker.grpc_addr) {
            s.increment_inflight();
            debug!(
                "Selected worker {} for expert {} (among {} replicas), inflight={}",
                worker.grpc_addr,
                expert_id,
                endpoints.len(),
                s.get_inflight()
            );
        }

        Some(worker)
    }

    /// Check if an expert exists in the routing table (for validation)
    #[allow(unused)]
    pub async fn has_expert(&self, expert_id: &str) -> bool {
        let table = self.routing_table.read().await;
        table.get(expert_id).map(|v| !v.is_empty()).unwrap_or(false)
    }

    /// Check which experts are missing from the routing table
    pub async fn find_missing_experts(&self, expert_ids: &[String]) -> Vec<String> {
        let table = self.routing_table.read().await;
        expert_ids
            .iter()
            .filter(|id| !table.get(*id).map(|v| !v.is_empty()).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// Get or create stats for a worker
    async fn get_or_create_stats(&self, grpc_addr: &str) -> Arc<WorkerStats> {
        // Try read lock first
        {
            let stats = self.worker_stats.read().await;
            if let Some(s) = stats.get(grpc_addr) {
                return s.clone();
            }
        }

        // Need to create new stats entry
        let mut stats = self.worker_stats.write().await;
        stats
            .entry(grpc_addr.to_string())
            .or_insert_with(|| Arc::new(WorkerStats::default()))
            .clone()
    }

    /// Track request start: increment inflight counter
    /// Note: Prefer using select_worker_and_mark_inflight for atomic selection + tracking
    #[allow(unused)]
    pub async fn on_request_start(&self, worker: &WorkerEndpoint) {
        let stats = self.get_or_create_stats(&worker.grpc_addr).await;
        stats.increment_inflight();
        debug!(
            "Request start for worker {}: inflight={}",
            worker.grpc_addr,
            stats.get_inflight()
        );
    }

    /// Track request completion: decrement inflight, update RTT, and accumulate layer stats.
    /// tpm is NOT updated here — call finalize_layer_tpm() after the layer barrier instead.
    pub async fn on_request_complete(
        &self,
        worker: &WorkerEndpoint,
        rtt_ms: f64,
        token_count: usize,
    ) {
        let stats = self.get_or_create_stats(&worker.grpc_addr).await;
        stats.decrement_inflight();
        stats.update_rtt(rtt_ms);
        stats.accumulate_layer_request(token_count, rtt_ms);
        debug!(
            "Request complete for worker {}: rtt={:.2}ms, avg_rtt={:.2}ms, inflight={}",
            worker.grpc_addr,
            rtt_ms,
            stats.get_avg_rtt(),
            stats.get_inflight()
        );
    }

    /// Update throughput_tpm for each worker using aggregate layer statistics.
    ///
    /// Must be called once after the MoE layer barrier (i.e., after all parallel expert
    /// requests for a layer have completed). Computes:
    ///
    ///   sample_tpm = total_tokens_to_worker / max_rtt_among_that_workers_requests
    ///
    /// This reflects the worker's actual aggregate throughput capacity regardless of
    /// how many parallel expert slots it has, avoiding the feedback loop where
    /// per-request tpm converges to equal values at the optimal assignment.
    pub async fn finalize_layer_tpm(&self) {
        let stats = self.worker_stats.read().await;
        let mut layer_workers: Vec<(String, u64, f64)> = Vec::new(); // (addr, tokens, rtt)
        for (addr, s) in stats.iter() {
            if let Some((tokens, max_rtt_ms)) = s.take_layer_stats() {
                let sample_tpm = tokens as f64 / max_rtt_ms;
                s.update_throughput_sample(sample_tpm);
                debug!(
                    "[Routing] Layer tpm update for {}: {:.3} tpm (tokens={}, max_rtt={:.1}ms)",
                    addr, sample_tpm, tokens, max_rtt_ms
                );
                layer_workers.push((addr.clone(), tokens, max_rtt_ms));
            }
        }
        // Log per-layer makespan distribution for diagnostics
        if layer_workers.len() > 1 {
            let max_rtt = layer_workers.iter().map(|(_, _, r)| *r).fold(0.0_f64, f64::max);
            let min_rtt = layer_workers.iter().map(|(_, _, r)| *r).fold(f64::INFINITY, f64::min);
            let spread = max_rtt - min_rtt;
            let summary: Vec<String> = layer_workers
                .iter()
                .map(|(a, t, r)| format!("{}:{}tok/{:.0}ms", a, t, r))
                .collect();
            debug!(
                "[Routing] Layer makespan: [{}] spread={:.0}ms",
                summary.join(", "),
                spread
            );
        }
    }

    /// Reset all worker statistics (RTT EMA, throughput EMA, inflight counters).
    ///
    /// Call this between benchmark iterations to prevent earlier runs from
    /// biasing later ones via stale EMA values.
    pub async fn reset_stats(&self) {
        let mut stats = self.worker_stats.write().await;
        for (addr, s) in stats.iter_mut() {
            *s = Arc::new(WorkerStats::default());
            debug!("[Routing] Reset stats for worker {}", addr);
        }
        info!(
            "[Routing] Reset stats for {} workers",
            stats.len()
        );
    }

    /// Assign a batch of expert calls to workers using the configured load
    /// balancing strategy (LPT for RTT, per-expert pick for others).
    pub async fn assign_layer_batch(
        &self,
        expert_calls: &[(String, usize)], // (expert_id, token_count)
    ) -> HashMap<String, WorkerEndpoint> {
        let table = self.routing_table.read().await;
        let mut stats = self.worker_stats.write().await;

        // Ensure all candidate workers have stats entries
        for (expert_id, _) in expert_calls {
            if let Some(endpoints) = table.get(expert_id.as_str()) {
                for ep in endpoints {
                    stats
                        .entry(ep.grpc_addr.clone())
                        .or_insert_with(|| Arc::new(WorkerStats::default()));
                }
            }
        }

        self.lb_strategy
            .assign_batch(expert_calls, &table, &stats, &self.rr_counter)
    }

    /// Get worker endpoint for an expert (legacy single-worker API)
    /// Uses select_worker internally for backwards compatibility
    #[allow(unused)]
    pub async fn get_worker(&self, expert_id: &str) -> Option<WorkerEndpoint> {
        self.select_worker(expert_id).await
    }

    /// Get workers for multiple experts (selects best worker for each)
    /// Note: This does NOT mark inflight - prefer select_worker_and_mark_inflight for load balancing
    #[allow(unused)]
    pub async fn get_workers(
        &self,
        expert_ids: &[String],
    ) -> HashMap<String, Option<WorkerEndpoint>> {
        let table = self.routing_table.read().await;
        let stats = self.worker_stats.read().await;

        expert_ids
            .iter()
            .map(|id| {
                let worker = table.get(id).and_then(|endpoints| {
                    if endpoints.is_empty() {
                        None
                    } else {
                        let idx = self.pick_best(id, endpoints, &stats)?;
                        Some(endpoints[idx].clone())
                    }
                });
                (id.clone(), worker)
            })
            .collect()
    }

    /// Get all routing entries (returns first worker for each expert for backwards compat)
    pub async fn get_all_routing(&self) -> HashMap<String, WorkerEndpoint> {
        let table = self.routing_table.read().await;
        table
            .iter()
            .filter_map(|(id, endpoints)| endpoints.first().cloned().map(|e| (id.clone(), e)))
            .collect()
    }

    /// Get all workers for all experts (full multi-replica routing table)
    pub async fn get_all_workers(&self) -> HashMap<String, Vec<WorkerEndpoint>> {
        let table = self.routing_table.read().await;
        table.clone()
    }

    /// Get routing version
    #[allow(unused)]
    pub async fn get_version(&self) -> u64 {
        let version = self.routing_version.read().await;
        *version
    }

    /// Remove all endpoints matching the given worker addresses from the local
    /// routing table.  Used by the retry loop to purge a dead worker whose
    /// address may have been re-introduced by a stale subscription update.
    #[allow(dead_code)]
    pub async fn purge_worker_addrs(&self, dead_addrs: &HashSet<String>) {
        if dead_addrs.is_empty() {
            return;
        }
        let mut table = self.routing_table.write().await;
        let mut removed_count = 0usize;
        let mut emptied = Vec::new();
        for (expert_id, endpoints) in table.iter_mut() {
            let before = endpoints.len();
            endpoints.retain(|ep| !dead_addrs.contains(&ep.grpc_addr));
            removed_count += before - endpoints.len();
            if endpoints.is_empty() {
                emptied.push(expert_id.clone());
            }
        }
        for eid in &emptied {
            table.remove(eid);
        }
        if removed_count > 0 {
            info!(
                "[RoutingClient] Purged {} endpoint entries for dead addrs {:?} ({} experts now empty)",
                removed_count,
                dead_addrs,
                emptied.len()
            );
        }
    }

    /// Start background routing refresh task
    /// This attempts gRPC subscription first, then falls back to polling
    pub fn start_background_refresh(&self) {
        let controller_addr = self.controller_addr.clone();
        let routing_table = self.routing_table.clone();
        let worker_stats = self.worker_stats.clone();
        let routing_version = self.routing_version.clone();
        let cancel_token = self.cancel_token.clone();
        let subscription_active = self.subscription_active.clone();
        let channel = self.channel.clone();

        tokio::spawn(async move {
            let refresh_task = Self::background_refresh_loop(
                controller_addr,
                routing_table,
                worker_stats,
                routing_version,
                subscription_active,
                channel,
                cancel_token.clone(),
            );

            tokio::select! {
                _ = refresh_task => {
                    info!("Background routing refresh task ended");
                }
                _ = cancel_token.cancelled() => {
                    info!("Background routing refresh cancelled");
                }
            }
        });
    }

    /// Background refresh loop that tries subscription first, then falls back to polling
    async fn background_refresh_loop(
        controller_addr: String,
        routing_table: Arc<RwLock<HashMap<String, Vec<WorkerEndpoint>>>>,
        worker_stats: Arc<RwLock<HashMap<String, Arc<WorkerStats>>>>,
        routing_version: Arc<RwLock<u64>>,
        subscription_active: Arc<AtomicBool>,
        channel: Option<Channel>,
        cancel_token: CancellationToken,
    ) {
        loop {
            // Try to subscribe to routing updates
            let subscription_result = Self::try_subscribe_routing(
                &controller_addr,
                &routing_table,
                &worker_stats,
                &routing_version,
                &subscription_active,
                channel.clone(),
                cancel_token.clone(),
            )
            .await;

            if let Err(e) = subscription_result {
                warn!(
                    "Routing subscription failed: {}, falling back to polling",
                    e
                );
                subscription_active.store(false, Ordering::Relaxed);
            }

            // If subscription ends or fails, fall back to polling
            if cancel_token.is_cancelled() {
                break;
            }

            // Polling fallback loop
            let poll_result = Self::polling_fallback_loop(
                &controller_addr,
                &routing_table,
                &worker_stats,
                &routing_version,
                &subscription_active,
                channel.clone(),
                cancel_token.clone(),
            )
            .await;

            if let Err(e) = poll_result {
                warn!("Routing polling failed: {}, will retry", e);
            }

            if cancel_token.is_cancelled() {
                break;
            }

            // Wait before retrying subscription
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
                _ = cancel_token.cancelled() => break,
            }
        }
    }

    /// Try to subscribe to routing updates via gRPC streaming
    async fn try_subscribe_routing(
        controller_addr: &str,
        routing_table: &Arc<RwLock<HashMap<String, Vec<WorkerEndpoint>>>>,
        worker_stats: &Arc<RwLock<HashMap<String, Arc<WorkerStats>>>>,
        routing_version: &Arc<RwLock<u64>>,
        subscription_active: &Arc<AtomicBool>,
        channel: Option<Channel>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let channel = match channel {
            Some(ch) => ch,
            None => {
                let uri = if controller_addr.starts_with("http://")
                    || controller_addr.starts_with("https://")
                {
                    controller_addr.to_string()
                } else {
                    format!("http://{}", controller_addr)
                };
                Channel::from_shared(uri)?.connect().await?
            }
        };

        let mut client = RoutingServiceClient::new(channel);
        let current_version = *routing_version.read().await;

        let req = SubscribeRoutingReq { current_version };
        let response = client.subscribe_routing_updates(req).await?;
        let mut stream = response.into_inner();

        subscription_active.store(true, Ordering::Relaxed);
        info!(
            "Routing subscription started at version {}",
            current_version
        );

        use tokio_stream::StreamExt;
        loop {
            tokio::select! {
                msg = stream.next() => {
                    match msg {
                        Some(Ok(update)) => {
                            let mut table = routing_table.write().await;
                            let mut version = routing_version.write().await;
                            let mut stats = worker_stats.write().await;

                            // Skip stale updates.  When fetch_routing() does a
                            // full replace it sets version to the broadcaster's
                            // current version.  Subscription updates with lower
                            // versions are from an earlier poller tick and may
                            // re-introduce a removed node's endpoints.
                            if update.version <= *version {
                                debug!(
                                    "Routing subscription: skipping stale update for {} \
                                     (update_v={}, current_v={})",
                                    update.expert_id, update.version, *version
                                );
                                continue;
                            }

                            match ChangeType::try_from(update.r#type).unwrap_or(ChangeType::Added) {
                                ChangeType::Added | ChangeType::Modified => {
                                    if let Some(endpoint_list) = update.endpoints {
                                        let endpoints = endpoint_list.endpoints;

                                        // Initialize stats for any new workers
                                        for endpoint in &endpoints {
                                            if !stats.contains_key(&endpoint.grpc_addr) {
                                                stats.insert(endpoint.grpc_addr.clone(), Arc::new(WorkerStats::default()));
                                            }
                                        }

                                        table.insert(update.expert_id.clone(), endpoints);
                                        debug!(
                                            "Routing subscription update: {} expert {} (version={})",
                                            if update.r#type == ChangeType::Added as i32 { "added" } else { "modified" },
                                            update.expert_id,
                                            update.version
                                        );
                                    }
                                }
                                ChangeType::Removed => {
                                    table.remove(&update.expert_id);
                                    debug!(
                                        "Routing subscription update: removed expert {} (version={})",
                                        update.expert_id, update.version
                                    );
                                }
                            }

                            *version = update.version;
                        }
                        Some(Err(e)) => {
                            warn!("Routing subscription error: {}", e);
                            return Err(anyhow::anyhow!("Subscription error: {}", e));
                        }
                        None => {
                            info!("Routing subscription stream ended");
                            return Ok(());
                        }
                    }
                }
                _ = cancel_token.cancelled() => {
                    info!("Routing subscription cancelled");
                    return Ok(());
                }
            }
        }
    }

    /// Polling fallback loop when subscription is not available or fails
    async fn polling_fallback_loop(
        controller_addr: &str,
        routing_table: &Arc<RwLock<HashMap<String, Vec<WorkerEndpoint>>>>,
        worker_stats: &Arc<RwLock<HashMap<String, Arc<WorkerStats>>>>,
        routing_version: &Arc<RwLock<u64>>,
        subscription_active: &Arc<AtomicBool>,
        channel: Option<Channel>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let channel = match channel {
            Some(ch) => ch,
            None => {
                let uri = if controller_addr.starts_with("http://")
                    || controller_addr.starts_with("https://")
                {
                    controller_addr.to_string()
                } else {
                    format!("http://{}", controller_addr)
                };
                Channel::from_shared(uri)?.connect().await?
            }
        };

        info!(
            "Starting routing polling fallback (interval={}s)",
            DEFAULT_ROUTING_REFRESH_INTERVAL_SECS
        );

        loop {
            // Check if subscription became active (another task reconnected)
            if subscription_active.load(Ordering::Relaxed) {
                debug!("Subscription became active, stopping polling");
                return Ok(());
            }

            // Wait for next poll interval
            tokio::select! {
                _ = tokio::time::sleep(std::time::Duration::from_secs(DEFAULT_ROUTING_REFRESH_INTERVAL_SECS)) => {}
                _ = cancel_token.cancelled() => {
                    info!("Routing polling cancelled");
                    return Ok(());
                }
            }

            // Fetch full routing table
            let mut client = RoutingServiceClient::new(channel.clone());
            let req = GetRoutingReq { expert_ids: vec![] };

            match client.get_routing(req).await {
                Ok(response) => {
                    let routing_response = response.into_inner();

                    let mut table = routing_table.write().await;
                    let mut version = routing_version.write().await;
                    let mut stats = worker_stats.write().await;

                    // Convert and update routing table
                    let new_routing: HashMap<String, Vec<WorkerEndpoint>> = routing_response
                        .routing
                        .into_iter()
                        .map(|(id, list)| (id, list.endpoints))
                        .collect();

                    // Full replace
                    *table = new_routing.clone();

                    // Initialize stats for any new workers
                    for endpoints in new_routing.values() {
                        for endpoint in endpoints {
                            if !stats.contains_key(&endpoint.grpc_addr) {
                                stats.insert(
                                    endpoint.grpc_addr.clone(),
                                    Arc::new(WorkerStats::default()),
                                );
                            }
                        }
                    }

                    let old_version = *version;
                    *version = routing_response.version;

                    if routing_response.version != old_version {
                        info!(
                            "Routing table refreshed via polling: {} experts (version {} -> {})",
                            table.len(),
                            old_version,
                            routing_response.version
                        );
                    } else {
                        debug!(
                            "Routing table unchanged via polling: {} experts (version={})",
                            table.len(),
                            *version
                        );
                    }
                }
                Err(e) => {
                    warn!("Routing poll failed: {}", e);
                    // Continue polling, don't exit the loop
                }
            }
        }
    }

    /// Stop background refresh task
    #[allow(unused)]
    pub fn stop_background_refresh(&self) {
        self.cancel_token.cancel();
    }

    /// Check if subscription is currently active
    #[allow(unused)]
    pub fn is_subscription_active(&self) -> bool {
        self.subscription_active.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod tests;
