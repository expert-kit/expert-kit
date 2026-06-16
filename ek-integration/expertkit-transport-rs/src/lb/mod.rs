use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use log::info;

use crate::routing::{WorkerStats, DEFAULT_RTT, DEFAULT_THROUGHPUT_TPM};
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

mod greedy_rtt;
mod greedy_throughput;
mod hash;
mod least_inflight;
mod random;
mod round_robin;
mod rtt;

pub(crate) use greedy_rtt::GreedyRttBalancer;
pub(crate) use greedy_throughput::GreedyThroughputBalancer;
pub(crate) use hash::HashBalancer;
pub(crate) use least_inflight::LeastInflightBalancer;
pub(crate) use random::RandomBalancer;
pub(crate) use round_robin::RoundRobinBalancer;
pub(crate) use rtt::RttBalancer;

/// Load balancing algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum LbAlgorithm {
    /// RTT-based adaptive scoring + LPT batch assignment (default)
    Rtt,
    /// Pure round-robin, ignores all stats
    RoundRobin,
    /// Random selection
    Random,
    /// Pick worker with fewest in-flight requests
    LeastInflight,
    /// Hash expert_id to a fixed worker (sticky routing)
    Hash,
    /// Per-call greedy RTT (same scoring as Rtt::pick, no LPT batch)
    GreedyRtt,
    /// Per-call greedy throughput (uses θ̂_w per call, no joint assignment)
    GreedyThroughput,
}

impl LbAlgorithm {
    /// Parse from environment variable value (case-insensitive)
    pub(crate) fn from_env_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rtt" | "adaptive" => Some(Self::Rtt),
            "round_robin" | "roundrobin" | "rr" => Some(Self::RoundRobin),
            "random" | "rand" => Some(Self::Random),
            "least_inflight" | "leastinflight" | "li" => Some(Self::LeastInflight),
            "hash" | "sticky" => Some(Self::Hash),
            "greedy_rtt" | "greedyrtt" => Some(Self::GreedyRtt),
            "greedy_throughput" | "greedythroughput" => Some(Self::GreedyThroughput),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Rtt => "rtt",
            Self::RoundRobin => "round_robin",
            Self::Random => "random",
            Self::LeastInflight => "least_inflight",
            Self::Hash => "hash",
            Self::GreedyRtt => "greedy_rtt",
            Self::GreedyThroughput => "greedy_throughput",
        }
    }
}

/// Trait for load balancing strategies
pub(crate) trait LoadBalancer: Send + Sync {
    /// Pick the index of the best worker from candidates.
    ///
    /// `expert_id` is provided for strategies that need it (e.g., hash).
    fn pick(
        &self,
        expert_id: &str,
        endpoints: &[WorkerEndpoint],
        stats: &HashMap<String, Arc<WorkerStats>>,
        rr_counter: &AtomicU64,
    ) -> Option<usize>;

    /// Batch assignment for a layer of expert calls.
    ///
    /// Default implementation calls `pick()` per expert and increments inflight.
    /// Tracks accumulated worker load within the batch so that subsequent picks
    /// see updated inflight counts from earlier assignments in the same layer.
    /// Override for strategies that need richer cross-expert awareness (e.g., LPT).
    fn assign_batch(
        &self,
        expert_calls: &[(String, usize)],
        table: &HashMap<String, Vec<WorkerEndpoint>>,
        stats: &HashMap<String, Arc<WorkerStats>>,
        rr_counter: &AtomicU64,
    ) -> HashMap<String, WorkerEndpoint> {
        let mut assignments = HashMap::new();
        for (expert_id, _) in expert_calls {
            if let Some(endpoints) = table.get(expert_id.as_str()) {
                if endpoints.is_empty() {
                    continue;
                }
                if let Some(idx) = self.pick(expert_id, endpoints, stats, rr_counter) {
                    let worker = endpoints[idx].clone();
                    // Increment inflight so subsequent picks in this batch see the load.
                    // This gives non-LPT algorithms intra-batch load awareness.
                    if let Some(s) = stats.get(&worker.grpc_addr) {
                        s.increment_inflight();
                    }
                    assignments.insert(expert_id.clone(), worker);
                }
            }
        }
        assignments
    }
}

/// Create a load balancer from the algorithm enum
pub(crate) fn create_balancer(algo: LbAlgorithm) -> Box<dyn LoadBalancer> {
    match algo {
        LbAlgorithm::Rtt => Box::new(RttBalancer),
        LbAlgorithm::RoundRobin => Box::new(RoundRobinBalancer),
        LbAlgorithm::Random => Box::new(RandomBalancer),
        LbAlgorithm::LeastInflight => Box::new(LeastInflightBalancer),
        LbAlgorithm::Hash => Box::new(HashBalancer),
        LbAlgorithm::GreedyRtt => Box::new(GreedyRttBalancer),
        LbAlgorithm::GreedyThroughput => Box::new(GreedyThroughputBalancer),
    }
}

// ── Device-aware warm-start helpers ──────────────────────────────────────────
//
// These defaults are used ONLY before any real measurements arrive (cold-start).
// The absolute values matter less than the **ratio** between device types,
// because the EMA self-corrects quickly once real data flows.
//
// The ratio determines how aggressively LPT favors GPU over CPU on the very
// first layer.  A 20:1 ratio means "assume GPU is ~20× faster" which is
// conservative for typical MoE FFN workloads (real ratio is often 10-50×
// depending on batch size and model dimensions).

/// Default throughput (tokens/ms) for a worker based on its device type.
/// Only the ratio matters — EMA converges to real values within a few layers.
pub(crate) fn device_default_tpm(device: &str) -> f64 {
    if device.starts_with("cuda") {
        // GPU default: 20× CPU.  Actual value self-corrects via EMA.
        DEFAULT_THROUGHPUT_TPM * 20.0
    } else {
        DEFAULT_THROUGHPUT_TPM
    }
}

/// Default RTT (ms) for a worker based on its device type.
/// Only the ratio matters — EMA converges to real values within a few layers.
pub(crate) fn device_default_rtt(device: &str) -> f64 {
    if device.starts_with("cuda") {
        DEFAULT_RTT // GPU: use global default as baseline
    } else {
        DEFAULT_RTT * 5.0 // CPU: assume 5× slower (conservative)
    }
}

/// Compute warm-start TPM by averaging measured workers.
/// Falls back to device-aware default for the given endpoint.
pub(crate) fn warm_start_tpm(
    stats: &HashMap<String, Arc<WorkerStats>>,
    device_hint: Option<&str>,
) -> f64 {
    let measured: Vec<f64> = stats
        .values()
        .filter(|s| s.has_measured_tpm())
        .map(|s| s.get_throughput_tpm())
        .collect();
    if measured.is_empty() {
        device_hint
            .map(|d| device_default_tpm(d))
            .unwrap_or(DEFAULT_THROUGHPUT_TPM)
    } else {
        measured.iter().sum::<f64>() / measured.len() as f64
    }
}

/// Compute warm-start RTT by averaging measured workers.
/// Falls back to device-aware default for the given endpoint.
pub(crate) fn warm_start_rtt(
    stats: &HashMap<String, Arc<WorkerStats>>,
    device_hint: Option<&str>,
) -> f64 {
    let measured: Vec<f64> = stats
        .values()
        .filter(|s| s.has_measured_rtt())
        .map(|s| s.get_avg_rtt())
        .collect();
    if measured.is_empty() {
        device_hint
            .map(|d| device_default_rtt(d))
            .unwrap_or(DEFAULT_RTT)
    } else {
        measured.iter().sum::<f64>() / measured.len() as f64
    }
}

/// Read `EK_LB_ALGORITHM` env var and create the appropriate balancer.
/// Returns `(balancer, algorithm)`.
pub(crate) fn balancer_from_env() -> (Box<dyn LoadBalancer>, LbAlgorithm) {
    let algo = std::env::var("EK_LB_ALGORITHM")
        .ok()
        .and_then(|s| LbAlgorithm::from_env_str(&s))
        .unwrap_or(LbAlgorithm::Rtt);

    info!("[Routing] Load balancing algorithm: {}", algo.as_str());
    (create_balancer(algo), algo)
}
