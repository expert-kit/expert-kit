use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::{LoadBalancer, warm_start_rtt};

// Same constants as rtt.rs
const PENALTY_FACTOR: f64 = 10.0;
const MIN_RTT: f64 = 1.0;

/// Per-call greedy RTT load balancer.
///
/// Same scoring as RttBalancer::pick() — Score = 1 / (avg_rtt + inflight² × PENALTY)
/// — but uses the default assign_batch() (per-call, no LPT joint assignment).
/// This isolates the effect of RTT-aware selection without layer-level coordination.
pub(crate) struct GreedyRttBalancer;

impl LoadBalancer for GreedyRttBalancer {
    fn pick(
        &self,
        _expert_id: &str,
        endpoints: &[WorkerEndpoint],
        stats: &HashMap<String, Arc<WorkerStats>>,
        rr_counter: &AtomicU64,
    ) -> Option<usize> {
        if endpoints.is_empty() {
            return None;
        }
        if endpoints.len() == 1 {
            return Some(0);
        }

        let scores: Vec<f64> = endpoints
            .iter()
            .map(|ep| {
                let ws_rtt = warm_start_rtt(stats, Some(&ep.device));
                stats
                    .get(&ep.grpc_addr)
                    .map(|s| {
                        let avg_rtt = if s.has_measured_rtt() {
                            s.get_avg_rtt()
                        } else {
                            ws_rtt
                        }
                        .max(MIN_RTT);
                        let inflight = s.get_inflight() as f64;
                        let penalty = inflight.powi(2) * PENALTY_FACTOR;
                        1.0 / (avg_rtt + penalty)
                    })
                    .unwrap_or(1.0 / ws_rtt)
            })
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let tied: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| (s - max_score).abs() < f64::EPSILON)
            .map(|(i, _)| i)
            .collect();

        let rr = rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
        Some(tied[rr % tied.len()])
    }

    // Uses default assign_batch() — calls pick() per expert, no joint assignment.
}
