use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::{LoadBalancer, warm_start_tpm};

/// Per-call greedy throughput load balancer.
///
/// Picks the worker minimizing projected cost for this single call:
///   cost = token_count / θ̂_w
/// where θ̂_w is the per-worker throughput estimate (tokens per millisecond).
///
/// Unlike RttBalancer's LPT assign_batch(), this makes each decision in isolation
/// without tracking accumulated load across the layer's invocations.
/// This isolates the value of the throughput signal without joint assignment.
pub(crate) struct GreedyThroughputBalancer;

impl LoadBalancer for GreedyThroughputBalancer {
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

        // Pick worker with highest throughput (lowest cost per token).
        // Also factor in current inflight to avoid piling onto a busy worker.
        let scores: Vec<f64> = endpoints
            .iter()
            .map(|ep| {
                let ws_tpm = warm_start_tpm(stats, Some(&ep.device));
                let tpm = stats
                    .get(&ep.grpc_addr)
                    .map(|s| {
                        if s.has_measured_tpm() {
                            s.get_throughput_tpm()
                        } else {
                            ws_tpm
                        }
                    })
                    .unwrap_or(ws_tpm)
                    .max(f64::EPSILON);

                let inflight = stats
                    .get(&ep.grpc_addr)
                    .map(|s| s.get_inflight() as f64)
                    .unwrap_or(0.0);

                // Projected completion: inflight / throughput gives current queue drain time
                // Higher throughput and lower inflight → lower projected time → better
                (inflight + 1.0) / tpm
            })
            .collect();

        // Pick minimum projected time
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let tied: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| (s - min_score).abs() < f64::EPSILON)
            .map(|(i, _)| i)
            .collect();

        let rr = rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
        Some(tied[rr % tied.len()])
    }

    // Uses default assign_batch() — calls pick() per expert, no joint assignment.
}
