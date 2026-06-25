use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use log::debug;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::{LoadBalancer, device_default_tpm, warm_start_rtt, warm_start_tpm};

// RTT-specific scoring constants
const PENALTY_FACTOR: f64 = 10.0; // ms penalty per inflight^2
const MIN_RTT: f64 = 1.0; // Floor to avoid division issues (ms)

/// RTT-based adaptive load balancer with LPT batch assignment.
///
/// Single-expert selection: Score = 1 / (avg_rtt + inflight^2 * PENALTY_FACTOR)
/// Batch assignment: LPT heuristic using throughput-based projected completion time.
pub(crate) struct RttBalancer;

impl LoadBalancer for RttBalancer {
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
                // Device-aware fallback RTT for unmeasured workers
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

        // Collect indices of all endpoints tied at max score
        let tied: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|&(_, &s)| (s - max_score).abs() < f64::EPSILON)
            .map(|(i, _)| i)
            .collect();

        // Round-robin among tied endpoints
        let rr = rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
        Some(tied[rr % tied.len()])
    }

    /// LPT (Longest Processing Time) batch assignment.
    ///
    /// Sorts calls by token_count descending, assigns each to worker with minimum
    /// projected completion time (LPT gives <=4/3 optimal makespan in O(N log N)).
    fn assign_batch(
        &self,
        expert_calls: &[(String, usize)],
        table: &HashMap<String, Vec<WorkerEndpoint>>,
        stats: &HashMap<String, Arc<WorkerStats>>,
        rr_counter: &AtomicU64,
    ) -> HashMap<String, WorkerEndpoint> {
        // Sort by token_count descending (LPT: largest jobs first)
        let mut sorted_calls: Vec<(String, usize)> = expert_calls
            .iter()
            .filter(|(eid, _)| {
                table
                    .get(eid.as_str())
                    .map(|v| !v.is_empty())
                    .unwrap_or(false)
            })
            .map(|(eid, tc)| (eid.clone(), *tc))
            .collect();
        sorted_calls.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        // Projected completion time (ms) per worker, accumulated as we assign
        let mut worker_load: HashMap<String, f64> = HashMap::new();
        let mut assignments: HashMap<String, WorkerEndpoint> = HashMap::new();

        for (expert_id, token_count) in &sorted_calls {
            let endpoints = match table.get(expert_id.as_str()) {
                Some(e) if !e.is_empty() => e,
                _ => continue,
            };

            // Pick worker minimizing projected completion time after this assignment.
            // Use device-aware warm-start for unmeasured workers.
            let proj_scores: Vec<f64> = endpoints
                .iter()
                .map(|ep| {
                    let current_load = worker_load.get(&ep.grpc_addr).copied().unwrap_or(0.0);
                    let ws_tpm = warm_start_tpm(stats, Some(&ep.device));
                    let dev_tpm = device_default_tpm(&ep.device);
                    let tpm = stats
                        .get(&ep.grpc_addr)
                        .map(|s| {
                            if s.has_measured_tpm() {
                                s.get_throughput_tpm()
                            } else {
                                ws_tpm
                            }
                        })
                        .unwrap_or(dev_tpm)
                        .max(f64::EPSILON);
                    current_load + *token_count as f64 / tpm
                })
                .collect();

            let min_proj = proj_scores.iter().cloned().fold(f64::INFINITY, f64::min);
            let tied: Vec<usize> = proj_scores
                .iter()
                .enumerate()
                .filter(|&(_, &s)| (s - min_proj).abs() < f64::EPSILON)
                .map(|(i, _)| i)
                .collect();
            let rr = rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
            let best = if tied.is_empty() {
                None
            } else {
                Some(&endpoints[tied[rr % tied.len()]])
            };

            if let Some(worker) = best {
                // Use the same device-aware tpm for load accumulation as for scoring.
                let ws_tpm = warm_start_tpm(stats, Some(&worker.device));
                let dev_tpm = device_default_tpm(&worker.device);
                let tpm = stats
                    .get(&worker.grpc_addr)
                    .map(|s| {
                        if s.has_measured_tpm() {
                            s.get_throughput_tpm()
                        } else {
                            ws_tpm
                        }
                    })
                    .unwrap_or(dev_tpm)
                    .max(f64::EPSILON);
                let added = *token_count as f64 / tpm;
                let new_load = worker_load.entry(worker.grpc_addr.clone()).or_insert(0.0);
                *new_load += added;
                if let Some(s) = stats.get(&worker.grpc_addr) {
                    s.increment_inflight();
                }
                debug!(
                    "[Routing] LPT: expert {} ({} tokens) -> worker {} (projected_load={:.1}ms, tpm={:.3})",
                    expert_id, token_count, worker.grpc_addr, *new_load, tpm
                );
                assignments.insert(expert_id.clone(), worker.clone());
            }
        }

        if !assignments.is_empty() {
            let load_summary: Vec<String> = worker_load
                .iter()
                .map(|(addr, load)| format!("{}={:.1}ms", addr, load))
                .collect();
            debug!(
                "[Routing] LPT assigned {} experts, worker loads: [{}]",
                assignments.len(),
                load_summary.join(", ")
            );
        }

        assignments
    }
}
