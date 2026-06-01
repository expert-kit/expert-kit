use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::LoadBalancer;

/// Least-inflight load balancer. Picks the worker with the fewest in-flight
/// requests, with round-robin tie-breaking.
pub(crate) struct LeastInflightBalancer;

impl LoadBalancer for LeastInflightBalancer {
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

        let inflights: Vec<u32> = endpoints
            .iter()
            .map(|ep| {
                stats
                    .get(&ep.grpc_addr)
                    .map(|s| s.get_inflight())
                    .unwrap_or(0)
            })
            .collect();

        let min_inflight = *inflights.iter().min().unwrap();
        let tied: Vec<usize> = inflights
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == min_inflight)
            .map(|(i, _)| i)
            .collect();

        let rr = rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
        Some(tied[rr % tied.len()])
    }
}
