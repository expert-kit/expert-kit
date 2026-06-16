use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::LoadBalancer;

/// Pure round-robin load balancer. Ignores all statistics.
pub(crate) struct RoundRobinBalancer;

impl LoadBalancer for RoundRobinBalancer {
    fn pick(
        &self,
        _expert_id: &str,
        endpoints: &[WorkerEndpoint],
        _stats: &HashMap<String, Arc<WorkerStats>>,
        rr_counter: &AtomicU64,
    ) -> Option<usize> {
        if endpoints.is_empty() {
            return None;
        }
        let idx = rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
        Some(idx % endpoints.len())
    }
}
