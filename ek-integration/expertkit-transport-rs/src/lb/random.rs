use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use rand::Rng;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::LoadBalancer;

/// Random load balancer. Selects a worker uniformly at random.
pub(crate) struct RandomBalancer;

impl LoadBalancer for RandomBalancer {
    fn pick(
        &self,
        _expert_id: &str,
        endpoints: &[WorkerEndpoint],
        _stats: &HashMap<String, Arc<WorkerStats>>,
        _rr_counter: &AtomicU64,
    ) -> Option<usize> {
        if endpoints.is_empty() {
            return None;
        }
        let idx = rand::rng().random_range(0..endpoints.len());
        Some(idx)
    }
}
