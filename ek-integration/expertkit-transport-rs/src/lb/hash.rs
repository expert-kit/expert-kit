use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use crate::routing::WorkerStats;
use crate::transport::grpc::proto::ek::control::v1::WorkerEndpoint;

use super::LoadBalancer;

/// Hash-based (sticky) load balancer. Routes the same expert_id to the same
/// worker deterministically, useful for testing cache locality effects.
pub(crate) struct HashBalancer;

impl LoadBalancer for HashBalancer {
    fn pick(
        &self,
        expert_id: &str,
        endpoints: &[WorkerEndpoint],
        _stats: &HashMap<String, Arc<WorkerStats>>,
        _rr_counter: &AtomicU64,
    ) -> Option<usize> {
        if endpoints.is_empty() {
            return None;
        }
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        expert_id.hash(&mut hasher);
        let hash = hasher.finish();
        Some((hash as usize) % endpoints.len())
    }
}
