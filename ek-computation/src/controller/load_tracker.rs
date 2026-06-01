use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// LoadTracker tracks worker load metrics for scheduling decisions
/// Load is represented as number of active requests per worker
#[derive(Clone)]
pub struct LoadTracker {
    inner: Arc<RwLock<LoadTrackerInner>>,
}

struct LoadTrackerInner {
    /// hostname → active request count
    load_map: HashMap<String, u32>,
}

impl LoadTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(LoadTrackerInner {
                load_map: HashMap::new(),
            })),
        }
    }

    /// Get current load for a worker (returns 0 if unknown)
    pub fn get_load(&self, hostname: &str) -> u32 {
        // Clone the Arc and use tokio's blocking mechanism
        let inner_clone = self.inner.clone();
        // Use try_read for non-blocking access
        match inner_clone.try_read() {
            Ok(inner) => *inner.load_map.get(hostname).unwrap_or(&0),
            Err(_) => {
                // If we can't get the lock, return 0 (assume no load)
                log::warn!("Failed to acquire read lock for load tracker");
                0
            }
        }
    }

    /// Update load for a worker
    pub async fn update_load(&self, hostname: String, load: u32) {
        let mut inner = self.inner.write().await;
        inner.load_map.insert(hostname, load);
    }

    /// Increment load for a worker (e.g., when request starts)
    pub async fn increment_load(&self, hostname: &str) {
        let mut inner = self.inner.write().await;
        let load = inner.load_map.entry(hostname.to_string()).or_insert(0);
        *load += 1;
    }

    /// Decrement load for a worker (e.g., when request completes)
    pub async fn decrement_load(&self, hostname: &str) {
        let mut inner = self.inner.write().await;
        if let Some(load) = inner.load_map.get_mut(hostname) {
            *load = load.saturating_sub(1);
        }
    }

    /// Batch update loads for multiple workers
    pub async fn batch_update(&self, loads: HashMap<String, u32>) {
        let mut inner = self.inner.write().await;
        for (hostname, load) in loads {
            inner.load_map.insert(hostname, load);
        }
    }

    /// Remove a worker from tracking (e.g., when worker goes offline)
    pub async fn remove_worker(&self, hostname: &str) {
        let mut inner = self.inner.write().await;
        inner.load_map.remove(hostname);
    }

    /// Get all worker loads (for debugging/metrics)
    pub async fn get_all_loads(&self) -> HashMap<String, u32> {
        let inner = self.inner.read().await;
        inner.load_map.clone()
    }
}

impl Default for LoadTracker {
    fn default() -> Self {
        Self::new()
    }
}
