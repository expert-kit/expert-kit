pub mod peer_client;
pub mod peer_server;

use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use bytes::Bytes;
use dashmap::DashMap;
use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use opendal::Operator;

use crate::{dal::op_from_settings, safetensor::ExpertKey, weight_srv::client::WeightSrvClient};
use peer_client::WeightManagerClient;

/// Per-tier hit counter and total latency accumulator.
pub struct TierStats {
    pub hits: AtomicU64,
    pub total_ns: AtomicU64,
}

impl TierStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
        }
    }

    fn record(&self, d: Duration) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.total_ns
            .fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
    }

    /// Returns `(hits, total_ns)` snapshot.
    pub fn snapshot(&self) -> (u64, u64) {
        (
            self.hits.load(Ordering::Relaxed),
            self.total_ns.load(Ordering::Relaxed),
        )
    }
}

/// Aggregated per-tier stats for the weight manager.
pub struct WmStats {
    pub mem: TierStats,
    pub disk: TierStats,
    pub peer: TierStats,
    pub central: TierStats,
}

impl WmStats {
    fn new() -> Self {
        Self {
            mem: TierStats::new(),
            disk: TierStats::new(),
            peer: TierStats::new(),
            central: TierStats::new(),
        }
    }
}

/// Thread-safe in-memory cache backed by DashMap.
/// Insertion-order eviction (approximate LRU): entries are evicted one at a time
/// in insertion order until the cache is within the byte limit.
struct MemCache {
    map: DashMap<String, Arc<Bytes>>,
    total_bytes: AtomicUsize,
    max_bytes: usize,
    eviction_queue: std::sync::Mutex<std::collections::VecDeque<String>>,
}

impl MemCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            map: DashMap::new(),
            total_bytes: AtomicUsize::new(0),
            max_bytes,
            eviction_queue: std::sync::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    fn get(&self, key: &str) -> Option<Arc<Bytes>> {
        self.map.get(key).map(|v| v.clone())
    }

    fn insert(&self, key: String, value: Arc<Bytes>) {
        let incoming = value.len();

        if let Some(old) = self.map.get(&key) {
            let old_len = old.len();
            drop(old);
            self.map.insert(key, value);
            let _ = self
                .total_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                    Some(cur.saturating_sub(old_len).saturating_add(incoming))
                });
            return;
        }

        self.map.insert(key.clone(), value);
        self.total_bytes.fetch_add(incoming, Ordering::Relaxed);
        self.eviction_queue.lock().unwrap().push_back(key);

        while self.total_bytes.load(Ordering::Relaxed) > self.max_bytes {
            let victim = self.eviction_queue.lock().unwrap().pop_front();
            match victim {
                None => break,
                Some(k) => {
                    if let Some((_, v)) = self.map.remove(&k) {
                        let _ = self.total_bytes.fetch_update(
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                            |cur| Some(cur.saturating_sub(v.len())),
                        );
                        log::debug!("mem_cache: evicted {k} ({}B)", v.len());
                    }
                }
            }
        }
    }

    fn remove(&self, key: &str) {
        if let Some((_, v)) = self.map.remove(key) {
            let _ = self
                .total_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
                    Some(cur.saturating_sub(v.len()))
                });
        }
    }

    fn clear(&self) {
        self.map.clear();
        self.total_bytes.store(0, Ordering::Relaxed);
        self.eviction_queue.lock().unwrap().clear();
    }
}

/// LocalWeightManager manages expert weight bytes with a four-tier cache:
///   Tier 1: in-memory DashMap cache (Arc<Bytes>, zero-copy)
///   Tier 2: disk cache via OpenDAL
///   Tier 3: peer fetch from sibling LocalWeightManagers via HTTP
///   Tier 4: central weight server (HTTP fallback)
pub struct LocalWeightManager {
    mem_cache: MemCache,
    dal: Operator,
    /// Peer WM clients keyed by wm_addr; populated lazily on first fetch
    peer_clients: DashMap<String, Arc<WeightManagerClient>>,
    /// Peer index: expert_obj_key -> list of wm_addr strings that have the expert
    peer_index: DashMap<String, Vec<String>>,
    central_client: Option<WeightSrvClient>,
    pub stats: Arc<WmStats>,
}

impl LocalWeightManager {
    /// Create a `LocalWeightManager` from pre-built components (useful for benchmarks/tests).
    pub fn new_with_parts(
        dal: Operator,
        central_client: Option<WeightSrvClient>,
        mem_cache_mb: usize,
        _log_stats: bool,
    ) -> Arc<Self> {
        let max_bytes = mem_cache_mb * 1024 * 1024;
        Arc::new(Self {
            mem_cache: MemCache::new(max_bytes),
            dal,
            peer_clients: DashMap::new(),
            peer_index: DashMap::new(),
            central_client,
            stats: Arc::new(WmStats::new()),
        })
    }

    pub fn new_shared() -> Arc<Self> {
        let settings = get_ek_settings();
        let ws = &settings.weight;

        let central_client = if let Some(srv) = &ws.server {
            log::info!("weight server configured: {}", srv.addr);
            Some(WeightSrvClient::new(srv.addr.clone()))
        } else {
            log::warn!(
                "weight server not configured; LocalWeightManager will rely on disk cache and peers"
            );
            None
        };

        let dal = op_from_settings(&ws.cache);
        let max_bytes = ws.mem_cache_mb * 1024 * 1024;

        Arc::new(Self {
            mem_cache: MemCache::new(max_bytes),
            dal,
            peer_clients: DashMap::new(),
            peer_index: DashMap::new(),
            central_client,
            stats: Arc::new(WmStats::new()),
        })
    }

    /// Register or update the peers that hold a given expert.
    pub fn update_peer_index(&self, expert_id: String, wm_addrs: Vec<String>) {
        self.peer_index.insert(expert_id, wm_addrs);
    }

    /// Remove an expert from the peer index.
    pub fn remove_from_peer_index(&self, expert_id: &str) {
        self.peer_index.remove(expert_id);
    }

    /// Evict all entries from the in-memory cache (disk cache is unaffected).
    pub fn evict_all(&self) {
        self.mem_cache.clear();
    }

    /// Return a clone of the shared stats handle.
    pub fn stats(&self) -> Arc<WmStats> {
        self.stats.clone()
    }

    /// Fetch expert weight bytes, checking all four tiers in order.
    /// Returns an `Arc<Bytes>` — callers get a cheap refcount increment, no data copy.
    pub async fn get_expert(&self, key: &ExpertKey) -> EKResult<Arc<Bytes>> {
        let obj_key = key.as_object_key();

        // Tier 1: memory cache (lock-free DashMap read)
        let t = Instant::now();
        if let Some(bytes) = self.mem_cache.get(&obj_key) {
            self.stats.mem.record(t.elapsed());
            return Ok(bytes);
        }

        // Tier 2: disk cache
        let t = Instant::now();
        if self.dal.exists(&obj_key).await? {
            let buf = self.dal.read(&obj_key).await?;
            self.stats.disk.record(t.elapsed());
            let bytes = Arc::new(buf.to_bytes());
            self.mem_cache.insert(obj_key.clone(), bytes.clone());
            return Ok(bytes);
        }

        // Tier 3: peer fetch
        let peer_addrs: Vec<String> = self
            .peer_index
            .get(&obj_key)
            .map(|r| r.clone())
            .unwrap_or_default();

        for wm_addr in &peer_addrs {
            let t = Instant::now();
            match self.fetch_from_peer(wm_addr, key).await {
                Ok(bytes) => {
                    self.stats.peer.record(t.elapsed());
                    self.mem_cache.insert(obj_key.clone(), bytes.clone());
                    self.write_to_disk_async(obj_key, bytes.clone());
                    return Ok(bytes);
                }
                Err(e) => {
                    log::warn!("peer fetch from {wm_addr} failed: {e}");
                }
            }
        }

        // Tier 4: central HTTP fallback
        let t = Instant::now();
        let raw = if let Some(ref client) = self.central_client {
            client
                .load_expert(key.model(), key.layer(), key.idx())
                .await?
        } else {
            return Err(EKError::NotFound(format!(
                "expert {obj_key} not found: no central server configured and all peer fetches failed"
            )));
        };
        self.stats.central.record(t.elapsed());
        log::debug!(
            obj_key = obj_key.as_str();
            "loaded from central weight server"
        );

        let bytes = Arc::new(Bytes::from(raw));
        self.mem_cache.insert(obj_key.clone(), bytes.clone());
        self.write_to_disk_async(obj_key, bytes.clone());
        Ok(bytes)
    }

    /// Fetch from local caches only (mem + disk). Used by the peer HTTP server
    /// to avoid cascading peer fetches and circular chains.
    pub async fn get_expert_local(&self, key: &ExpertKey) -> EKResult<Arc<Bytes>> {
        let obj_key = key.as_object_key();

        if let Some(bytes) = self.mem_cache.get(&obj_key) {
            return Ok(bytes);
        }

        if self.dal.exists(&obj_key).await? {
            let buf = self.dal.read(&obj_key).await?;
            let bytes = Arc::new(buf.to_bytes());
            self.mem_cache.insert(obj_key.clone(), bytes.clone());
            return Ok(bytes);
        }

        Err(EKError::NotFound(format!(
            "expert {obj_key} not found in local caches"
        )))
    }

    /// Evict an expert from the in-memory cache (disk cache is unaffected).
    pub fn evict_mem(&self, key: &ExpertKey) {
        self.mem_cache.remove(&key.as_object_key());
    }

    // --- private helpers ---

    async fn fetch_from_peer(&self, wm_addr: &str, key: &ExpertKey) -> EKResult<Arc<Bytes>> {
        let client = self
            .peer_clients
            .entry(wm_addr.to_string())
            .or_insert_with(|| Arc::new(WeightManagerClient::new(wm_addr.to_string())))
            .clone();
        client.get_expert(key).await
    }

    fn write_to_disk_async(&self, obj_key: String, bytes: Arc<Bytes>) {
        let dal = self.dal.clone();
        tokio::spawn(async move {
            let data = (*bytes).clone();
            if let Err(e) = dal.write(&obj_key, data).await {
                log::error!("failed to write {obj_key} to disk cache: {e}");
            }
        });
    }
}
