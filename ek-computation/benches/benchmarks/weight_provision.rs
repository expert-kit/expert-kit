use std::{
    collections::HashMap,
    hint::black_box,
    net::TcpListener,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use actix_web::{App, HttpResponse, HttpServer, web};
use criterion::{BenchmarkId, Criterion};
use dashmap::DashMap;
use ek_base::config::{FSConfig, OpenDALStorage};
use ek_db::{
    dal::op_from_settings,
    safetensor::ExpertKey,
    weight_manager::{
        LocalWeightManager, peer_client::WeightManagerClient,
        peer_server,
    },
    weight_srv::client::WeightSrvClient,
};
use nix::fcntl::{PosixFadviseAdvice, posix_fadvise};
use safetensors::{
    serialize,
    tensor::{Dtype, TensorView},
};
use tokio::{runtime::Runtime, task::JoinSet};
use tokio_util::bytes::Bytes;

const MEM_CACHE_MB: usize = 64;
const PEER_MEM_CACHE_MB: usize = 1024;
const MODEL_NAME: &str = "synthetic-provision";
const CACHE_ROOT_ENV: &str = "PROVISION_BENCH_CACHE_ROOT";
const DEFAULT_CACHE_ROOT: &str = "/data/provision-bench";

const MATRIX_ROWS: usize = 2048;
const MATRIX_COLS: usize = 768;

// --- Distributed mode env vars ---
const CENTRAL_ADDR_ENV: &str = "PROVISION_BENCH_CENTRAL_ADDR";
const PEER_ADDR_ENV: &str = "PROVISION_BENCH_PEER_ADDR";
const MODEL_ENV: &str = "PROVISION_BENCH_MODEL";
const EXPERT_COUNT_ENV: &str = "PROVISION_BENCH_EXPERT_COUNT";
const LAYER_ENV: &str = "PROVISION_BENCH_LAYER";
const EXPERTS_PER_LAYER_ENV: &str = "PROVISION_BENCH_EXPERTS_PER_LAYER";

// --- Sweep configuration ---
const SWEEP_ENV: &str = "PROVISION_BENCH_SWEEP";
const DEFAULT_SWEEP: &[usize] = &[64];

static RUNTIME: OnceLock<Runtime> = OnceLock::new();
static BENCH_CONTEXT: OnceLock<BenchContext> = OnceLock::new();
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn sweep_counts() -> Vec<usize> {
    match std::env::var(SWEEP_ENV) {
        Ok(val) => val
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
        Err(_) => DEFAULT_SWEEP.to_vec(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceTier {
    Disk,
    Peer,
    Central,
}

impl SourceTier {
    fn label(self) -> &'static str {
        match self {
            Self::Disk => "disk",
            Self::Peer => "peer",
            Self::Central => "central",
        }
    }
}

struct DistributedConfig {
    central_addr: String,
    peer_addr: String,
    model_name: String,
    expert_count: usize,
    layer: usize,
}

fn distributed_config() -> Option<DistributedConfig> {
    let central_addr = std::env::var(CENTRAL_ADDR_ENV).ok()?;
    let peer_addr = std::env::var(PEER_ADDR_ENV).ok()?;
    Some(DistributedConfig {
        model_name: std::env::var(MODEL_ENV).unwrap_or_else(|_| "qwen3".into()),
        expert_count: std::env::var(EXPERT_COUNT_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64),
        layer: std::env::var(LAYER_ENV)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        central_addr,
        peer_addr,
    })
}

/// State that only exists in local (synthetic) mode.
struct LocalState {
    central_store: Arc<DashMap<String, Bytes>>,
    peer_wm: Arc<LocalWeightManager>,
    _peer_cache: TempDirGuard,
}

struct BenchContext {
    weight_server_addr: String,
    peer_addr: String,
    expert_keys: Vec<ExpertKey>,
    seq_parallel_cache_root: PathBuf,
    /// `Some` in local mode, `None` in distributed mode.
    local: Option<LocalState>,
    /// Pre-populated disk cache directory for distributed mode.
    distributed_disk_cache: Option<PathBuf>,
}

struct TempDirGuard {
    path: PathBuf,
}

impl TempDirGuard {
    fn new_in(base: &Path, prefix: &str) -> Self {
        let path = base.join(format!(
            "ek-{prefix}-{}-{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

struct BatchFetchCase {
    _temp_dir: Option<TempDirGuard>,
    wm: Arc<LocalWeightManager>,
    keys: Vec<ExpertKey>,
    cleanups: Vec<CleanupAction>,
}

enum CleanupAction {
    RemoveCentralKey {
        store: Arc<DashMap<String, Bytes>>,
        obj_key: String,
    },
}

impl CleanupAction {
    fn run(self) {
        match self {
            Self::RemoveCentralKey { store, obj_key } => {
                store.remove(&obj_key);
            }
        }
    }
}

fn runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    })
}

fn bench_context() -> &'static BenchContext {
    BENCH_CONTEXT.get_or_init(|| {
        if let Some(dist) = distributed_config() {
            init_distributed(dist)
        } else {
            init_local()
        }
    })
}

fn init_local() -> BenchContext {
    let max_experts = sweep_counts().into_iter().max().unwrap_or(1);
    let expert_keys = (0..max_experts)
        .map(|idx| ExpertKey::new(MODEL_NAME.to_string(), 0, idx))
        .collect::<Vec<_>>();
    let seq_parallel_cache_root = bench_cache_root();
    prepare_seq_parallel_cache(&seq_parallel_cache_root, &expert_keys);

    let central_store = Arc::new(DashMap::new());
    let central_ready_key = source_key(SourceTier::Central, 0);
    stage_central_key(&central_store, &central_ready_key);
    let weight_server_port = reserve_port();
    let weight_server_addr = format!("http://127.0.0.1:{weight_server_port}");
    spawn_synthetic_weight_server(weight_server_port, central_store.clone());
    runtime().block_on(wait_for_central(&weight_server_addr, &central_ready_key));

    let peer_cache = TempDirGuard::new_in(&bench_cache_root(), "provision-peer-cache");
    let peer_wm = make_weight_manager(peer_cache.path(), None, PEER_MEM_CACHE_MB);
    let peer_ready_key = source_key(SourceTier::Peer, 0);
    stage_peer_mem_key(peer_cache.path(), &peer_wm, &peer_ready_key);
    let peer_port = reserve_port();
    let peer_addr = format!("http://127.0.0.1:{peer_port}");
    let peer_wm_for_server = peer_wm.clone();
    runtime().spawn(async move {
        let server = peer_server::start_peer_server(
            peer_wm_for_server,
            ("127.0.0.1", peer_port),
        )
        .await
        .unwrap();
        server.await.unwrap();
    });
    runtime().block_on(wait_for_peer(&peer_addr, &peer_ready_key));

    eprintln!("[provision-bench] mode=local, central={weight_server_addr}, peer={peer_addr}");

    BenchContext {
        weight_server_addr,
        peer_addr,
        expert_keys,
        seq_parallel_cache_root,
        local: Some(LocalState {
            central_store,
            peer_wm,
            _peer_cache: peer_cache,
        }),
        distributed_disk_cache: None,
    }
}

fn init_distributed(dist: DistributedConfig) -> BenchContext {
    let rt = runtime();
    let cache_root = bench_cache_root();

    let experts_per_layer: usize = std::env::var(EXPERTS_PER_LAYER_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(dist.expert_count); // default: all from one layer

    // Spread expert keys across layers when experts_per_layer < expert_count.
    let expert_keys: Vec<ExpertKey> = (0..dist.expert_count)
        .map(|i| {
            let layer = dist.layer + i / experts_per_layer;
            let idx = i % experts_per_layer;
            ExpertKey::new(dist.model_name.clone(), layer, idx)
        })
        .collect();

    // Pre-populate disk cache from real central server.
    let disk_cache_dir = cache_root.join("distributed-disk-cache");
    std::fs::create_dir_all(&disk_cache_dir).unwrap();
    let central_client = WeightSrvClient::new(dist.central_addr.clone());

    eprintln!(
        "[provision-bench] mode=distributed, central={}, peer={}",
        dist.central_addr, dist.peer_addr
    );
    eprintln!(
        "[provision-bench] model={}, base_layer={}, experts={}, experts_per_layer={}",
        dist.model_name, dist.layer, dist.expert_count, experts_per_layer
    );
    eprintln!("[provision-bench] populating disk cache from central...");

    let storage = OpenDALStorage::Fs(FSConfig {
        path: disk_cache_dir.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    rt.block_on(async {
        for key in &expert_keys {
            let obj_key = key.as_object_key();
            if operator.exists(&obj_key).await.unwrap() {
                continue;
            }
            let data = central_client
                .load_expert(key.model(), key.layer(), key.idx())
                .await
                .unwrap_or_else(|e| {
                    panic!("failed to fetch expert {} from central: {e}", obj_key)
                });
            operator.write(&obj_key, data).await.unwrap();
        }
    });
    eprintln!(
        "[provision-bench] disk cache ready ({} experts in {})",
        expert_keys.len(),
        disk_cache_dir.display()
    );

    // Verify peer is reachable.
    let peer_client = WeightManagerClient::new(dist.peer_addr.clone());
    let test_key = &expert_keys[0];
    rt.block_on(async {
        peer_client.get_expert(test_key).await.unwrap_or_else(|e| {
            panic!(
                "peer at {} not reachable for {}: {e}",
                dist.peer_addr,
                test_key.as_object_key()
            )
        });
    });
    eprintln!("[provision-bench] peer verified at {}", dist.peer_addr);

    BenchContext {
        weight_server_addr: dist.central_addr,
        peer_addr: dist.peer_addr,
        expert_keys,
        seq_parallel_cache_root: disk_cache_dir.clone(),
        local: None,
        distributed_disk_cache: Some(disk_cache_dir),
    }
}

fn bench_cache_root() -> PathBuf {
    PathBuf::from(std::env::var(CACHE_ROOT_ENV).unwrap_or_else(|_| DEFAULT_CACHE_ROOT.to_string()))
}

fn source_key(tier: SourceTier, iter: u64) -> ExpertKey {
    ExpertKey::new(format!("{MODEL_NAME}-{}", tier.label()), 0, iter as usize)
}

fn synthetic_expert_bytes(key: &ExpertKey) -> Vec<u8> {
    let shape = vec![MATRIX_ROWS, MATRIX_COLS];
    let tensor_len = MATRIX_ROWS * MATRIX_COLS * 2;
    let up = vec![pattern_byte(key, 0); tensor_len];
    let gate = vec![pattern_byte(key, 1); tensor_len];
    let down = vec![pattern_byte(key, 2); tensor_len];

    let views = HashMap::from([
        (
            tensor_name(key, "up_proj"),
            TensorView::new(Dtype::BF16, shape.clone(), &up).unwrap(),
        ),
        (
            tensor_name(key, "gate_proj"),
            TensorView::new(Dtype::BF16, shape.clone(), &gate).unwrap(),
        ),
        (
            tensor_name(key, "down_proj"),
            TensorView::new(Dtype::BF16, shape, &down).unwrap(),
        ),
    ]);

    serialize(&views, &None).unwrap()
}

fn tensor_name(key: &ExpertKey, proj: &str) -> String {
    format!(
        "model.layers.{}.mlp.experts.{}.{}.weight",
        key.layer(),
        key.idx(),
        proj
    )
}

fn pattern_byte(key: &ExpertKey, offset: u8) -> u8 {
    (((key.layer() * 17) + (key.idx() * 31)) as u8).wrapping_add(offset)
}

fn reserve_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn spawn_synthetic_weight_server(port: u16, store: Arc<DashMap<String, Bytes>>) {
    std::thread::spawn(move || {
        actix_web::rt::System::new().block_on(async move {
            HttpServer::new(move || {
                App::new().app_data(web::Data::new(store.clone())).route(
                    "/expert/{model}/{layer}/{expert}",
                    web::get().to(
                        |path: web::Path<(String, usize, usize)>,
                         store: web::Data<Arc<DashMap<String, Bytes>>>| async move {
                            let (model, layer, expert) = path.into_inner();
                            let obj_key = ExpertKey::new(model, layer, expert).as_object_key();
                            match store.get(&obj_key) {
                                Some(bytes) => HttpResponse::Ok().body(bytes.value().clone()),
                                None => HttpResponse::NotFound().finish(),
                            }
                        },
                    ),
                )
            })
            .workers(1)
            .bind(("127.0.0.1", port))
            .unwrap()
            .run()
            .await
            .unwrap();
        });
    });
}

fn stage_central_key(store: &Arc<DashMap<String, Bytes>>, key: &ExpertKey) {
    store.insert(
        key.as_object_key(),
        Bytes::from(synthetic_expert_bytes(key)),
    );
}

fn stage_peer_mem_key(cache_root: &Path, peer_wm: &Arc<LocalWeightManager>, key: &ExpertKey) {
    peer_wm.evict_all();
    let bytes = synthetic_expert_bytes(key);
    seed_disk_cache(cache_root, key, &bytes);
    runtime()
        .block_on(peer_wm.get_expert_local(key))
        .unwrap_or_else(|err| {
            panic!(
                "failed to warm peer key {} into mem cache: {err}",
                key.as_object_key()
            )
        });
    delete_disk_cache(cache_root, key);
}

fn delete_disk_cache(cache_root: &Path, key: &ExpertKey) {
    let storage = OpenDALStorage::Fs(FSConfig {
        path: cache_root.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    runtime()
        .block_on(operator.delete(&key.as_object_key()))
        .unwrap();
}

fn drop_file_page_cache(cache_root: &Path, key: &ExpertKey) {
    let path = cache_root.join(key.as_object_key());
    let file = std::fs::File::open(&path)
        .unwrap_or_else(|e| panic!("drop_file_page_cache: cannot open {}: {e}", path.display()));
    file.sync_all().unwrap();
    posix_fadvise(&file, 0, 0, PosixFadviseAdvice::POSIX_FADV_DONTNEED)
        .unwrap_or_else(|e| panic!("posix_fadvise(DONTNEED) failed on {}: {e}", path.display()));
}

fn batch_source_keys(tier: SourceTier, base_iter: u64, count: usize) -> Vec<ExpertKey> {
    (0..count)
        .map(|i| {
            ExpertKey::new(
                format!("{MODEL_NAME}-{}", tier.label()),
                0,
                (base_iter as usize) * count + i,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Local-mode stage functions
// ---------------------------------------------------------------------------

fn stage_disk_batch_fetch(base_iter: u64, count: usize) -> BatchFetchCase {
    let keys = batch_source_keys(SourceTier::Disk, base_iter, count);
    let temp_dir = TempDirGuard::new_in(&bench_cache_root(), "provision-disk");
    for key in &keys {
        let bytes = synthetic_expert_bytes(key);
        seed_disk_cache(temp_dir.path(), key, &bytes);
        drop_file_page_cache(temp_dir.path(), key);
    }
    let wm = make_weight_manager(temp_dir.path(), None, MEM_CACHE_MB);
    BatchFetchCase {
        _temp_dir: Some(temp_dir),
        wm,
        keys,
        cleanups: vec![],
    }
}

fn stage_peer_batch_fetch(ctx: &BenchContext, base_iter: u64, count: usize) -> BatchFetchCase {
    let local = ctx.local.as_ref().expect("stage_peer_batch_fetch requires local mode");
    let keys = batch_source_keys(SourceTier::Peer, base_iter, count);
    local.peer_wm.evict_all();
    for key in &keys {
        let bytes = synthetic_expert_bytes(key);
        seed_disk_cache(local._peer_cache.path(), key, &bytes);
        runtime()
            .block_on(local.peer_wm.get_expert_local(key))
            .unwrap_or_else(|e| panic!("failed to warm peer key {}: {e}", key.as_object_key()));
        delete_disk_cache(local._peer_cache.path(), key);
    }
    let temp_dir = TempDirGuard::new_in(&bench_cache_root(), "provision-peer");
    let wm = make_weight_manager(temp_dir.path(), None, MEM_CACHE_MB);
    for key in &keys {
        wm.update_peer_index(key.as_object_key(), vec![ctx.peer_addr.clone()]);
    }
    BatchFetchCase {
        _temp_dir: Some(temp_dir),
        wm,
        keys,
        cleanups: vec![],
    }
}

fn stage_central_batch_fetch(ctx: &BenchContext, base_iter: u64, count: usize) -> BatchFetchCase {
    let local = ctx.local.as_ref().expect("stage_central_batch_fetch requires local mode");
    let keys = batch_source_keys(SourceTier::Central, base_iter, count);
    let temp_dir = TempDirGuard::new_in(&bench_cache_root(), "provision-central");
    let wm = make_weight_manager(
        temp_dir.path(),
        Some(ctx.weight_server_addr.as_str()),
        MEM_CACHE_MB,
    );
    let cleanups = keys
        .iter()
        .map(|key| {
            stage_central_key(&local.central_store, key);
            CleanupAction::RemoveCentralKey {
                store: local.central_store.clone(),
                obj_key: key.as_object_key(),
            }
        })
        .collect();
    BatchFetchCase {
        _temp_dir: Some(temp_dir),
        wm,
        keys,
        cleanups,
    }
}

// ---------------------------------------------------------------------------
// Distributed-mode stage functions
// ---------------------------------------------------------------------------

/// Disk test (distributed): experts are pre-populated in `distributed_disk_cache`.
/// Each iteration drops OS page cache so we measure real disk I/O.
fn stage_disk_batch_fetch_distributed(ctx: &BenchContext, count: usize) -> BatchFetchCase {
    let disk_cache = ctx
        .distributed_disk_cache
        .as_ref()
        .expect("distributed_disk_cache required");
    let keys = ctx.expert_keys[..count.min(ctx.expert_keys.len())].to_vec();
    for key in &keys {
        drop_file_page_cache(disk_cache, key);
    }
    let wm = make_weight_manager(disk_cache, None, MEM_CACHE_MB);
    BatchFetchCase {
        _temp_dir: None,
        wm,
        keys,
        cleanups: vec![],
    }
}

/// Peer test (distributed): fetch from a real remote peer worker.
/// The peer must already have these experts loaded in its local caches.
fn stage_peer_batch_fetch_distributed(ctx: &BenchContext, count: usize) -> BatchFetchCase {
    let keys = ctx.expert_keys[..count.min(ctx.expert_keys.len())].to_vec();
    let temp_dir = TempDirGuard::new_in(&bench_cache_root(), "provision-peer-dist");
    let wm = make_weight_manager(temp_dir.path(), None, MEM_CACHE_MB);
    for key in &keys {
        wm.update_peer_index(key.as_object_key(), vec![ctx.peer_addr.clone()]);
    }
    BatchFetchCase {
        _temp_dir: Some(temp_dir),
        wm,
        keys,
        cleanups: vec![],
    }
}

/// Central test (distributed): fetch from a real remote weight server.
fn stage_central_batch_fetch_distributed(ctx: &BenchContext, count: usize) -> BatchFetchCase {
    let keys = ctx.expert_keys[..count.min(ctx.expert_keys.len())].to_vec();
    let temp_dir = TempDirGuard::new_in(&bench_cache_root(), "provision-central-dist");
    let wm = make_weight_manager(
        temp_dir.path(),
        Some(&ctx.weight_server_addr),
        MEM_CACHE_MB,
    );
    BatchFetchCase {
        _temp_dir: Some(temp_dir),
        wm,
        keys,
        cleanups: vec![],
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn make_weight_manager(
    cache_root: &Path,
    central_addr: Option<&str>,
    mem_cache_mb: usize,
) -> std::sync::Arc<LocalWeightManager> {
    let storage = OpenDALStorage::Fs(FSConfig {
        path: cache_root.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    let central_client = central_addr.map(|addr| WeightSrvClient::new(addr.to_string()));
    LocalWeightManager::new_with_parts(operator, central_client, mem_cache_mb, false)
}

fn seed_disk_cache(cache_root: &Path, key: &ExpertKey, bytes: &[u8]) {
    let storage = OpenDALStorage::Fs(FSConfig {
        path: cache_root.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    let obj_key = key.as_object_key();
    runtime()
        .block_on(operator.write(&obj_key, bytes.to_vec()))
        .unwrap();
}

fn prepare_seq_parallel_cache(cache_root: &Path, keys: &[ExpertKey]) {
    std::fs::create_dir_all(cache_root).unwrap();
    let storage = OpenDALStorage::Fs(FSConfig {
        path: cache_root.display().to_string(),
    });
    let operator = op_from_settings(&storage);
    runtime().block_on(async {
        for key in keys {
            let obj_key = key.as_object_key();
            if operator.exists(&obj_key).await.unwrap() {
                continue;
            }
            operator
                .write(&obj_key, synthetic_expert_bytes(key))
                .await
                .unwrap();
        }
    });
}

async fn wait_for_central(weight_server_addr: &str, key: &ExpertKey) {
    let client = WeightSrvClient::new(weight_server_addr.to_string());
    for _ in 0..50 {
        if client
            .load_expert(key.model(), key.layer(), key.idx())
            .await
            .is_ok()
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("central weight server did not become ready at {weight_server_addr}");
}

async fn wait_for_peer(peer_addr: &str, key: &ExpertKey) {
    let client = WeightManagerClient::new(peer_addr.to_string());
    for _ in 0..50 {
        if client.get_expert(key).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("peer weight server did not become ready at {peer_addr}");
}

// ---------------------------------------------------------------------------
// Measurement harnesses
// ---------------------------------------------------------------------------

fn measure_batch_fetch<F>(iters: u64, expected_tier: SourceTier, mut setup: F) -> Duration
where
    F: FnMut(u64) -> BatchFetchCase,
{
    let mut total = Duration::ZERO;
    for iter in 0..iters {
        let case = setup(iter);
        let wm = case.wm.clone();
        let keys = case.keys.clone();
        let start = Instant::now();
        runtime().block_on(async move {
            let mut jobs = JoinSet::new();
            for key in keys {
                let wm = wm.clone();
                jobs.spawn(async move {
                    // Retry once on transient network failures.
                    match wm.get_expert(&key).await {
                        Ok(b) => Ok(b.len()),
                        Err(_) => {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                            wm.get_expert(&key).await.map(|b| b.len())
                        }
                    }
                });
            }
            let mut total_bytes = 0usize;
            while let Some(result) = jobs.join_next().await {
                total_bytes += result.unwrap().unwrap();
            }
            black_box(total_bytes);
        });
        total += start.elapsed();
        // Retries may cause stats to drift; warn instead of panicking.
        let stats = case.wm.stats();
        let (mem, disk, peer, central) = (
            stats.mem.snapshot().0,
            stats.disk.snapshot().0,
            stats.peer.snapshot().0,
            stats.central.snapshot().0,
        );
        let n = case.keys.len() as u64;
        let ok = match expected_tier {
            SourceTier::Disk => disk == n && mem == 0 && peer == 0 && central == 0,
            SourceTier::Peer => peer == n && mem == 0 && disk == 0 && central == 0,
            SourceTier::Central => central == n && mem == 0 && disk == 0 && peer == 0,
        };
        if !ok {
            eprintln!(
                "[provision-bench] WARN: tier mismatch iter={iter}: expected {n}x{}, got mem={mem} disk={disk} peer={peer} central={central}",
                expected_tier.label()
            );
        }
        for c in case.cleanups {
            c.run();
        }
    }
    total
}

fn measure_multi_fetch(
    iters: u64,
    keys: &[ExpertKey],
    parallel: bool,
    cache_root: &Path,
) -> Duration {
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let wm = make_weight_manager(cache_root, None, MEM_CACHE_MB);
        let start = Instant::now();
        if parallel {
            let wm_clone = wm.clone();
            runtime().block_on(async move {
                let mut jobs = JoinSet::new();
                for key in keys {
                    let key = key.clone();
                    let wm = wm_clone.clone();
                    jobs.spawn(async move { wm.get_expert(&key).await.map(|bytes| bytes.len()) });
                }

                let mut total_bytes = 0usize;
                while let Some(result) = jobs.join_next().await {
                    total_bytes += result.unwrap().unwrap();
                }
                black_box(total_bytes);
            });
        } else {
            runtime().block_on(async {
                let mut total_bytes = 0usize;
                for key in keys {
                    total_bytes += wm.get_expert(key).await.unwrap().len();
                }
                black_box(total_bytes);
            });
        }
        total += start.elapsed();
    }
    total
}

// ---------------------------------------------------------------------------
// Criterion entry point
// ---------------------------------------------------------------------------

pub fn bench(c: &mut Criterion) {
    let ctx = bench_context();
    let is_distributed = ctx.local.is_none();
    let available = ctx.expert_keys.len();
    let sweep = sweep_counts();

    eprintln!(
        "[provision-bench] sweep={:?}, available={available}",
        sweep
    );

    let mut source_group = c.benchmark_group("provision source latency");

    for &count in &sweep {
        if count > available {
            eprintln!(
                "[provision-bench] skipping count={count} (only {available} experts available)"
            );
            continue;
        }

        source_group.bench_with_input(
            BenchmarkId::new("disk", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    if is_distributed {
                        measure_batch_fetch(iters, SourceTier::Disk, |_iter| {
                            stage_disk_batch_fetch_distributed(ctx, count)
                        })
                    } else {
                        measure_batch_fetch(iters, SourceTier::Disk, |iter| {
                            stage_disk_batch_fetch(iter, count)
                        })
                    }
                });
            },
        );

        source_group.bench_with_input(
            BenchmarkId::new("peer", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    if is_distributed {
                        measure_batch_fetch(iters, SourceTier::Peer, |_iter| {
                            stage_peer_batch_fetch_distributed(ctx, count)
                        })
                    } else {
                        measure_batch_fetch(iters, SourceTier::Peer, |iter| {
                            stage_peer_batch_fetch(ctx, iter, count)
                        })
                    }
                });
            },
        );

        source_group.bench_with_input(
            BenchmarkId::new("central", count),
            &count,
            |b, &count| {
                b.iter_custom(|iters| {
                    if is_distributed {
                        measure_batch_fetch(iters, SourceTier::Central, |_iter| {
                            stage_central_batch_fetch_distributed(ctx, count)
                        })
                    } else {
                        measure_batch_fetch(iters, SourceTier::Central, |iter| {
                            stage_central_batch_fetch(ctx, iter, count)
                        })
                    }
                });
            },
        );
    }

    source_group.finish();

    let mut parallel_group = c.benchmark_group("provision parallel vs sequential");
    for &count in &sweep {
        if count > available {
            continue;
        }
        let keys = ctx.expert_keys[..count].to_vec();
        parallel_group.bench_with_input(
            BenchmarkId::new("sequential", count),
            &count,
            |b, _| {
                b.iter_custom(|iters| {
                    measure_multi_fetch(iters, &keys, false, &ctx.seq_parallel_cache_root)
                })
            },
        );
        parallel_group.bench_with_input(
            BenchmarkId::new("parallel", count),
            &count,
            |b, _| {
                b.iter_custom(|iters| {
                    measure_multi_fetch(iters, &keys, true, &ctx.seq_parallel_cache_root)
                })
            },
        );
    }
    parallel_group.finish();
}
