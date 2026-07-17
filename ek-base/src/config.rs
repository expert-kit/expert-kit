use std::{
    net::SocketAddr,
    path::Path,
    sync::{LazyLock, OnceLock},
};

use config::{Config, Environment};
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct Addr {
    pub host: String,
    pub port: u16,
}

impl Addr {
    pub fn to_socket_addr(&self) -> SocketAddr {
        format!("{}:{}", self.host, self.port).parse().unwrap()
    }
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct InferenceSettings {
    pub instance_name: String,
    pub model_name: String,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct DBSettings {
    pub db_dsn: String,
    pub max_conn_size: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct ControllerSettings {
    pub listen: String,
    pub broadcast: String,
    pub ports: ControllerPorts,
    /// Fault detection and recovery timing parameters
    #[serde(default)]
    pub fault_detection: FaultDetectionSettings,
    /// Hot-expert replication settings (opt-in, default: disabled)
    #[serde(default)]
    pub replication: ReplicationSettings,
    /// Cloud node provisioning settings (opt-in, default: disabled)
    #[serde(default)]
    pub provisioning: ProvisioningSettings,
    /// Progressive expert assignment for new workers (opt-in, default: disabled)
    #[serde(default)]
    pub scaling: ScalingSettings,
}

/// Timing parameters for detecting node failures and running recovery.
///
/// These control how quickly the system detects a dead worker and how the
/// poller rebuilds the routing table after node removal.
#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct FaultDetectionSettings {
    /// Seconds without a heartbeat before the controller drops the stream
    /// and starts recovery. Must be > heartbeat interval (3 s). (default: 10)
    #[serde(default = "default_heartbeat_timeout_secs")]
    pub heartbeat_timeout_secs: u64,
    /// Seconds of heartbeat silence before a node is excluded from active
    /// queries (routing, recovery target selection). Should be ≥
    /// heartbeat_timeout_secs. (default: 60)
    #[serde(default = "default_node_active_threshold_secs")]
    pub node_active_threshold_secs: u64,
    /// Interval between poller ticks in seconds. Each tick rebuilds the
    /// routing table from DB and runs the elastic manager. (default: 5)
    #[serde(default = "default_poller_interval_secs")]
    pub poller_interval_secs: u64,
}

impl Default for FaultDetectionSettings {
    fn default() -> Self {
        Self {
            heartbeat_timeout_secs: default_heartbeat_timeout_secs(),
            node_active_threshold_secs: default_node_active_threshold_secs(),
            poller_interval_secs: default_poller_interval_secs(),
        }
    }
}

fn default_heartbeat_timeout_secs() -> u64 {
    10
}
fn default_node_active_threshold_secs() -> u64 {
    60
}
fn default_poller_interval_secs() -> u64 {
    5
}

/// Controls reactive replication of hot experts onto additional workers.
///
/// Hotspot detection is rate-based: the controller compares each expert's request count
/// over a 3-tick sliding window (≈ 15 s) against `hotspot_threshold`.
/// `target_rate_per_worker` encodes the maximum requests per window a single worker can
/// sustain for one expert; desired replicas = ceil(rate / target_rate_per_worker).
/// In the spot-cluster context, where GPU throughput is the binding constraint, this
/// per-worker throughput threshold serves as the operational SLO.
#[derive(Debug, Deserialize, Clone, Default)]
#[allow(unused)]
pub struct ReplicationSettings {
    /// Enable reactive replication (default: false)
    #[serde(default)]
    pub enabled: bool,
    /// Total requests across the 3-tick window (≈ 15 s) above which an expert is
    /// considered a hotspot. Must be > 0. (default: 5)
    #[serde(default = "default_hotspot_threshold")]
    pub hotspot_threshold: u64,
    /// Max requests per window a single worker can handle for one expert (per-worker
    /// throughput threshold). Desired replicas = ceil(rate / target_rate_per_worker).
    /// (default: 5)
    #[serde(default = "default_target_rate_per_worker")]
    pub target_rate_per_worker: u64,
    /// Hard cap on replicas per expert (default: 3)
    #[serde(default = "default_max_replicas")]
    pub max_replicas: u32,
}

fn default_hotspot_threshold() -> u64 {
    5
}
fn default_target_rate_per_worker() -> u64 {
    5
}
fn default_max_replicas() -> u32 {
    3
}

/// Controls automatic provisioning of new cloud nodes on capacity shortfall.
#[derive(Debug, Deserialize, Clone, Default)]
#[allow(unused)]
pub struct ProvisioningSettings {
    /// Enable capacity-shortfall provisioning (default: false)
    #[serde(default)]
    pub enabled: bool,
    /// Path to provisioner script; invoked with --count --model --instance args
    pub script: Option<String>,
    /// Minimum seconds between successive provisioner invocations (default: 120)
    #[serde(default = "default_provision_cooldown_secs")]
    pub cooldown_secs: u64,
}

fn default_provision_cooldown_secs() -> u64 {
    120
}

/// Controls progressive expert assignment when a new worker registers.
#[derive(Debug, Deserialize, Clone, Default)]
#[allow(unused)]
pub struct ScalingSettings {
    /// Enable automatic expert assignment on worker join (default: false)
    #[serde(default)]
    pub auto_assign: bool,
    /// Pause between loading stripes in ms; 0 = bulk load (default: 0)
    #[serde(default)]
    pub step_delay_ms: u64,
    /// Experts per layer per stripe; 0 = load all at once (default: 0)
    #[serde(default)]
    pub experts_per_step: usize,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct ControllerPorts {
    pub intra: u16,
    pub inter: u16,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerPorts {
    pub main: u16,
}

impl Default for WorkerPorts {
    fn default() -> Self {
        Self { main: 51234 }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct CpuAffinityConfig {
    // List of CPU core IDs to bind the worker to
    // Example: [0, 1, 2, 3] to bind to cores 0-3
    pub cores: Option<Vec<usize>>,
    // NUMA node IDs to bind the worker to
    // Example: [0] for NUMA node 0, [0, 1] for NUMA nodes 0 and 1
    pub numa_nodes: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerAdvancedSettings {
    // CPU affinity and NUMA configuration
    pub cpu_affinity: Option<CpuAffinityConfig>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerSettings {
    #[serde(default = "default_worker_id")]
    pub id: String,
    #[serde(default = "default_worker_channel")]
    pub channel: String,
    pub listen: String,
    pub broadcast: String,
    pub ports: WorkerPorts,
    pub device: String,
    #[serde(default = "default_backend")]
    pub backend: String,
    #[serde(default)]
    pub drop_cache: bool,
    #[serde(default = "default_worker_metrics")]
    pub metrics: String,
    pub advanced: Option<WorkerAdvancedSettings>,
    /// Memory capacity of this worker in megabytes, reported in heartbeats (default: 4096)
    #[serde(default = "default_worker_mem_capacity_mb")]
    pub mem_capacity_mb: u64,
    /// Grace period in seconds after SIGTERM before forced exit; controller migrates
    /// unique experts within this window (default: 30; set 270 for Aliyun 5-min window)
    #[serde(default = "default_shutdown_grace_secs")]
    pub shutdown_grace_secs: u64,
}

impl Default for WorkerSettings {
    fn default() -> Self {
        Self {
            id: default_worker_id(),
            channel: default_worker_channel(),
            listen: "0.0.0.0".to_string(),
            broadcast: "127.0.0.1".to_string(),
            ports: WorkerPorts::default(),
            device: "cpu".to_string(),
            backend: default_backend(),
            drop_cache: false,
            metrics: default_worker_metrics(),
            advanced: None,
            mem_capacity_mb: default_worker_mem_capacity_mb(),
            shutdown_grace_secs: default_shutdown_grace_secs(),
        }
    }
}

fn default_worker_mem_capacity_mb() -> u64 {
    4096
}

fn default_shutdown_grace_secs() -> u64 {
    30
}

fn default_worker_metrics() -> String {
    "0.0.0.0:9091".to_string()
}

fn default_worker_channel() -> String {
    "grpc".to_string()
}

fn default_backend() -> String {
    "torch".to_string()
}

fn default_worker_id() -> String {
    use gethostname::gethostname;
    gethostname().into_string().unwrap()
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WeightSettings {
    pub server: Option<WeightServerSettings>,
    pub cache: OpenDALStorage,
    /// gRPC listen address for the LocalWeightManager server (e.g., "0.0.0.0:5004")
    #[serde(default = "default_wm_listen")]
    pub wm_listen: String,
    /// External address peers use to reach this node's LocalWeightManager (e.g., "hostname:5004")
    #[serde(default = "default_wm_broadcast")]
    pub wm_broadcast: String,
    /// Maximum memory cache size in megabytes for LocalWeightManager (default: 4096 MB)
    #[serde(default = "default_mem_cache_mb")]
    pub mem_cache_mb: usize,
}

fn default_wm_listen() -> String {
    "0.0.0.0:5004".to_string()
}

fn default_wm_broadcast() -> String {
    "localhost:5004".to_string()
}

fn default_mem_cache_mb() -> usize {
    4096
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WeightServerSettings {
    pub addr: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct S3Config {
    pub access_key_id: String,
    pub access_key_secret: String,
    pub endpoint: String,
    pub region: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct FSConfig {
    pub path: String,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub enum OpenDALStorage {
    Fs(FSConfig),
    S3(S3Config),
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct LogSettings {
    #[serde(default = "default_log_enable")]
    pub enable: bool,
    #[serde(default = "default_log_root")]
    pub root: String,
}
fn default_log_enable() -> bool {
    false
}

fn default_log_root() -> String {
    "/var/log/expert-kit".to_string()
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct Settings {
    pub inference: InferenceSettings,
    pub db: DBSettings,
    pub weight: WeightSettings,
    pub controller: ControllerSettings,
    #[serde(default)]
    pub worker: WorkerSettings,
}

pub fn env_source() -> Environment {
    static ENV_SRC: LazyLock<Environment> = std::sync::LazyLock::new(|| {
        Environment::with_prefix("EK")
            .try_parsing(false)
            .separator("_")
    });
    ENV_SRC.clone()
}
pub fn get_ek_settings_base(src: &[&str]) -> &'static Settings {
    static CONFIG: OnceLock<Settings> = OnceLock::new();

    (CONFIG.get_or_init(|| {
        let mut settings = Config::builder();
        let candidates = src.iter().chain(["/etc/expert-kit/config.yaml"].iter());

        for path in candidates {
            if Path::new(path).exists() {
                log::info!("Loading config from {path}");
                settings = settings.add_source(config::File::with_name(path));
                break;
            }
        }
        settings = settings.add_source(env_source());
        let settings = settings.build().unwrap();

        settings.try_deserialize::<Settings>().unwrap()
    })) as _
}

pub fn get_ek_settings() -> &'static Settings {
    get_ek_settings_base(&[])
}

#[cfg(test)]
mod test {
    use config::{File, FileFormat};

    use crate::config::env_source;

    use super::Settings;

    fn get_example_config() -> &'static str {
        r#"
inference:
  instance_name: qwen3_moe_30b_local_test
  model_name: ds-tiny
  hidden_dim: 2048
  intermediate_dim: 768
  
db:
  db_dsn: postgres://dev:dev@localhost:5432/dev
  max_conn_size: 32

weight:
  server:
    addr: http://?
  cache:
    Fs:
      path: /

worker:
  id: local_test
  listen: 0.0.0.0
  broadcast: 0.0.0.0
  ports:
    main: 51234
  device: cpu
  advanced:
    cpu_affinity:
      cores: [0, 1, 2, 3]
      numa_nodes: [0, 1]

controller:
  listen: 0.0.0.0
  broadcast: localhost
  ports:
    intra: 5001
    inter: 5002
  registry_backend: Grpc
"#
    }

    #[test]
    fn basic_test() {
        let example_yaml = get_example_config();
        let config = config::Config::builder()
            .add_source(File::from_str(example_yaml, FileFormat::Yaml))
            .build()
            .unwrap();
        let res = config.try_deserialize::<Settings>().unwrap();
        assert_eq!(res.inference.hidden_dim, 2048);
        assert_eq!(res.worker.metrics, "0.0.0.0:9091");

        // Test advanced settings
        let advanced = res.worker.advanced.as_ref().unwrap();
        let cpu_affinity = advanced.cpu_affinity.as_ref().unwrap();
        assert_eq!(cpu_affinity.cores.as_ref().unwrap(), &vec![0, 1, 2, 3]);
        assert_eq!(cpu_affinity.numa_nodes.as_ref().unwrap(), &vec![0, 1]);
    }

    #[test]
    fn test_env_override() {
        let example_yaml = get_example_config();
        unsafe { std::env::set_var("EK_WORKER_ID", "override_test") };
        let config = config::Config::builder()
            .add_source(File::from_str(example_yaml, FileFormat::Yaml))
            .add_source(env_source())
            .build()
            .unwrap();
        let res = config.try_deserialize::<Settings>().unwrap();
        assert_eq!(res.worker.id, "override_test");
    }

    #[test]
    fn controller_config_does_not_require_legacy_worker_settings() {
        let example_yaml = r#"
inference:
  instance_name: qwen3-demo
  model_name: Qwen3-30B-A3B
  hidden_dim: 2048
  intermediate_dim: 768

db:
  db_dsn: postgres://dev:dev@localhost:5432/dev
  max_conn_size: 32

weight:
  server:
    addr: http://localhost:6543
  cache:
    Fs:
      path: /tmp/expert-kit/cache

controller:
  listen: 0.0.0.0
  broadcast: localhost
  ports:
    intra: 5001
    inter: 5002
"#;
        let config = config::Config::builder()
            .add_source(File::from_str(example_yaml, FileFormat::Yaml))
            .build()
            .unwrap();

        let settings = config.try_deserialize::<Settings>().unwrap();

        assert_eq!(settings.worker.listen, "0.0.0.0");
        assert_eq!(settings.worker.ports.main, 51234);
    }
}
