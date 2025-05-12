use std::{net::SocketAddr, path::Path, sync::LazyLock};

use config::{Config, Environment};
use once_cell::sync::OnceCell;
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
    pub intermediate_dim: usize
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

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerSettings {
    pub id: Option<String>,
    pub listen: String,
    pub broadcast: String,
    pub ports: WorkerPorts,
    pub device: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WeightSettings {
    pub server: Option<WeightServerSettings>,
    pub cache: OpenDALStorage,
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
pub struct Settings {
    pub inference: InferenceSettings,
    pub db: DBSettings,
    pub weight: WeightSettings,
    pub controller: ControllerSettings,
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
    static CONFIG: OnceCell<Settings> = OnceCell::new();
    let res = CONFIG.get_or_init(|| {
        let mut settings = Config::builder();
        let candidates = src.iter().chain(["/etc/expert-kit/config.yaml"].iter());

        for path in candidates {
            if Path::new(path).exists() {
                log::info!("Loading config from {}", path);
                settings = settings.add_source(config::File::with_name(path));
                break;
            }
        }
        settings = settings.add_source(env_source());
        let settings = settings.build().unwrap();

        settings.try_deserialize::<Settings>().unwrap()
    });
    res
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

controller:
  listen: 0.0.0.0
  broadcast: localhost
  ports:
    intra: 5001
    inter: 5002
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
        assert_eq!(res.hidden_dim, 2048);
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
        assert_eq!(res.worker.id.unwrap(), "override_test");
    }
}
