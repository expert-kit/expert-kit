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
pub struct ControllerSettings {
    pub inter_listen: Addr,
    pub intra_listen: Addr,
    pub broadcast: Addr,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerSettings {
    pub id: Option<String>,
    pub listen: Addr,
    pub broadcast: Addr,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WeightSettings {
    pub server: WeightServerSettings,
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
    pub db_dsn: String,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
    pub instance_name: String,
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
pub fn get_ek_settings() -> &'static Settings {
    static CONFIG: OnceCell<Settings> = OnceCell::new();
    let res = CONFIG.get_or_init(|| {
        let mut settings = Config::builder();
        let possible_config_files = vec!["/etc/expert-kit/config.yaml"];
        for path in possible_config_files {
            if Path::new(path).exists() {
                settings = settings.add_source(config::File::with_name(path));
            }
        }
        settings = settings.add_source(env_source());
        let settings = settings.build().unwrap();

        settings.try_deserialize::<Settings>().unwrap()
    });
    res
}

#[cfg(test)]
mod test {
    use config::{File, FileFormat};

    use crate::config::env_source;

    use super::Settings;

    fn get_example_config() -> &'static str {
        r#"
db_dsn: postgres://dev:dev@localhost:5432/dev
hidden_dim: 2048
intermediate_dim: 768
instance_name: qwen3_moe_30b_local_test

weight:
  server:
    addr: http://?
  cache:
    Fs:
      path: /

worker:
  id: local_test
  listen:
    host: 0.0.0.0
    port: 51234
  broadcast:
    host: 0.0.0.0
    port: 51234

controller:
  intra_listen:
    host: 0.0.0.0
    port: 5002
  inter_listen:
    host: 0.0.0.0
    port: 5002
  broadcast:
    host: 0.0.0.0
    port: 5002"#
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
