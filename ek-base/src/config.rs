use std::path::Path;

use config::{Config, Environment};
use once_cell::sync::OnceCell;
use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct Addr {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct ControllerSettings {
    pub broadcast: Addr,
    pub listen: Addr,
}

#[derive(Debug, Deserialize, Clone)]
#[allow(unused)]
pub struct WorkerSettings {
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
    FS(FSConfig),
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

pub fn get_ek_settings() -> &'static Settings {
    static CONFIG: OnceCell<Settings> = OnceCell::new();
    let res = CONFIG.get_or_init(|| {
        let mut settings = Config::builder().add_source(config::Environment::with_prefix("EK"));
        let possible_config_files = vec!["/etc/expert-kit/config.yaml"];
        for path in possible_config_files {
            if Path::new(path).exists() {
                settings = settings.add_source(config::File::with_name(path));
            }
        }
        settings = settings.add_source(
            Environment::with_prefix("EK")
                .try_parsing(false)
                .separator("_"),
        );
        let settings = settings.build().unwrap();

        settings.try_deserialize::<Settings>().unwrap()
    });
    res
}

#[cfg(test)]
mod test {
    use super::get_ek_settings;

    #[test]
    fn basic_test() {
        get_ek_settings();
    }
}
