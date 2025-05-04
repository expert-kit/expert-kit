use std::{collections::HashMap, path::Path, sync::Arc};

use config::Config;
use once_cell::sync::OnceCell;

pub fn get_config_key(key: &str) -> &str {
    static CONFIG: OnceCell<Arc<HashMap<String, String>>> = OnceCell::new();
    let res = CONFIG.get_or_init(|| {
        let mut settings = Config::builder().add_source(config::Environment::with_prefix("EK"));
        let possible_config_files = vec!["/etc/expert-kit/config.yaml"];
        for path in possible_config_files {
            if Path::new(path).exists() {
                settings = settings.add_source(config::File::with_name(path));
            }
        }
        let settings = settings.build().unwrap();
        let global_config = settings
            .try_deserialize::<HashMap<String, String>>()
            .unwrap();
        Arc::new(global_config)
    });

    let value = res
        .get(key)
        .unwrap_or_else(|| panic!("Key {} not found", key))
        .as_str();
    value
}
