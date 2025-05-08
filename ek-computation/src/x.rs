use std::path::PathBuf;

use clap::ValueEnum;
use ek_base::config::get_config_key;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum ExpertBackendType {
    Torch,
    Onnx,
}

impl From<&str> for ExpertBackendType {
    fn from(value: &str) -> Self {
        match value {
            "torch" => ExpertBackendType::Torch,
            "ort" => ExpertBackendType::Onnx,
            _ => unimplemented!(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct EKInstance {
    pub dim: usize,
    pub hidden: usize,
    pub backend: ExpertBackendType,
}

impl Default for EKInstance {
    fn default() -> Self {
        Self {
            dim: get_config_key("hidden_dim").parse().unwrap(),
            hidden: get_config_key("intermediate_dim").parse().unwrap(),
            backend: ExpertBackendType::Torch,
        }
    }
}

pub fn test_root() -> PathBuf {
    let root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(root.to_owned())
}
