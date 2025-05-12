use std::path::PathBuf;

use clap::ValueEnum;
use ek_base::config::get_ek_settings;

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
        let settings = get_ek_settings();
        Self {
            dim: settings.inference.hidden_dim,
            hidden: settings.inference.intermediate_dim,
            backend: ExpertBackendType::Torch,
        }
    }
}

pub fn test_root() -> PathBuf {
    let root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(root.to_owned())
}
