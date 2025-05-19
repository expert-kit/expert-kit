use std::{path::PathBuf, sync::Arc};

use clap::ValueEnum;
use ek_base::config::get_ek_settings;
use once_cell::sync::OnceCell;
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};

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

pub fn get_graceful_shutdown_ch() -> (Sender<()>, Arc<Mutex<Receiver<()>>>) {
    static GRACEFUL_SHUTDOWN: OnceCell<(Sender<()>, Arc<Mutex<Receiver<()>>>)> = OnceCell::new();
    let res = GRACEFUL_SHUTDOWN.get_or_init(|| {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        return (tx, Arc::new(Mutex::new(rx)));
    });
    (res.0.clone(), res.1.clone())
}
