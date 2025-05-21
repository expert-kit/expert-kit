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
    pub hidden: usize,
    pub intermediate: usize,
    pub backend: ExpertBackendType,
}

impl Default for EKInstance {
    fn default() -> Self {
        let settings = get_ek_settings();
        Self {
            hidden: settings.inference.hidden_dim,
            intermediate: settings.inference.intermediate_dim,
            backend: ExpertBackendType::Torch,
        }
    }
}

pub fn test_root() -> PathBuf {
    let root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(root.to_owned())
}

type GracefulChannelPair = (Sender<()>, Arc<Mutex<Receiver<()>>>);

pub fn get_graceful_shutdown_ch() -> GracefulChannelPair {
    static GRACEFUL_SHUTDOWN: OnceCell<GracefulChannelPair> = OnceCell::new();
    let res = GRACEFUL_SHUTDOWN.get_or_init(|| {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        (tx, Arc::new(Mutex::new(rx)))
    });
    (res.0.clone(), res.1.clone())
}
