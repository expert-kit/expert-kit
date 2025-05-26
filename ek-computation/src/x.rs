use std::{path::PathBuf, sync::Arc};

use super::ffn::Device;
use std::sync::atomic::{AtomicUsize, Ordering};

static INSTANCE_COUNTER: AtomicUsize = AtomicUsize::new(0);
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
    pub device: Device,
}

impl Default for EKInstance {
    fn default() -> Self {
        let settings = get_ek_settings();
        let count = INSTANCE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let device = if count < 1536 {
            Device::CUDA(1)
        } else if count < 3072 {
            Device::CUDA(2)
        } else if count < 4608 {
            Device::CUDA(3)
        } else {
            Device::CPU
        };
        // let device = Device::CPU;
        Self {
            dim: settings.inference.hidden_dim,
            hidden: settings.inference.intermediate_dim,
            backend: ExpertBackendType::Torch,
            device,
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

#[cfg(test)]
mod test {
    use tch::Cuda;
    #[test]
    fn test_env() {
        println!("CUDA Device Count: {}", Cuda::device_count());
        println!("CUDA available: {}", Cuda::is_available());
    }

    #[test]
    fn test_force_cuda() {
        let _ = tch::Tensor::zeros(&[1, 2], (tch::Kind::Float, tch::Device::Cuda(0)));
        println!("Tensor on CUDA successfully created.");
    }
}
