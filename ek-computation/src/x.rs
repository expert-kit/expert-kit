use std::{
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

use clap::ValueEnum;
use ek_base::config::get_ek_settings;
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};

use super::backend::Device;

static INSTANCE_COUNTER: AtomicUsize = AtomicUsize::new(0);
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
    pub device: Device,
}

impl Default for EKInstance {
    fn default() -> Self {
        let settings = get_ek_settings();
        let _ = INSTANCE_COUNTER.fetch_add(1, Ordering::SeqCst);

        let device = Device::from(settings.worker.device.as_str());

        Self {
            hidden: settings.inference.hidden_dim,
            intermediate: settings.inference.intermediate_dim,
            backend: ExpertBackendType::Torch,
            device,
        }
    }
}

pub fn test_root() -> PathBuf {
    let root = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(root.to_owned())
}

type GracefulChannelPair = (Sender<()>, Arc<Mutex<Receiver<()>>>);

pub fn get_graceful_shutdown_ch() -> GracefulChannelPair {
    static GRACEFUL_SHUTDOWN: OnceLock<GracefulChannelPair> = OnceLock::new();
    let res = GRACEFUL_SHUTDOWN.get_or_init(|| {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        (tx, Arc::new(Mutex::new(rx)))
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
        let _ = tch::Tensor::zeros([1, 2], (tch::Kind::Float, tch::Device::Cuda(0)));
        println!("Tensor on CUDA successfully created.");
    }
}
