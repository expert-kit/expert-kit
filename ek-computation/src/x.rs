use clap::ValueEnum;

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
            dim: 2048,
            hidden: 7168,
            backend: ExpertBackendType::Torch,
        }
    }
}
