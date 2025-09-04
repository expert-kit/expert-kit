use safetensors::tensor::TensorView;

pub mod ggml;
pub mod ort;
pub mod torch;

#[derive(Clone, Copy)]
pub enum DType {
    Uint8,
    Int8,
    Int16,
    BFloat16,
    Float,
    Float8e4m3fn,
    Float8e4m3fnuz,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::Uint8 => 1,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::BFloat16 => 2,
            DType::Float => 4,
            DType::Float8e4m3fn | DType::Float8e4m3fnuz => 1, // Assuming these are packed formats
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(usize),
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU => write!(f, "CPU"),
            Device::CUDA(idx) => write!(f, "CUDA({idx})"),
        }
    }
}

impl From<&str> for Device {
    fn from(value: &str) -> Self {
        let str_dev = value.to_lowercase();
        if str_dev == "cpu" {
            Device::CPU
        } else if let Some(str_dev) = str_dev.strip_prefix("cuda") {
            let idx = str_dev.parse::<usize>().unwrap_or(0);
            Device::CUDA(idx)
        } else {
            panic!("Unsupported device: {value}");
        }
    }
}

pub trait EkTensor: Sized {
    fn rand(shape: Vec<usize>, dtype: DType, dev: Device) -> Self;
    fn shape(&self) -> Vec<usize>;
    fn serialize(&self) -> Vec<u8>;
    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self;
    fn from_tensor_view(tv: &TensorView<'_>) -> Self;
    fn device(&self) -> Device;
    fn to_device(&self, dev: Device) -> Self;
}

pub trait FromSafeTensor
where
    Self: Sized + EkTensor,
{
    fn lookup_suffix(st: &safetensors::SafeTensors, name: &[&str], dev: Device) -> Option<Self>;
}

impl From<safetensors::Dtype> for DType {
    fn from(value: safetensors::Dtype) -> Self {
        match value {
            safetensors::Dtype::U16 => DType::Uint8,
            safetensors::Dtype::U8 => DType::Uint8,
            safetensors::Dtype::I8 => DType::Int8,
            safetensors::Dtype::BF16 => DType::BFloat16,
            _ => unimplemented!(),
        }
    }
}
