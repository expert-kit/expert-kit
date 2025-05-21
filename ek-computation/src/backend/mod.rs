use safetensors::tensor::TensorView;

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

#[derive(Clone, Copy)]
pub enum Device {
    CPU,
}

pub trait EkTensor: Sized {
    fn rand(shape: Vec<usize>, dtype: DType, dev: Device) -> Self;
    fn stack(tensors: &[Self], dim: usize) -> Self;
    fn shape(&self) -> Vec<usize>;
    fn serialize(&self) -> Vec<u8>;
    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self;
    fn from_tensor_view(tv: &TensorView<'_>) -> Self;
}

pub trait FromSafeTensor
where
    Self: Sized + EkTensor,
{
    fn lookup_suffix(st: &safetensors::SafeTensors, name: &[&str]) -> Option<Self>;
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
