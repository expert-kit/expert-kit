use ek_base::error::{EKError, EKResult};
use expert_ort::OnnxFFN;
use expert_torch::{TchTensor, TorchFFN};
use safetensors::tensor::TensorView;
use tch::Tensor;

use crate::{
    tch_safetensors::dtype_to_tch_kind,
    x::{self, EKInstance},
};

pub mod expert_ort;
pub mod expert_torch;

pub struct ExpertShape {
    pub dim: usize,
    pub hidden: usize,
}

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
impl From<tch::Kind> for DType {
    fn from(k: tch::Kind) -> Self {
        match k {
            tch::Kind::Uint8 => DType::Uint8,
            tch::Kind::Int16 => DType::Int16,
            tch::Kind::Int8 => DType::Int8,
            tch::Kind::BFloat16 => DType::BFloat16,
            tch::Kind::Float8e4m3fn => DType::Float8e4m3fn,
            tch::Kind::Float8e4m3fnuz => DType::Float8e4m3fnuz,
            tch::Kind::Float => DType::Float,
            _ => unimplemented!(),
        }
    }
}

pub trait EkTensor: Sized {
    fn rand(shape: Vec<usize>, dtype: DType, dev: Device) -> Self;
    fn cat(tensors: &[Self], dim: usize) -> Self;
    fn serialize(&self) -> Vec<u8>;
    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self;
    fn from_tensor_view(tv: &TensorView<'_>) -> Self;
}

pub trait FromSafeTensor
where
    Self: Sized + EkTensor,
{
    fn lookup_suffix(st: &safetensors::SafeTensors, name: &str) -> Option<Self> {
        let idx = st.names().iter().position(|x| x.ends_with(name));
        let tensors = st.tensors();
        idx.map(|x| {
            let (_name, view) = tensors.get(x).unwrap();
            let size: Vec<usize> = view.shape().iter().map(|&x| x as usize).collect();
            let kind: DType = DType::from(dtype_to_tch_kind(view.dtype()).unwrap());
            Self::from_raw(view.data(), &size, kind);
        });
        None
    }
}

pub struct ExpertWeight<T>
where
    T: EkTensor,
{
    pub up_w: T,
    pub up_b: Option<T>,
    pub down_w: T,
    pub down_b: Option<T>,
    pub gate_w: T,
    pub gate_b: Option<T>,
}
impl<T: EkTensor + FromSafeTensor> ExpertWeight<T> {
    pub fn from_safetensor(st: &safetensors::SafeTensors) -> EKResult<Self> {
        let name = "down_proj.weight";
        let down_w =
            T::lookup_suffix(st, name).ok_or(EKError::ExpertWeightNotFound(name.into()))?;
        let name = "up_proj.weight";
        let up_w = T::lookup_suffix(st, name).ok_or(EKError::ExpertWeightNotFound(name.into()))?;
        let name = "gate_proj.weight";
        let gate_w =
            T::lookup_suffix(st, name).ok_or(EKError::ExpertWeightNotFound(name.into()))?;
        Ok(Self {
            up_w,
            up_b: None,
            down_w,
            down_b: None,
            gate_w,
            gate_b: None,
        })
    }

    pub fn rand(dim: usize, hidden: usize, dtype: DType, dev: Device) -> Self {
        return Self {
            down_w: T::rand(vec![dim, hidden], dtype, dev),
            down_b: None,
            up_w: T::rand(vec![hidden, dim], dtype, dev),
            up_b: None,
            gate_w: T::rand(vec![hidden, dim], dtype, dev),
            gate_b: None,
        };
    }
}

pub trait Expert<T>: Sized
where
    T: EkTensor,
{
    fn backend(&self) -> std::string::String;
    fn shape(&self) -> ExpertShape;
    fn rand_input(&self, batch: usize) -> T;
    fn forward(&self, x: &T) -> T;
    fn construct(x: EKInstance, weight: ExpertWeight<T>) -> EKResult<Self>;
}

pub enum ExpertBackend {
    Torch(TorchFFN),
    Onnx(OnnxFFN),
}

impl ExpertBackend {
    pub async fn build<'a>(
        instance: x::EKInstance,
        tensor: &'a safetensors::SafeTensors<'a>,
    ) -> EKResult<ExpertBackend> {
        let backend = match instance.backend {
            x::ExpertBackendType::Torch => {
                let weight = ExpertWeight::<TchTensor>::from_safetensor(tensor)?;
                ExpertBackend::Torch(TorchFFN::construct(instance, weight)?)
            }
            x::ExpertBackendType::Onnx => todo!(),
        };
        Ok(backend)
    }
}

impl ExpertBackend {
    pub fn forward(&self, st_raw: &[u8]) -> EKResult<TchTensor> {
        let st = safetensors::SafeTensors::deserialize(st_raw).unwrap();
        let input_pos = st
            .names()
            .into_iter()
            .position(|x| x == "input")
            .ok_or(EKError::SafeTensorNotFound)?;
        let tensor_view = &st.tensors()[input_pos].1;

        match self {
            ExpertBackend::Torch(exp) => {
                let inp = TchTensor::from_tensor_view(tensor_view);
                Ok(exp.forward(&inp))
            }
            ExpertBackend::Onnx(exp) => {
                todo!()
            }
        }
    }
}
