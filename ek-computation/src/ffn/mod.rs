use std::fmt::Display;

use ek_base::error::{EKError, EKResult};
use expert_ort::OnnxFFN;
use expert_torch::{TchTensor, TorchFFN};
use safetensors::tensor::TensorView;

use crate::{
    tch_safetensors::dtype_to_tch_kind,
    x::{self, EKInstance},
};

#[allow(dead_code)]
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
    fn lookup_suffix(st: &safetensors::SafeTensors, name: &[&str]) -> Option<Self> {
        let idx = st
            .names()
            .iter()
            .position(|x| name.iter().any(|v| x.ends_with(v)));
        let tensors = st.tensors();
        if let Some(x) = idx {
            let (_name, view) = tensors.get(x).unwrap();
            let size: Vec<usize> = view.shape().to_vec();
            let kind: DType = DType::from(dtype_to_tch_kind(view.dtype()).unwrap());
            Some(Self::from_raw(view.data(), &size, kind))
        } else {
            None
        }
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

impl<T> Display for ExpertWeight<T>
where
    T: EkTensor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ExpertWeight(up_w: {:?}, down_w: {:?}, gate_w: {:?})",
            self.up_w.shape(),
            self.down_w.shape(),
            self.gate_w.shape()
        )
    }
}

impl<T: EkTensor + FromSafeTensor> ExpertWeight<T> {
    pub fn from_safetensor(st: &safetensors::SafeTensors) -> EKResult<Self> {
        let up_w = T::lookup_suffix(st, &["w1.weight","up_proj.weight"])
            .ok_or(EKError::ExpertWeightNotFound("w1/up_w".to_owned()))?;
        let down_w = T::lookup_suffix(st, &["w2.weight","down_proj.weight"])
            .ok_or(EKError::ExpertWeightNotFound("w2/down_w".to_owned()))?;
        let gate_w = T::lookup_suffix(st, &["w3.weight","gate_proj.weight"])
            .ok_or(EKError::ExpertWeightNotFound("w3/gate_w".to_owned()))?;
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
        Self {
            down_w: T::rand(vec![dim, hidden], dtype, dev),
            down_b: None,
            up_w: T::rand(vec![hidden, dim], dtype, dev),
            up_b: None,
            gate_w: T::rand(vec![hidden, dim], dtype, dev),
            gate_b: None,
        }
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
    pub fn forward(&self, view: &TensorView) -> EKResult<TchTensor> {
        match self {
            ExpertBackend::Torch(exp) => {
                let inp = TchTensor::from_tensor_view(view);
                let shape = inp.inner().size();
                log::debug!("input shape {:?}", shape);
                assert!(shape.len() == 2);
                Ok(exp.forward(&inp))
            }
            ExpertBackend::Onnx(_exp) => {
                todo!()
            }
        }
    }
}
