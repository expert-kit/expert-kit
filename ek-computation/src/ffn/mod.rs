use std::{ops::Index, str::pattern::Pattern};

use tch::Tensor;

use crate::tch_safetensors::dtype_to_tch_kind;

pub mod expert_ort;
pub mod expert_torch;

pub struct ExpertShape {
    pub dim: usize,
    pub hidden: usize,
}
pub trait Expert<T> {
    fn backend(&self) -> std::string::String;
    fn shape(&self) -> ExpertShape;
    fn rand_input(&self, batch: usize) -> T;
    fn forward(&self, x: &T) -> T;
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
}

pub trait FromSafeTensor
where
    Self: Sized + EkTensor,
{
    fn lookup(st: safetensors::SafeTensors, name: &str) -> Option<Self> {
        let idx = st.names().iter().position(|x| x.contains(name));
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
impl<T: EkTensor> ExpertWeight<T> {
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
