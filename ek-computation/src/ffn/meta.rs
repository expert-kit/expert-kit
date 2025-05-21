use std::fmt::Display;

use ek_base::error::{EKError, EKResult};

use crate::{
    backend::{DType, Device, EkTensor, FromSafeTensor},
    x::EKInstance,
};

pub struct ExpertWeight<T>
where
    T: EkTensor,
{
    pub up_w: T,
    pub up_b: Option<T>,
    pub up_scale: Option<T>,
    pub down_w: T,
    pub down_b: Option<T>,
    pub down_scale: Option<T>,
    pub gate_w: T,
    pub gate_b: Option<T>,
    pub gate_scale: Option<T>,
}

pub struct ExpertShape {
    pub hidden: usize,
    pub intermediate: usize,
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
        let up_w = T::lookup_suffix(st, &["w1.weight", "up_proj.weight"])
            .ok_or(EKError::ExpertWeightNotFound("w1/up_w".to_owned()))?;
        let down_w = T::lookup_suffix(st, &["w2.weight", "down_proj.weight"])
            .ok_or(EKError::ExpertWeightNotFound("w2/down_w".to_owned()))?;
        let gate_w = T::lookup_suffix(st, &["w3.weight", "gate_proj.weight"])
            .ok_or(EKError::ExpertWeightNotFound("w3/gate_w".to_owned()))?;
        Ok(Self {
            up_w,
            up_b: None,
            up_scale: None,
            down_w,
            down_b: None,
            down_scale: None,
            gate_w,
            gate_b: None,
            gate_scale: None,
        })
    }

    pub fn from_rand_linear(hidden: usize, intermediate: usize, dtype: DType, dev: Device) -> Self {
        Self {
            down_w: T::rand(vec![hidden, intermediate], dtype, dev),
            down_b: None,
            up_scale: None,
            up_w: T::rand(vec![intermediate, hidden], dtype, dev),
            up_b: None,
            down_scale: None,
            gate_w: T::rand(vec![intermediate, hidden], dtype, dev),
            gate_b: None,
            gate_scale: None,
        }
    }

    pub fn from_rand_matmul(hidden: usize, intermediate: usize, dtype: DType, dev: Device) -> Self {
        Self {
            down_w: T::rand(vec![intermediate, hidden], dtype, dev),
            down_b: None,
            up_scale: None,
            up_w: T::rand(vec![hidden, intermediate], dtype, dev),
            up_b: None,
            down_scale: None,
            gate_w: T::rand(vec![hidden, intermediate], dtype, dev),
            gate_b: None,
            gate_scale: None,
        }
    }
}
