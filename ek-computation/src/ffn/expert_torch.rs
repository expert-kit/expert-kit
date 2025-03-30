use std::borrow::Borrow;

use crate::tch_safetensors::{read_safetensors, write_safetensors};

use ek_base::error::{EKError, EKResult};
use tch::{
    self, Tensor,
    nn::{self, Module},
};

use super::{DType, Device, EkTensor, Expert, ExpertShape, ExpertWeight};

pub struct TchTensor(Tensor);

impl Borrow<Tensor> for TchTensor {
    fn borrow(&self) -> &Tensor {
        &self.0
    }
}

pub struct TorchFFN {
    dim: usize,
    hidden: usize,
    module: Box<dyn Module>,
}

impl Into<tch::Kind> for DType {
    fn into(self) -> tch::Kind {
        match self {
            DType::Uint8 => tch::Kind::Uint8,
            DType::Int16 => tch::Kind::Int16,
            DType::Int8 => tch::Kind::Int8,
            DType::BFloat16 => tch::Kind::BFloat16,
            DType::Float8e4m3fn => tch::Kind::Float8e4m3fn,
            DType::Float8e4m3fnuz => tch::Kind::Float8e4m3fnuz,
            DType::Float => tch::Kind::Float,
        }
    }
}

impl Into<tch::Device> for Device {
    fn into(self) -> tch::Device {
        match self {
            Device::CPU => tch::Device::Cpu,
            _ => unimplemented!(),
        }
    }
}

impl EkTensor for TchTensor {
    fn rand(shape: Vec<usize>, typ: DType, dev: Device) -> Self {
        let rand = Tensor::randn(
            shape.into_iter().map(|x| x as i64).collect::<Vec<i64>>(),
            (typ.into(), dev.into()),
        );
        return TchTensor(rand);
    }

    fn cat(tensors: &[Self], dim: usize) -> Self {
        TchTensor(tch::Tensor::cat(tensors, dim as i64))
    }

    fn serialize(&self) -> Vec<u8> {
        write_safetensors(&[("output", &self.0)])
            .map_err(|_e| EKError::SafeTensorError)
            .unwrap()
    }
}

impl TorchFFN {
    pub fn new(dim: usize, hidden: usize) -> Self {
        let weight = ExpertWeight::rand(dim, hidden, DType::Float, Device::CPU);
        Self::new_with_weight(dim, hidden, weight)
    }

    pub fn new_with_weight(dim: usize, hidden: usize, weight: ExpertWeight<TchTensor>) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let path = vs.root();
        let dim = dim as i64;
        let hidden_dim = hidden as i64;
        let mut w1 = nn::linear(&path / "up", dim, hidden_dim, Default::default());
        let mut w2 = nn::linear(&path / "down", hidden_dim, dim, Default::default());
        let mut w3 = nn::linear(&path / "gate", dim, hidden_dim, Default::default());
        w1.ws = weight.up_w.0;
        w1.bs = weight.up_b.map(|x| x.0);
        w2.ws = weight.down_w.0;
        w2.bs = weight.down_b.map(|x| x.0);
        w3.ws = weight.gate_w.0;
        w3.bs = weight.gate_b.map(|x| x.0);
        let module = nn::seq().add_fn(move |x| (x.apply(&w1).silu() * x.apply(&w3)).apply(&w2));
        return TorchFFN {
            hidden,
            dim: dim as usize,
            module: Box::new(module),
        };
    }
}

impl TryFrom<&[u8]> for TchTensor {
    type Error = ek_base::error::EKError;

    fn try_from(value: &[u8]) -> EKResult<Self> {
        let tensors = read_safetensors(value).map_err(|_e| EKError::SafeTensorError)?;
        assert!(tensors.len() == 1);
        let pos = tensors.iter().position(|x| x.0 == "input");
        if let Some(pos) = pos {
            // TODO: can we zero copy here?
            let tensor = tensors[pos].1.copy();
            return Ok(TchTensor(tensor));
        } else {
            return Err(ek_base::error::EKError::SafeTensorError);
        }
    }
}

impl TryFrom<Vec<u8>> for TchTensor {
    type Error = ek_base::error::EKError;
    fn try_from(value: Vec<u8>) -> EKResult<Self> {
        return Self::try_from(value.as_slice());
    }
}

impl Expert<TchTensor> for TorchFFN {
    fn forward(&self, x: &TchTensor) -> TchTensor {
        let res = self.module.forward(&x.0);
        TchTensor(res)
    }

    fn rand_input(&self, batch: usize) -> TchTensor {
        TchTensor::rand(vec![batch, self.dim], DType::Float, Device::CPU)
    }

    fn shape(&self) -> ExpertShape {
        ExpertShape {
            dim: self.dim,
            hidden: self.hidden,
        }
    }

    fn backend(&self) -> std::string::String {
        "torch".to_string()
    }
}
