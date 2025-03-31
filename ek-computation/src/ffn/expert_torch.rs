use std::borrow::Borrow;

use crate::{
    tch_safetensors::{read_safetensors, tch_kind_to_dtype, write_safetensors},
    x,
};

use ek_base::error::{EKError, EKResult};
use safetensors::{Dtype, View, tensor::TensorView};
use tch::{
    self, Tensor,
    nn::{self, Module},
};

use super::{DType, Device, EkTensor, Expert, ExpertShape, ExpertWeight, FromSafeTensor};

pub struct TchTensor(Tensor);

impl From<tch::Tensor> for TchTensor {
    fn from(value: tch::Tensor) -> Self {
        return TchTensor(value);
    }
}

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

struct TchSafeView<'a> {
    tensor: &'a Tensor,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}

impl<'a> safetensors::View for TchSafeView<'a> {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        let mut data = vec![0; self.data_len()];
        let numel = self.tensor.numel();
        self.tensor.f_copy_data_u8(&mut data, numel).unwrap();
        data.into()
    }

    fn data_len(&self) -> usize {
        self.tensor.numel() * self.tensor.kind().elt_size_in_bytes()
    }
}

impl<'a> From<&'a TchTensor> for TchSafeView<'a> {
    fn from(val: &'a TchTensor) -> Self {
        let dtype = tch_kind_to_dtype(val.0.kind()).unwrap();
        let shape = val.0.size().iter().map(|&x| x as usize).collect();
        return Self {
            tensor: &val.0,
            shape,
            dtype,
        };
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
        write_safetensors(&[("output", &self.0)]).unwrap()
    }

    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self {
        Tensor::f_from_data_size(
            data,
            &shape.iter().map(|x| *x as i64).collect::<Vec<i64>>(),
            dtype.into(),
        )
        .unwrap() // TODO: is it safe to unwrap?
        .into()
    }

    fn from_tensor_view(tv: &TensorView<'_>) -> Self {
        todo!()
    }
}

impl FromSafeTensor for TchTensor {}

impl From<&TensorView<'_>> for TchTensor {
    fn from(value: &TensorView<'_>) -> Self {
        todo!()
    }
}

impl TorchFFN {
    pub fn new(inst: x::EKInstance) -> Self {
        let weight = ExpertWeight::rand(inst.dim, inst.hidden, DType::Float, Device::CPU);
        Self::construct(inst, weight).unwrap()
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

    fn construct(x: crate::x::EKInstance, weight: ExpertWeight<TchTensor>) -> EKResult<Self> {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let path = vs.root();
        let dim = x.dim as i64;
        let hidden_dim = x.hidden as i64;
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
        Ok(TorchFFN {
            hidden: x.hidden,
            dim: dim as usize,
            module: Box::new(module),
        })
    }
}
