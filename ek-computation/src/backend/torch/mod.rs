mod tch_safetensors;
use std::borrow::Borrow;

use super::{DType, Device, EkTensor, FromSafeTensor};
use safetensors::tensor::TensorView;
use tch;
use tch_safetensors::{dtype_to_tch_kind, tch_kind_to_dtype, write_safetensors};

pub struct TchTensor(pub tch::Tensor);

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

impl From<DType> for tch::Kind {
    fn from(val: DType) -> Self {
        match val {
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

impl From<Device> for tch::Device {
    fn from(val: Device) -> Self {
        match val {
            Device::CPU => tch::Device::Cpu,
        }
    }
}

struct TchSafeView<'a> {
    tensor: &'a tch::Tensor,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}

impl safetensors::View for TchSafeView<'_> {
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
        Self {
            tensor: &val.0,
            shape,
            dtype,
        }
    }
}

impl From<tch::Tensor> for TchTensor {
    fn from(value: tch::Tensor) -> Self {
        TchTensor(value)
    }
}

impl Borrow<tch::Tensor> for TchTensor {
    fn borrow(&self) -> &tch::Tensor {
        &self.0
    }
}

impl EkTensor for TchTensor {
    fn rand(shape: Vec<usize>, typ: DType, dev: Device) -> Self {
        let rand = tch::Tensor::randn(
            shape.into_iter().map(|x| x as i64).collect::<Vec<i64>>(),
            (typ.into(), dev.into()),
        );
        TchTensor(rand)
    }

    fn stack(tensors: &[Self], dim: usize) -> Self {
        TchTensor(tch::Tensor::stack(tensors, dim as i64))
    }

    fn shape(&self) -> Vec<usize> {
        self.0.size().iter().map(|&x| x as usize).collect()
    }

    fn serialize(&self) -> Vec<u8> {
        write_safetensors(&[("data", &self.0)]).unwrap()
    }

    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self {
        unsafe {
            tch::Tensor::from_blob(
                data.as_ptr(),
                &shape.iter().map(|x| *x as i64).collect::<Vec<i64>>(),
                &[],
                dtype.into(),
                tch::Device::Cpu,
            )
            .into()
        }
    }

    fn from_tensor_view(tv: &TensorView<'_>) -> Self {
        tv.into()
    }
}

impl FromSafeTensor for TchTensor {
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

impl From<&TensorView<'_>> for TchTensor {
    fn from(value: &TensorView<'_>) -> Self {
        let size: Vec<i64> = value.shape().iter().map(|&x| x as i64).collect();
        let kind: tch::Kind = dtype_to_tch_kind(value.dtype()).unwrap();
        let t = tch::Tensor::f_from_data_size(value.data(), &size, kind).unwrap();
        TchTensor(t)
    }
}

impl TchTensor {
    pub fn inner(&self) -> tch::Tensor {
        self.0.shallow_clone()
    }
}
