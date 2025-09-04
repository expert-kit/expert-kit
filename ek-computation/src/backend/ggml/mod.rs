mod ggml_safetensors;

use ek_ggml::Kind;

use crate::backend::{
    EkTensor, FromSafeTensor,
    ggml::ggml_safetensors::{dtype_to_ggml_kind, write_safetensors},
};

impl From<crate::backend::DType> for Kind {
    fn from(value: crate::backend::DType) -> Self {
        match value {
            crate::backend::DType::Float => Kind::F32,
            crate::backend::DType::BFloat16 => Kind::BF16,
            _ => unimplemented!(),
        }
    }
}

impl From<ek_ggml::Kind> for crate::backend::DType {
    fn from(value: Kind) -> Self {
        match value {
            Kind::F32 => crate::backend::DType::Float,
            Kind::BF16 => crate::backend::DType::BFloat16,
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgmlTensor {
    pub(crate) data: Vec<u8>,
    pub(crate) shape: Vec<i64>,
    pub(crate) kind: Kind,
}

impl GgmlTensor {
    pub fn empty(shape: &[usize], dtype: crate::backend::DType) -> Option<Self> {
        let shape_i64: Vec<i64> = shape.iter().map(|&s| s as i64).collect();
        let kind: Kind = dtype.into();
        Some(Self {
            data: Vec::new(),
            shape: shape_i64,
            kind,
        })
    }
}

impl FromSafeTensor for GgmlTensor {
    fn lookup_suffix(
        st: &safetensors::SafeTensors,
        name: &[&str],
        _dev: super::Device,
    ) -> Option<Self> {
        let idx = st
            .names()
            .iter()
            .position(|x| name.iter().any(|v| x.ends_with(v)));
        let tensors = st.tensors();
        if let Some(x) = idx {
            let (_, view) = tensors.get(x).unwrap();
            Some(Self::from_tensor_view(view))
        } else {
            None
        }
    }
}

impl EkTensor for GgmlTensor {
    fn rand(shape: Vec<usize>, dtype: crate::backend::DType, dev: crate::backend::Device) -> Self {
        assert_eq!(dev, crate::backend::Device::CPU);
        let mut data = vec![0u8; shape.iter().product::<usize>() * dtype.size()];
        for elem in data.iter_mut() {
            *elem = rand::random();
        }
        GgmlTensor {
            data,
            shape: shape.into_iter().map(|s| s as i64).collect(),
            kind: dtype.into(),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.iter().map(|&s| s as usize).collect()
    }

    fn serialize(&self) -> Vec<u8> {
        write_safetensors(&[("data", self)]).unwrap()
    }

    fn from_raw(data: &[u8], shape: &[usize], dtype: crate::backend::DType) -> Self {
        GgmlTensor {
            data: data.to_vec(),
            shape: shape.iter().map(|&s| s as i64).collect(),
            kind: dtype.into(),
        }
    }

    fn from_tensor_view(tv: &safetensors::tensor::TensorView<'_>) -> Self {
        let shape = tv.shape().iter().map(|&s| s as i64).collect::<Vec<_>>();
        let kind = dtype_to_ggml_kind(tv.dtype()).unwrap();
        GgmlTensor {
            data: tv.data().to_vec(),
            shape,
            kind,
        }
    }

    fn device(&self) -> crate::backend::Device {
        crate::backend::Device::CPU
    }

    fn to_device(&self, dev: crate::backend::Device) -> Self {
        assert_eq!(dev, crate::backend::Device::CPU);
        self.clone()
    }
}
