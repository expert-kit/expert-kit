//! Safetensors support for tensors.
//!
//! This module implements reading and writing tensors in the `.safetensors` format.
//! <https://github.com/huggingface/safetensors>
use ek_base::error::EKError;
use tch::{Kind, TchError, Tensor};

use std::convert::{TryFrom, TryInto};

use safetensors::tensor::{Dtype, SafeTensors, View};

pub fn tch_kind_to_dtype(kind: Kind) -> Result<Dtype, TchError> {
    let dtype = match kind {
        Kind::Bool => Dtype::BOOL,
        Kind::Uint8 => Dtype::U8,
        Kind::Int8 => Dtype::I8,
        Kind::Int16 => Dtype::I16,
        Kind::Int => Dtype::I32,
        Kind::Int64 => Dtype::I64,
        Kind::BFloat16 => Dtype::BF16,
        Kind::Half => Dtype::F16,
        Kind::Float => Dtype::F32,
        Kind::Double => Dtype::F64,
        kind => return Err(TchError::Convert(format!("unsupported kind ({kind:?})"))),
    };
    Ok(dtype)
}

pub fn dtype_to_tch_kind(dtype: Dtype) -> Result<Kind, TchError> {
    let kind = match dtype {
        Dtype::BOOL => Kind::Bool,
        Dtype::U8 => Kind::Uint8,
        Dtype::I8 => Kind::Int8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 => Kind::Int64,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::F16 => Kind::Half,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        dtype => return Err(TchError::Convert(format!("unsupported dtype {dtype:?}"))),
    };
    Ok(kind)
}

pub fn read_safetensors(data: &[u8]) -> Result<Vec<(String, Tensor)>, EKError> {
    let safetensors = SafeTensors::deserialize(data).map_err(|_e| EKError::SafeTensorError)?;
    safetensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let size: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
            let kind: Kind = dtype_to_tch_kind(view.dtype()).unwrap();
            let tensor_w: Result<Tensor, _> = Tensor::f_from_data_size(view.data(), &size, kind);
            if let Ok(tensor) = tensor_w {
                Ok((name, tensor))
            } else {
                Err(EKError::SafeTensorError)
            }
        })
        .collect()
}
struct SafeView<'a> {
    tensor: &'a Tensor,
    shape: Vec<usize>,
    dtype: Dtype,
}

impl<'a> TryFrom<&'a Tensor> for SafeView<'a> {
    type Error = TchError;

    fn try_from(tensor: &'a Tensor) -> Result<Self, Self::Error> {
        if tensor.is_sparse() {
            return Err(TchError::Convert("Cannot save sparse tensors".to_string()));
        }

        if !tensor.is_contiguous() {
            return Err(TchError::Convert(
                "Cannot save non contiguous tensors".to_string(),
            ));
        }

        let dtype = tch_kind_to_dtype(tensor.kind())?;
        let shape = tensor.size().iter().map(|&x| x as usize).collect();
        Ok(Self {
            tensor,
            shape,
            dtype,
        })
    }
}

impl View for SafeView<'_> {
    fn dtype(&self) -> Dtype {
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

pub fn write_safetensors<S: AsRef<str>, T: AsRef<Tensor>>(
    tensors: &[(S, T)],
) -> Result<Vec<u8>, EKError> {
    let views = tensors
        .iter()
        .map(|(name, tensor)| {
            Ok::<(&str, SafeView), TchError>((name.as_ref(), tensor.as_ref().try_into()?))
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_e| EKError::SafeTensorError)?;

    safetensors::tensor::serialize(views, &None).map_err(|_e| EKError::SafeTensorError)
}
