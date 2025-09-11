use ek_ggml::Kind;
use safetensors::{Dtype, View};

use crate::backend::ggml::GgmlTensor;

pub fn ggml_kind_to_dtype(kind: Kind) -> Result<Dtype, Kind> {
    match kind {
        Kind::F32 => Ok(Dtype::F32),
        Kind::BF16 => Ok(Dtype::BF16),
        _ => Err(kind),
    }
}

pub fn dtype_to_ggml_kind(dtype: Dtype) -> Result<Kind, Dtype> {
    match dtype {
        Dtype::F32 => Ok(Kind::F32),
        Dtype::BF16 => Ok(Kind::BF16),
        _ => Err(dtype),
    }
}

struct SafeView<'a> {
    tensor_data: &'a [u8],
    shape: Vec<usize>,
    dtype: Dtype,
}

impl<'a> SafeView<'a> {
    fn new(tensor: &'a GgmlTensor) -> Self {
        Self {
            tensor_data: tensor.data.as_ref(),
            shape: tensor.shape.iter().map(|&s| s as usize).collect(),
            dtype: ggml_kind_to_dtype(tensor.kind).unwrap(),
        }
    }
}

impl View for SafeView<'_> {
    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        std::borrow::Cow::Borrowed(self.tensor_data)
    }

    fn data_len(&self) -> usize {
        self.tensor_data.len()
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn dtype(&self) -> Dtype {
        self.dtype
    }
}

pub fn write_safetensors<S>(
    tensors: &[(S, &GgmlTensor)],
) -> Result<Vec<u8>, Box<dyn std::error::Error>>
where
    S: AsRef<str>,
{
    let views = tensors
        .iter()
        .map(|(name, tensor)| {
            let view = SafeView::new(tensor);
            Ok::<(&str, SafeView), Box<dyn std::error::Error>>((name.as_ref(), view))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = safetensors::tensor::serialize(views, &None)?;
    Ok(result)
}
