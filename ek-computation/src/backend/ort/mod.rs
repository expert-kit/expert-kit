use std::fmt::Debug;

use ndarray::{Array, ArrayD, IxDyn};
use ort::{tensor::PrimitiveTensorElementType, value::Value};
use rand_distr::num_traits;
use safetensors::tensor::TensorView;

use super::{EkTensor, FromSafeTensor};

pub trait OrtDType:
    PrimitiveTensorElementType + num_traits::Num + Clone + Debug + Copy + 'static
{
}
impl OrtDType for f32 {}
impl OrtDType for half::bf16 {}

#[derive(Clone, Debug)]
pub struct NDArrayTensor<D: OrtDType>(pub ArrayD<D>);

impl<D> From<TensorView<'_>> for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn from(view: TensorView<'_>) -> Self {
        let raw = view.data();
        unsafe {
            let (_, d_slice, _) = raw.align_to::<D>();
            let copied = d_slice.to_vec();

            let v = Array::from_vec(copied)
                .into_dimensionality::<IxDyn>()
                .unwrap();

            NDArrayTensor(v)
        }
    }
}

impl<D> From<ArrayD<D>> for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn from(value: ArrayD<D>) -> Self {
        NDArrayTensor(value)
    }
}

impl<D> FromSafeTensor for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn lookup_suffix(_st: &safetensors::SafeTensors, _name: &[&str]) -> Option<Self> {
        todo!()
    }
}

impl<D> EkTensor for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn rand(shape: Vec<usize>, _dtype: super::DType, _dev: super::Device) -> Self {
        let res = ArrayD::zeros(shape);
        Self(res)
    }

    fn stack(tensors: &[Self], dim: usize) -> Self {
        let views = tensors.iter().map(|x| x.0.view()).collect::<Vec<_>>();
        let res = ndarray::stack(ndarray::Axis(dim), &views)
            .unwrap()
            .into_dimensionality::<IxDyn>()
            .unwrap();
        NDArrayTensor(res)
    }

    fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    fn serialize(&self) -> Vec<u8> {
        todo!()
    }

    fn from_raw(data: &[u8], shape: &[usize], _dtype: super::DType) -> Self {
        let raw = data;
        unsafe {
            let (_, d_slice, _) = raw.align_to::<D>();
            let copied = d_slice.to_vec();

            let v = Array::from_vec(copied)
                .into_dimensionality::<IxDyn>()
                .unwrap()
                .into_shape_with_order(shape)
                .unwrap();
            NDArrayTensor(v)
        }
    }

    fn from_tensor_view(tv: &TensorView<'_>) -> Self {
        let raw = tv.data();
        Self::from_raw(raw, tv.shape(), tv.dtype().into())
    }
}

impl<D> From<NDArrayTensor<D>> for Value
where
    D: OrtDType,
{
    fn from(val: NDArrayTensor<D>) -> Self {
        let v = ort::value::Tensor::from_array(val.0.view())
            .unwrap()
            .into_dyn();
        v
    }
}
