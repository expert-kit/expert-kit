use std::fmt::Debug;

use ek_base::error::EKResult;
use ndarray::{Array, ArrayD, IxDyn};

use ndarray_rand::rand_distr::num_traits::{self};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    tensor::PrimitiveTensorElementType,
    value::{Tensor, Value},
};
use safetensors::tensor::TensorView;

use crate::onnx::exporter::ExpertOnnxBuilder;

use super::{DType, EkTensor, Expert, ExpertShape, ExpertWeight, FromSafeTensor};

pub struct OnnxFFN<D: OrtDType> {
    dim: i64,
    hidden: i64,
    sess: Session,
    _phantom: std::marker::PhantomData<D>,
}
pub trait OrtDType:
    PrimitiveTensorElementType + num_traits::Num + Clone + Debug + Copy + 'static
{
}
impl OrtDType for f32 {}
impl OrtDType for half::bf16 {}

#[derive(Clone, Debug)]
pub struct NDArrayTensor<D: OrtDType>(ArrayD<D>);

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

impl<D: OrtDType> OnnxFFN<D> {
    pub fn new(
        hidden_dim: i64,
        intermediate_dim: i64,
        dt: DType,
        weight: ExpertWeight<NDArrayTensor<D>>,
    ) -> EKResult<Self> {
        let builder = ExpertOnnxBuilder {
            intermediate_size: intermediate_dim,
            hidden_size: hidden_dim,
            data_type: dt,
        };
        let raw = builder.build_raw();
        let model = Session::builder()?
            .with_external_initializer("onnx::MatMul_13", weight.gate_w.into())
            .expect("should load up")
            .with_external_initializer("onnx::MatMul_14", weight.up_w.into())
            .expect("should load down")
            .with_external_initializer("onnx::MatMul_15", weight.down_w.into())
            .expect("should loadgate")
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            // .with_execution_providers([OneDNNExecutionProvider::default().build()])
            // .unwrap()
            .with_intra_threads(std::thread::available_parallelism().unwrap().into())?
            .with_parallel_execution(true)?
            .commit_from_memory(raw.as_slice())?;

        Ok(OnnxFFN {
            sess: model,
            dim: hidden_dim,
            hidden: intermediate_dim,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<D> From<NDArrayTensor<D>> for Value
where
    D: OrtDType,
{
    fn from(val: NDArrayTensor<D>) -> Self {
        let v = Tensor::from_array(val.0.view()).unwrap().into_dyn();
        v
    }
}

impl<T> Expert<NDArrayTensor<T>> for OnnxFFN<T>
where
    T: OrtDType + Clone,
{
    fn rand_input(&self, batch: usize) -> NDArrayTensor<T> {
        let shape = vec![batch, self.dim as usize];
        let input = ArrayD::zeros(shape);
        input.into()
    }

    fn forward(&self, x: &NDArrayTensor<T>) -> NDArrayTensor<T> {
        let v = Tensor::from_array(x.clone().0.view()).unwrap().into_dyn();
        let outputs = self
            .sess
            .run(
                ort::inputs![
                "input"=>v,
                ]
                .unwrap(),
            )
            .unwrap();

        let vals = outputs
            .get("output")
            .unwrap()
            .try_extract_tensor::<T>()
            .unwrap();
        vals.to_owned().into()
    }

    fn shape(&self) -> super::ExpertShape {
        ExpertShape {
            dim: self.dim as usize,
            hidden: self.hidden as usize,
        }
    }

    fn backend(&self) -> std::string::String {
        "onnxruntime".to_string()
    }

    fn construct(
        _x: crate::x::EKInstance,
        _weight: super::ExpertWeight<NDArrayTensor<T>>,
    ) -> ek_base::error::EKResult<Self> {
        unimplemented!()
    }
}
