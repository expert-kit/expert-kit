use std::fmt::Debug;

use ndarray::{Array, ArrayD, IxDyn};

use ndarray_rand::{
    RandomExt,
    rand_distr::{
        Distribution, Standard, Uniform,
        num_traits::{One, Zero},
        uniform::SampleUniform,
    },
};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    tensor::PrimitiveTensorElementType,
    value::Tensor,
};
use safetensors::tensor::TensorView;

use super::{EkTensor, Expert, ExpertShape};

pub struct OnnxFFN {
    dim: usize,
    hidden: usize,
    sess: Session,
}
trait OrtDType:
    PrimitiveTensorElementType + Clone + Zero + One + SampleUniform + Debug + Copy + 'static
{
}

#[derive(Clone, Debug)]
struct NDArrayTensor<D: OrtDType>(ArrayD<D>);

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

impl<D> EkTensor for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn rand(shape: Vec<usize>, _dtype: super::DType, _dev: super::Device) -> Self {
        let res = ArrayD::random(shape, Uniform::new(D::zero(), D::one()));
        Self(res)
    }

    fn cat(tensors: &[Self], dim: usize) -> Self {
        let views = tensors.iter().map(|x| x.0.view()).collect::<Vec<_>>();
        let res = ndarray::concatenate(ndarray::Axis(dim), &views)
            .unwrap()
            .into_dimensionality::<IxDyn>()
            .unwrap();
        NDArrayTensor(res)
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

impl OnnxFFN {
    fn new_sess(model_path: std::path::PathBuf) -> ort::Result<Session> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(std::thread::available_parallelism().unwrap().into())?
            .with_parallel_execution(true)?
            .commit_from_file(model_path)?;
        Ok(model)
    }

    pub fn new(model_path: std::path::PathBuf, dim: usize, hidden: usize) -> Self {
        let sess = OnnxFFN::new_sess(model_path).unwrap();
        OnnxFFN { sess, dim, hidden }
    }
}

impl<T> Expert<NDArrayTensor<T>> for OnnxFFN
where
    T: OrtDType + Clone,
    Standard: Distribution<T>,
{
    fn rand_input(&self, batch: usize) -> NDArrayTensor<T> {
        let shape = vec![batch, self.dim];
        let input = ArrayD::random(shape, Standard);
        input.into()
    }

    fn forward(&self, x: &NDArrayTensor<T>) -> NDArrayTensor<T> {
        let v = Tensor::from_array(x.clone().0.view()).unwrap().into_dyn();
        let outputs = self.sess.run(ort::inputs!["input"=>v].unwrap()).unwrap();

        let vals = outputs
            .get("output")
            .unwrap()
            .try_extract_tensor::<T>()
            .unwrap();
        vals.to_owned().into()
    }

    fn shape(&self) -> super::ExpertShape {
        ExpertShape {
            dim: self.dim,
            hidden: self.hidden,
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
