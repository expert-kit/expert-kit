use std::clone;

use ek_base::error::EKError;
use ndarray::ArrayD;
use ndarray_rand::{
    RandomExt,
    rand_distr::{Distribution, Standard},
};
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    tensor::PrimitiveTensorElementType,
    value::Value,
};
use safetensors::tensor::TensorView;

use super::{EkTensor, Expert, ExpertShape};

pub struct OnnxFFN {
    dim: usize,
    hidden: usize,
    sess: Session,
}
trait OrtDType: PrimitiveTensorElementType {}

#[derive(Clone, Debug)]
pub struct NDArrayTensor<D: OrtDType>(ArrayD<D>);

impl<D> From<TensorView<'_>> for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn from(value: TensorView<'_>) -> Self {
        todo!()
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
    fn rand(shape: Vec<usize>, dtype: super::DType, dev: super::Device) -> Self {
        todo!()
    }

    fn cat(tensors: &[Self], dim: usize) -> Self {
        todo!()
    }

    fn serialize(&self) -> Vec<u8> {
        todo!()
    }

    fn from_raw(data: &[u8], shape: &[usize], dtype: super::DType) -> Self {
        todo!()
    }

    fn from_tensor_view(tv: &TensorView<'_>) -> Self {
        todo!()
    }
}

impl<D> Into<Value> for NDArrayTensor<D>
where
    D: OrtDType,
{
    fn into(self) -> Value {
        todo!()
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
        let outputs = self
            .sess
            .run(ort::inputs!["input"=>x.clone()].unwrap())
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
            dim: self.dim,
            hidden: self.hidden,
        }
    }

    fn backend(&self) -> std::string::String {
        "onnxruntime".to_string()
    }

    fn construct(
        x: crate::x::EKInstance,
        weight: super::ExpertWeight<NDArrayTensor<T>>,
    ) -> ek_base::error::EKResult<Self> {
        todo!()
    }
}
