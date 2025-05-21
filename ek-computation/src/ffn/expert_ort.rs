use std::marker::PhantomData;

use ek_base::error::EKResult;
use ndarray::ArrayD;

use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

use crate::{
    backend::{
        DType,
        ort::{NDArrayTensor, OrtDType},
    },
    onnx::exporter::ExpertOnnxBuilder,
};

use super::{
    ExpertWeight,
    meta::{Expert, ExpertShape},
};

pub struct OnnxFFN<T: OrtDType> {
    dim: i64,
    hidden: i64,
    sess: Session,
    phantom: PhantomData<T>,
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
            .with_intra_threads(std::thread::available_parallelism().unwrap().into())?
            .with_parallel_execution(true)?
            .commit_from_memory(raw.as_slice())?;

        Ok(OnnxFFN {
            sess: model,
            dim: hidden_dim,
            hidden: intermediate_dim,
            phantom: PhantomData,
        })
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

    fn shape(&self) -> ExpertShape {
        ExpertShape {
            hidden: self.dim as usize,
            intermediate: self.hidden as usize,
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
