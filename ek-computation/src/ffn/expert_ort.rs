use ndarray::ArrayD;
use ndarray_rand::{RandomExt, rand_distr::Standard};
use ort::session::{Session, builder::GraphOptimizationLevel};

use super::{Expert, ExpertShape};

pub struct OnnxFFN {
    dim: usize,
    hidden: usize,
    sess: Session,
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

impl Expert<ArrayD<f32>> for OnnxFFN {
    fn rand_input(&self, batch: usize) -> ArrayD<f32> {
        let shape = vec![batch, self.dim];
        let input = ArrayD::random(shape, Standard);
        input
    }

    fn forward(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        let outputs = self
            .sess
            .run(ort::inputs!["input"=>x.clone()].unwrap())
            .unwrap();
        let vals = outputs
            .get("output")
            .unwrap()
            .try_extract_tensor::<f32>()
            .unwrap();
        vals.to_owned()
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
}
