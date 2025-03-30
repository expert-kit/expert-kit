use ort::execution_providers::TensorRTExecutionProvider;
use tonic::Status;

use crate::tch_safetensors::read_safetensors;
use ek_base::error::{EKError, EKResult};
use tch::{
    Device, Tensor,
    nn::{self, Module},
};

use super::{Expert, ExpertShape};

pub struct TchTensor(Tensor);

pub struct TorchFFN {
    dim: usize,
    hidden: usize,
    module: Box<dyn Module>,
}
impl TorchFFN {
    pub fn new(dim: usize, hidden: usize) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let path = vs.root();
        let dim = dim as i64;
        let hidden_dim = hidden as i64;
        let w1 = nn::linear(&path / "up", dim, hidden_dim, Default::default());
        let w2 = nn::linear(&path / "down", hidden_dim, dim, Default::default());
        let w3 = nn::linear(&path / "gate", dim, hidden_dim, Default::default());
        let module = nn::seq().add_fn(move |x| (x.apply(&w1).silu() * x.apply(&w3)).apply(&w2));
        return TorchFFN {
            hidden,
            dim: dim as usize,
            module: Box::new(module),
        };
    }
}

impl TryFrom<Vec<u8>> for TchTensor {
    type Error = ek_base::error::EKError;

    fn try_from(value: Vec<u8>) -> EKResult<Self> {
        let tensors = read_safetensors(value.as_slice()).map_err(|_e| EKError::SafeTensorError)?;
        assert!(tensors.len() == 1);
        let pos = tensors.iter().position(|x| x.0 == "input");
        if let Some(pos) = pos {
            // TODO: can we zero copy here?
            let tensor = tensors[pos].1.copy();
            return Ok(TchTensor(tensor));
        } else {
            return Err(ek_base::error::EKError::SafeTensorError);
        }
    }
}

impl Expert<TchTensor> for TorchFFN {
    fn forward(&self, x: TchTensor) -> TchTensor {
        let res = self.module.forward(&x.0);
        TchTensor(res)
    }

    fn rand_input(&self, batch: usize) -> TchTensor {
        let input = Tensor::randn(
            [batch as i64, self.dim as i64],
            (tch::Kind::Float, Device::Cpu),
        );
        TchTensor(input)
    }

    fn shape(&self) -> ExpertShape {
        ExpertShape {
            dim: self.dim,
            hidden: self.hidden,
        }
    }

    fn backend(&self) -> std::string::String {
        "torch".to_string()
    }
}
