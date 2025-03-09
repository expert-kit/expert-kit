use ndarray::ArrayD;
use std::time::Instant;
use tch::{
    Device, Tensor,
    nn::{self, Module, OptimizerConfig},
};

use crate::expert::Expert;

struct TorchFFN {
    dim: i64,
    hidden_dim: i64,
    module: Box<dyn Module>,
}
impl TorchFFN {
    pub fn new(dim: i64, hidden_dim: i64) -> Self {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let path = vs.root();

        let w1 = nn::linear(&path / "up", dim, hidden_dim, Default::default());
        let w2 = nn::linear(&path / "down", hidden_dim, dim, Default::default());
        let w3 = nn::linear(&path / "gate", dim, hidden_dim, Default::default());
        let module = nn::seq().add_fn(move |x| (x.apply(&w1).silu() * x.apply(&w3)).apply(&w2));
        return TorchFFN {
            dim: dim,
            hidden_dim: hidden_dim,
            module: Box::new(module),
        };
    }
}

impl Expert<Tensor> for TorchFFN {
    fn forward(&self, x: Tensor) -> Tensor {
        let res = self.module.forward(&x);
        res
    }
}

pub fn batch_scan() {
    let dim = 2048;
    let hidden = 7168;
    let expert_count = 64;
    let mut experts = Vec::new();

    for _ in 0..expert_count {
        let expert = TorchFFN::new(dim, hidden);
        experts.push(expert);
    }

    for batch in [1, 2, 4, 8, 16, 32, 64].iter() {
        let start = Instant::now();
        for expert in experts.iter() {
            let input = Tensor::randn([*batch as i64, dim], (tch::Kind::Float, Device::Cpu));
            let _ = expert.forward(input);
        }
        println!(
            "batch={} total={}ms layer={} layer_elapsed={} ms",
            batch,
            start.elapsed().as_millis(),
            expert_count,
            start.elapsed().as_millis() as f64 / expert_count as f64
        );
    }
}
