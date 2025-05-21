use std::sync::{Arc, Mutex};

use crate::backend::{DType, Device, EkTensor, torch::TchTensor};

use ek_base::error::EKResult;
use once_cell::sync::OnceCell;
use tch::{
    self,
    nn::{self, Module},
};

use super::{
    ExpertWeight,
    meta::{Expert, ExpertShape},
};

pub struct TorchFFN {
    dim: usize,
    intermediate_dim: usize,
    module: OnceCell<Arc<Mutex<nn::Sequential>>>,
    weight: ExpertWeight<TchTensor>,
}

pub fn w8a16_activate(x: &tch::Tensor, s: &tch::Tensor, block_size: i64) -> tch::Tensor {
    let shape = s.size();
    let x_shape = x.size();
    assert!(shape.len() == 2);
    assert!(x_shape.len() == 2);
    let m = shape[0];
    let n = shape[1];
    let pad = x_shape[0] % block_size;
    let s = s.reshape([shape[0], shape[1], 1]);
    let l = if pad > 0 {
        let t = tch::Tensor::zeros([pad, x_shape[1]], (x.kind(), x.device()));

        (tch::Tensor::cat(&[x, &t], 0))
            .reshape([m, block_size, n, block_size])
            .permute([0, 2, 1, 3])
            .reshape([m, n, block_size * block_size])
            .to_kind(tch::Kind::Float)
    } else {
        x.reshape([m, block_size, n, block_size])
            .permute([0, 2, 1, 3])
            .reshape([m, n, block_size * block_size])
            .to_kind(tch::Kind::Float)
    };

    (l * s)
        .to_kind(tch::Kind::BFloat16)
        .reshape([m, n, block_size, block_size])
        .permute([0, 2, 1, 3])
        .reshape(x_shape.clone())
}

unsafe impl Sync for TorchFFN {}

impl TorchFFN {
    pub fn load_module(&self) -> Arc<Mutex<nn::Sequential>> {
        let m = self.module.get_or_init(|| {
            tch::no_grad(|| {
                let w1_tensor = self
                    .weight
                    .up_w
                    .inner()
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16);
                let w2_tensor = self
                    .weight
                    .down_w
                    .inner()
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16);
                let w3_tensor = self
                    .weight
                    .gate_w
                    .inner()
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16);

                let module = nn::seq().add_fn(move |x| {
                    let _up = x.matmul(&w1_tensor.transpose(0, 1));
                    let _gate = x.matmul(&w3_tensor.transpose(0, 1));
                    let _hidden = _up * _gate.silu();
                    _hidden.matmul(&w2_tensor.transpose(0, 1))
                });

                Arc::new(Mutex::new(module))
            })
        });
        m.clone()
    }
}

impl Expert<TchTensor> for TorchFFN {
    fn forward(&self, x: &TchTensor) -> TchTensor {
        let module = self.load_module();
        let guard = module.lock().unwrap();

        let res = guard.forward(&x.inner());
        TchTensor(res)
    }

    fn rand_input(&self, batch: usize) -> TchTensor {
        TchTensor::rand(vec![batch, self.dim], DType::BFloat16, Device::CPU)
    }
    fn shape(&self) -> ExpertShape {
        ExpertShape {
            hidden: self.dim,
            intermediate: self.intermediate_dim,
        }
    }

    fn backend(&self) -> std::string::String {
        "torch".to_string()
    }

    fn construct(x: crate::x::EKInstance, weight: ExpertWeight<TchTensor>) -> EKResult<Self> {
        let cell: OnceCell<Arc<Mutex<nn::Sequential>>> = OnceCell::new();
        let res = TorchFFN {
            intermediate_dim: x.intermediate,
            dim: x.hidden,
            module: cell,
            weight,
        };
        // res.load_module();

        Ok(res)
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use ek_base::utils::workspace_root;
    use safetensors::SafeTensors;
    use tch::IndexOp;
    use test::Bencher;
    extern crate test;

    use crate::{
        backend::EkTensor,
        ffn::{Expert, ExpertWeight, expert_torch::TorchFFN},
        x::{self, test_root},
    };

    use super::{TchTensor, w8a16_activate};

    #[test]
    fn test_io() {
        let rand_t = tch::Tensor::randn(vec![128, 128], (tch::Kind::Float, tch::Device::Cpu));
        let target = TchTensor::from(rand_t.copy());
        let bytes = target.serialize();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let tv = st.tensor("data").unwrap();
        let processed = TchTensor::from_tensor_view(&tv);
        assert!(processed.inner().sum(tch::Kind::Float) == rand_t.sum(tch::Kind::Float))
    }

    #[test]
    fn test_correctness() {
        let st_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.weight.safetensors");
        let st_bytes = fs::read(st_fp).unwrap();
        let st = SafeTensors::deserialize(&st_bytes).unwrap();
        let weight = ExpertWeight::from_safetensor(&st).unwrap();
        let inst = x::EKInstance {
            hidden: 2048,
            intermediate: 768,
            backend: x::ExpertBackendType::Torch,
        };
        let ffn = TorchFFN::construct(inst, weight).unwrap();

        let ground_truth_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.result.safetensors");
        let ground_truth_bytes = fs::read(ground_truth_fp).unwrap();
        let gt_st = SafeTensors::deserialize(&ground_truth_bytes).unwrap();

        let tv = gt_st.tensor("1-input").unwrap();
        let inp = TchTensor::from_tensor_view(&tv);

        let res = ffn.forward(&inp).inner();
        let truth = TchTensor::from_tensor_view(&gt_st.tensor("1-output").unwrap()).inner();

        let _vec1 = Vec::<f32>::try_from(res.i((0, 0..100))).unwrap();
        let _vec2 = Vec::<f32>::try_from(truth.i((0, 0..100))).unwrap();
        (res - truth).sum(tch::Kind::BFloat16).print();
    }

    #[test]
    fn test_fp8_dequant() {
        let st_fp = workspace_root()
            .join("ek-computation")
            .join("resources")
            .join("w8a16active-l0q_a_proj.safetensors");
        let st_bytes = fs::read(st_fp).unwrap();
        let st = SafeTensors::deserialize(&st_bytes).unwrap();
        let tv1 = st.tensor("src").unwrap();
        let tv2 = st.tensor("src_scale").unwrap();
        let expected = st.tensor("triton_dequanted").unwrap();
        let tv1 = TchTensor::from_tensor_view(&tv1).inner();
        let tv2 = TchTensor::from_tensor_view(&tv2).inner();
        let expected = TchTensor::from_tensor_view(&expected).inner();
        let res = w8a16_activate(&tv1, &tv2, 128);
        let diff = (res - expected)
            .sum(tch::Kind::Double)
            .abs()
            .double_value(&[]);
        dbg!(diff);
        assert!(diff < 0.2);
    }

    #[bench]
    fn pressure_test(b: &mut Bencher) {
        let tv1 = tch::Tensor::randn(vec![7168, 2048], (tch::Kind::Float, tch::Device::Cpu));
        let tv2 = tch::Tensor::randn(vec![56, 16], (tch::Kind::Float, tch::Device::Cpu));
        let _ = w8a16_activate(&tv1, &tv2, 128);
        let _ = w8a16_activate(&tv1, &tv2, 128);
        let _ = w8a16_activate(&tv1, &tv2, 128);
        b.iter(|| {
            let _ = w8a16_activate(&tv1, &tv2, 128);
        });
    }
}
