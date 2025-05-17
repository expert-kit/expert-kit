use std::{
    borrow::Borrow,
    sync::{Arc, Mutex},
};

use crate::{
    tch_safetensors::{dtype_to_tch_kind, tch_kind_to_dtype, write_safetensors},
    x,
};

use ek_base::error::EKResult;
use once_cell::sync::OnceCell;
use safetensors::tensor::TensorView;
use tch::{
    self, Tensor,
    nn::{self, Module, LinearConfig},
};

use super::{DType, Device, EkTensor, Expert, ExpertShape, ExpertWeight, FromSafeTensor};

pub struct TchTensor(Tensor);

impl From<tch::Tensor> for TchTensor {
    fn from(value: tch::Tensor) -> Self {
        TchTensor(value)
    }
}

impl Borrow<Tensor> for TchTensor {
    fn borrow(&self) -> &Tensor {
        &self.0
    }
}

impl From<DType> for tch::Kind {
    fn from(val: DType) -> Self {
        match val {
            DType::Uint8 => tch::Kind::Uint8,
            DType::Int16 => tch::Kind::Int16,
            DType::Int8 => tch::Kind::Int8,
            DType::BFloat16 => tch::Kind::BFloat16,
            DType::Float8e4m3fn => tch::Kind::Float8e4m3fn,
            DType::Float8e4m3fnuz => tch::Kind::Float8e4m3fnuz,
            DType::Float => tch::Kind::Float,
        }
    }
}

impl From<Device> for tch::Device {
    fn from(val: Device) -> Self {
        match val {
            Device::CPU => tch::Device::Cpu,
        }
    }
}

struct TchSafeView<'a> {
    tensor: &'a Tensor,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}

impl safetensors::View for TchSafeView<'_> {
    fn dtype(&self) -> safetensors::Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        let mut data = vec![0; self.data_len()];
        let numel = self.tensor.numel();
        self.tensor.f_copy_data_u8(&mut data, numel).unwrap();
        data.into()
    }

    fn data_len(&self) -> usize {
        self.tensor.numel() * self.tensor.kind().elt_size_in_bytes()
    }
}

impl<'a> From<&'a TchTensor> for TchSafeView<'a> {
    fn from(val: &'a TchTensor) -> Self {
        let dtype = tch_kind_to_dtype(val.0.kind()).unwrap();
        let shape = val.0.size().iter().map(|&x| x as usize).collect();
        Self {
            tensor: &val.0,
            shape,
            dtype,
        }
    }
}

impl EkTensor for TchTensor {
    fn rand(shape: Vec<usize>, typ: DType, dev: Device) -> Self {
        let rand = Tensor::randn(
            shape.into_iter().map(|x| x as i64).collect::<Vec<i64>>(),
            (typ.into(), dev.into()),
        );
        TchTensor(rand)
    }

    fn stack(tensors: &[Self], dim: usize) -> Self {
        TchTensor(tch::Tensor::stack(tensors, dim as i64))
    }

    fn shape(&self) -> Vec<usize> {
        self.0.size().iter().map(|&x| x as usize).collect()
    }

    fn serialize(&self) -> Vec<u8> {
        write_safetensors(&[("data", &self.0)]).unwrap()
    }

    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self {
        unsafe {
            Tensor::from_blob(
                data.as_ptr(),
                &shape.iter().map(|x| *x as i64).collect::<Vec<i64>>(),
                &[],
                dtype.into(),
                tch::Device::Cpu,
            )
            .into()
        }
    }

    fn from_tensor_view(tv: &TensorView<'_>) -> Self {
        tv.into()
    }
}

impl FromSafeTensor for TchTensor {}

impl From<&TensorView<'_>> for TchTensor {
    fn from(value: &TensorView<'_>) -> Self {
        let size: Vec<i64> = value.shape().iter().map(|&x| x as i64).collect();
        let kind: tch::Kind = dtype_to_tch_kind(value.dtype()).unwrap();
        let t = Tensor::f_from_data_size(value.data(), &size, kind).unwrap();
        TchTensor(t)
    }
}

impl TchTensor {
    pub fn inner(&self) -> Tensor {
        self.0.shallow_clone()
    }
}

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
    pub fn new(inst: x::EKInstance) -> Self {
        let weight = ExpertWeight::rand(inst.dim, inst.hidden, DType::Float, Device::CPU);
        Self::construct(inst, weight).unwrap()
    }

    pub fn load_module(&self) -> Arc<Mutex<nn::Sequential>> {
        let m = self.module.get_or_init(|| {
            tch::no_grad(|| {
                let dim = self.dim as i64;
                let hidden_dim = self.intermediate_dim as i64;
                let vs = nn::VarStore::new(tch::Device::Cpu);
                let path = vs.root();
                let mut w1 = nn::linear(&path / "up", dim, hidden_dim, LinearConfig{
                    ws_init: nn::Init::Const(0.0),
                    bs_init: None,
                    bias: false,
                });
                let mut w2 = nn::linear(&path / "down", hidden_dim, dim, LinearConfig{
                    ws_init: nn::Init::Const(0.0),
                    bs_init: None,
                    bias: false,
                });
                let mut w3 = nn::linear(&path / "gate", dim, hidden_dim, LinearConfig{
                    ws_init: nn::Init::Const(0.0),
                    bs_init: None,
                    bias: false,
                });
                w1.ws = self
                    .weight
                    .up_w
                    .0
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16);
                w2.ws = self
                    .weight
                    .down_w
                    .0
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16);
                w3.ws = self
                    .weight
                    .gate_w
                    .0
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16);
                w1.bs = None;
                w2.bs = None;
                w3.bs = None;
                let module =
                    nn::seq().add_fn(move |x| (x.apply(&w1) * x.apply(&w3).silu()).apply(&w2));
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

        log::debug!("input shape={:?} dtype={:?}", x.0.size(), x.0.kind());
        log::debug!("weight {}", self.weight);
        let res = guard.forward(&x.0);
        TchTensor(res)
    }

    fn rand_input(&self, batch: usize) -> TchTensor {
        TchTensor::rand(vec![batch, self.dim], DType::Float, Device::CPU)
    }
    fn shape(&self) -> ExpertShape {
        ExpertShape {
            dim: self.dim,
            hidden: self.intermediate_dim,
        }
    }

    fn backend(&self) -> std::string::String {
        "torch".to_string()
    }

    fn construct(x: crate::x::EKInstance, weight: ExpertWeight<TchTensor>) -> EKResult<Self> {
        let cell: OnceCell<Arc<Mutex<nn::Sequential>>> = OnceCell::new();
        let res = TorchFFN {
            intermediate_dim: x.hidden,
            dim: x.dim,
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
        ffn::{EkTensor, Expert, ExpertWeight, expert_torch::TorchFFN},
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
            dim: 2048,
            hidden: 768,
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
