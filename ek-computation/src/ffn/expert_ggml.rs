use std::{collections::HashMap, sync::Mutex};

use ek_ggml::{Graph, Kind, Tensor};

use crate::{
    backend::{DType, Device, EkTensor, ggml::GgmlTensor},
    ffn::meta::{Expert, ExpertShape, ExpertWeight},
};

pub struct GgmlFFN {
    dim: usize,
    intermediate_dim: usize,
    weight: ExpertWeight<GgmlTensor>,
    io: Mutex<HashMap<u32, (Tensor, Graph, Tensor)>>,
    n_threads: usize,
}

impl GgmlFFN {
    pub fn new(
        dim: usize,
        intermediate_dim: usize,
        weight: ExpertWeight<GgmlTensor>,
        n_threads: usize,
    ) -> Self {
        Self {
            dim,
            intermediate_dim,
            weight,
            io: Mutex::new(HashMap::new()),
            n_threads,
        }
    }
}

impl Expert<GgmlTensor> for GgmlFFN {
    fn forward(&self, x: &GgmlTensor) -> GgmlTensor {
        let shape = x.shape(); // [B, N]
        let batch_size = shape[0];
        let padded_batch_size = batch_size.next_power_of_two();
        let log2 = padded_batch_size.ilog2();

        let mut io = self.io.lock().unwrap();
        let (input, graph, output) = io.entry(log2).or_insert_with(|| {
            let w1 = unsafe {
                Tensor::from_raw(
                    &self.weight.up_w.data,
                    &self.weight.up_w.shape,
                    self.weight.up_w.kind,
                )
                .unwrap()
            }; // [I, N]

            let w2 = unsafe {
                Tensor::from_raw(
                    &self.weight.down_w.data,
                    &self.weight.down_w.shape,
                    self.weight.down_w.kind,
                )
                .unwrap()
            }; // [N, I]

            let w3 = unsafe {
                Tensor::from_raw(
                    &self.weight.gate_w.data,
                    &self.weight.gate_w.shape,
                    self.weight.gate_w.kind,
                )
                .unwrap()
            }; // [I, N]

            let input =
                Tensor::empty(&vec![padded_batch_size as _, self.dim as _], Kind::F32).unwrap(); // [B, N]

            let up = input.matmul(&w1); // [B, I]
            let gate = input.matmul(&w3); // [B, I]

            let hidden = up.mul(&gate.silu()); // [B, I]
            let output = hidden.transpose().matmul(&w2).transpose(); // [B, N]

            let mut graph = Graph::new();
            graph.build_forward(&output);

            (input, graph, output)
        });

        if padded_batch_size > batch_size {
            input
                .set_data(
                    &x.data
                        .iter()
                        .cloned()
                        .chain(std::iter::repeat(0u8))
                        .take(padded_batch_size * self.dim * input.kind().size())
                        .collect::<Vec<_>>()
                        .as_ref(),
                )
                .unwrap();
            graph.compute(self.n_threads);

            GgmlTensor {
                data: output
                    .raw_data()
                    .iter()
                    .cloned()
                    .take(batch_size * self.dim * output.kind().size())
                    .collect(),
                kind: output.kind(),
                shape: x.shape.clone(),
            }
        } else {
            input.set_data(&x.data).unwrap();
            graph.compute(self.n_threads);

            GgmlTensor {
                data: output.raw_data().to_vec(),
                kind: output.kind(),
                shape: output.shape(),
            }
        }
    }

    fn rand_input(&self, batch: usize) -> GgmlTensor {
        GgmlTensor::rand(vec![batch, self.dim], DType::Float, Device::CPU)
    }

    fn shape(&self) -> super::meta::ExpertShape {
        ExpertShape {
            hidden: self.dim,
            intermediate: self.intermediate_dim,
        }
    }

    fn backend(&self) -> std::string::String {
        "ggml".to_string()
    }

    fn construct(
        x: crate::x::EKInstance,
        weight: ExpertWeight<GgmlTensor>,
    ) -> ek_base::error::EKResult<Self> {
        Ok(Self::new(x.hidden, x.intermediate, weight, 16)) // TODO: Make n_threads configurable
    }
}

unsafe impl Sync for GgmlFFN {}

#[cfg(test)]
mod test {
    use std::fs;

    use ek_ggml::Context;
    use safetensors::SafeTensors;

    use crate::{
        backend::{Device, EkTensor, ggml::GgmlTensor, torch::TchTensor},
        ffn::{
            expert_ggml::GgmlFFN,
            meta::{Expert, ExpertWeight},
        },
        x::{self, test_root},
    };

    #[test]
    fn test_ggml_correctness() {
        Context::init(1024 * 1024 * 1024);

        let st_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.weight.safetensors");
        let st_bytes = fs::read(st_fp).unwrap();
        let st = SafeTensors::deserialize(&st_bytes).unwrap();
        let weight = ExpertWeight::from_safetensor(&st, Device::CPU).unwrap();
        let inst = x::EKInstance {
            hidden: 2048,
            intermediate: 768,
            backend: x::ExpertBackendType::Ggml,
            device: Device::CPU,
        };
        let ffn = GgmlFFN::construct(inst, weight).unwrap();

        let ground_truth_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.result.safetensors");
        let ground_truth_bytes = fs::read(ground_truth_fp).unwrap();
        let gt_st = SafeTensors::deserialize(&ground_truth_bytes).unwrap();

        let tv = gt_st.tensor("1-input").unwrap();

        let inp = GgmlTensor::from_tensor_view(&tv);

        let inp_tch = TchTensor(
            TchTensor::from_tensor_view(&tv)
                .inner()
                .to_kind(tch::Kind::Float),
        );

        let inp_data = inp.data.as_slice();
        let inp_tch_data = unsafe {
            std::slice::from_raw_parts(
                inp_tch.inner().data_ptr() as *const u8,
                inp_tch.inner().numel() * inp_tch.inner().kind().elt_size_in_bytes(),
            )
        };
        assert_eq!(inp_data.len(), inp_tch_data.len());
        assert_eq!(inp_data[..20], inp_tch_data[..20]);

        let res = ffn.forward(&inp);

        let truth = TchTensor::from_tensor_view(&gt_st.tensor("1-output").unwrap())
            .inner()
            .to_kind(tch::Kind::Float);

        let res_tch = TchTensor::from_raw(&res.data, &res.shape(), res.kind.into()).inner();

        let diff = (&res_tch - &truth).sum(tch::Kind::Float);
        diff.print();
    }
}
