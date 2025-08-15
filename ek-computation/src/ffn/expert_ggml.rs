use std::{collections::HashMap, sync::Mutex};

use ek_ggml::{Context, Graph, Kind, Tensor};

use crate::{
    backend::{DType, Device, EkTensor, ggml::GgmlTensor},
    ffn::meta::{Expert, ExpertShape, ExpertWeight},
};

pub struct GgmlFFN {
    dim: usize,
    intermediate_dim: usize,
    inner: Mutex<GgmlForwardInner>,
    n_threads: usize,
}

impl GgmlFFN {
    pub fn new(
        dim: usize,
        intermediate_dim: usize,
        weight: ExpertWeight<GgmlTensor>,
        n_threads: usize,
    ) -> Self {
        let mut context_size = 0;
        context_size += weight.up_w.shape.iter().sum::<i64>() as usize * weight.up_w.kind.size();
        context_size +=
            weight.down_w.shape.iter().sum::<i64>() as usize * weight.down_w.kind.size();
        context_size +=
            weight.gate_w.shape.iter().sum::<i64>() as usize * weight.gate_w.kind.size();
        context_size += 3 * Tensor::overhead();

        for i in 0..=9 {
            let batch_size = 2_usize.pow(i);
            context_size += batch_size * dim * Kind::BF16.size(); // input 
            context_size += batch_size * intermediate_dim * Kind::F32.size(); // up
            context_size += batch_size * intermediate_dim * Kind::F32.size(); // gate
            context_size += batch_size * intermediate_dim * Kind::BF16.size(); // hidden
            context_size += batch_size * intermediate_dim * Kind::BF16.size(); // hidden.T
            context_size += batch_size * dim * Kind::F32.size(); // output
            context_size += batch_size * dim * Kind::BF16.size(); // output.cast
            context_size += 7 * Tensor::overhead(); // overhead for tensors
            context_size += Graph::overhead(); // overhead for graph
            context_size += 1024; // additional overhead
        }

        let mut context = Context::new(context_size.next_multiple_of(4096));
        let mut weights = [
            context.create_tensor(&weight.up_w.shape, weight.up_w.kind),
            context.create_tensor(&weight.down_w.shape, weight.down_w.kind),
            context.create_tensor(&weight.gate_w.shape, weight.gate_w.kind),
        ];
        weights[0].set_data(&weight.up_w.data).unwrap();
        weights[1].set_data(&weight.down_w.data).unwrap();
        weights[2].set_data(&weight.gate_w.data).unwrap();
        Self {
            dim,
            intermediate_dim,
            inner: Mutex::new(GgmlForwardInner {
                dim,
                context,
                weights,
                compute: HashMap::default(),
            }),
            n_threads,
        }
    }
}

impl Expert<GgmlTensor> for GgmlFFN {
    fn forward(&self, x: &GgmlTensor) -> GgmlTensor {
        let mut inner = self.inner.lock().unwrap();
        GgmlTensor {
            data: inner.forward(&x.shape(), &x.data, self.n_threads),
            shape: x.shape.clone(),
            kind: x.kind,
        }
    }

    fn rand_input(&self, batch: usize) -> GgmlTensor {
        GgmlTensor::rand(vec![batch, self.dim], DType::BFloat16, Device::CPU)
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
        Ok(Self::new(x.hidden, x.intermediate, weight, 8)) // TODO: Make n_threads configurable
    }
}

struct GgmlForwardInner {
    dim: usize,
    context: Context,
    weights: [Tensor; 3],
    compute: HashMap<usize, (Tensor, Graph, Tensor)>,
}

impl GgmlForwardInner {
    fn forward(&mut self, shape: &[usize], x: &[u8], n_threads: usize) -> Vec<u8> {
        let batch_size = shape[0];
        let padded_batch_size = batch_size.next_power_of_two();
        let (input, graph, output) = self.compute.entry(padded_batch_size).or_insert_with(|| {
            let [w1, w2, w3] = &self.weights;
            let input = self
                .context
                .create_tensor(&[padded_batch_size as _, self.dim as _], Kind::BF16); // [B, N] x bf16
            let up = input.matmul(w1); // [I, B] x f32
            let gate = input.matmul(w3); // [I, B] x f32
            let hidden = up.mul_inplace(&gate.silu_inplace()).cast(input.kind()); // [I, B] x bf16
            let hidden = hidden.transpose(); // [B, I] x bf16
            let output = w2.matmul(&hidden); // [B, N] x f32
            let output = output.cast(input.kind()); // [B, N] x bf16
            let mut graph = self.context.create_graph();
            graph.build_forward(&output);
            (input, graph, output)
        });
        if padded_batch_size > batch_size {
            input
                .set_data(
                    &x.iter()
                        .cloned()
                        .chain(std::iter::repeat(0u8))
                        .take(padded_batch_size * self.dim * input.kind().size())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        } else {
            input.set_data(x).unwrap();
        }
        graph.compute(n_threads);
        if padded_batch_size > batch_size {
            output
                .get_data()
                .iter()
                .cloned()
                .take(batch_size * self.dim * output.kind().size())
                .collect()
        } else {
            output.get_data().to_vec()
        }
    }
}

#[cfg(test)]
mod test {
    use std::fs;

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

        let inp_tch = TchTensor::from_tensor_view(&tv);

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

        let truth = TchTensor::from_tensor_view(&gt_st.tensor("1-output").unwrap()).inner();

        let res_tch = TchTensor::from_raw(&res.data, &res.shape(), res.kind.into()).inner();

        let diff = (&res_tch - &truth).sum(tch::Kind::BFloat16);
        diff.print();
    }
}
