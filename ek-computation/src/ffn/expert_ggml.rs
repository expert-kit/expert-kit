use std::sync::Mutex;

use ek_ggml::{Context, Graph, Kind, SharedTensor, Tensor};

use crate::{
    backend::{DType, Device, EkTensor, ggml::GgmlTensor},
    ffn::meta::{Expert, ExpertShape, ExpertWeight},
};

const MAX_BATCH_SIZE_LOG2: usize = 8;
const MAX_BATCH_SIZE: usize = 2_usize.pow(MAX_BATCH_SIZE_LOG2 as u32);

/// GGML-based Feed Forward Network implementation with divide-and-conquer support for large batches.
///
/// This implementation supports batch sizes up to 2^MAX_BATCH_SIZE_LOG2 directly using pre-allocated
/// computation graphs. For larger batch sizes, it automatically uses a divide-and-conquer algorithm
/// to split the input into smaller chunks that can be processed individually and then concatenated.
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
        context_size += 3 * Tensor::overhead();
        log::debug!("Context size after initial overhead: {}", context_size);

        for i in 0..=MAX_BATCH_SIZE_LOG2 {
            let mut overhead = 0;
            overhead += 7 * Tensor::overhead(); // overhead for tensors
            overhead += Tensor::overhead() * Graph::<1>::default_size() + Graph::<1>::overhead(); // overhead for graph
            log::debug!(
                "Adding overhead for batch size 2^{}, overhead: {}",
                i,
                overhead
            );
            context_size += overhead;
        }
        let context = Context::new(context_size);

        log::debug!("Created context with size {}", context_size);
        let weights = [
            context
                .create_tensor(
                    &weight.up_w.shape,
                    weight.up_w.kind,
                    weight.up_w.data.into_boxed_slice(),
                )
                .unwrap(),
            context
                .create_tensor(
                    &weight.down_w.shape,
                    weight.down_w.kind,
                    weight.down_w.data.into_boxed_slice(),
                )
                .unwrap(),
            context
                .create_tensor(
                    &weight.gate_w.shape,
                    weight.gate_w.kind,
                    weight.gate_w.data.into_boxed_slice(),
                )
                .unwrap(),
        ];
        Self {
            dim,
            intermediate_dim,
            inner: Mutex::new(GgmlForwardInner {
                dim,
                context,
                weights,
                compute: Box::new([const { None }; MAX_BATCH_SIZE_LOG2 + 1]),
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
        Ok(Self::new(x.hidden, x.intermediate, weight, 1)) // TODO: Make n_threads configurable
    }
}

unsafe impl Send for GgmlFFN {}
unsafe impl Sync for GgmlFFN {}

struct GgmlForwardInner {
    dim: usize,
    context: Context,
    weights: [SharedTensor; 3],
    compute: Box<[Option<Graph<1>>; MAX_BATCH_SIZE_LOG2 + 1]>,
}

impl GgmlForwardInner {
    /// Forward pass with automatic large batch handling.
    ///
    /// For batch sizes <= 2^MAX_BATCH_SIZE_LOG2, uses pre-allocated computation graphs.
    /// For larger batch sizes, automatically switches to divide-and-conquer algorithm.
    fn forward(&mut self, shape: &[usize], x: &[u8], n_threads: usize) -> Vec<u8> {
        let batch_size = shape[0];
        let padded_batch_size = batch_size.next_power_of_two();
        let index = padded_batch_size.ilog2() as usize;

        // If batch size exceeds the maximum supported size, use divide-and-conquer algorithm
        if index > MAX_BATCH_SIZE_LOG2 {
            return self.forward_divide_and_conquer(shape, x, n_threads);
        }

        let graph = self.compute[index].get_or_insert_with(|| {
            self.context
                .create_graph(|allocator| {
                    let [w1, w2, w3] = &self.weights;

                    let [w1, w2, w3] = [
                        allocator.borrow(w1),
                        allocator.borrow(w2),
                        allocator.borrow(w3),
                    ];

                    let input =
                        allocator.alloc(&[padded_batch_size as _, self.dim as _], Kind::BF16); // [B, N] x bf16
                    let up = input.matmul(&w1); // [I, B] x f32
                    let gate = input.matmul(&w3); // [I, B] x f32
                    let hidden = up.mul_inplace(&gate.silu_inplace()?)?.cast(input.kind()); // [I, B] x bf16
                    let hidden = hidden.transpose(); // [B, I] x bf16
                    let output = w2.matmul(&hidden); // [B, N] x f32
                    let output = output.cast(input.kind()); // [B, N] x bf16

                    Ok(([input], output))
                })
                .unwrap()
        });

        if padded_batch_size > batch_size {
            let [input_kind] = graph.inputs_kind();
            let output_kind = graph.output_kind();
            let input = x
                .iter()
                .cloned()
                .chain(std::iter::repeat(0u8))
                .take(padded_batch_size * self.dim * input_kind.size())
                .collect::<Vec<_>>();
            graph
                .compute([&input], n_threads)
                .unwrap()
                .into_iter()
                .take(batch_size * self.dim * output_kind.size())
                .collect::<Vec<_>>()
        } else {
            graph.compute([x], n_threads).unwrap()
        }
    }

    /// Process large batch sizes using divide-and-conquer algorithm.
    ///
    /// This method splits large batches into multiple smaller batches that don't exceed
    /// 2^MAX_BATCH_SIZE_LOG2, processes them separately, and then concatenates the results.
    /// This allows efficient processing of arbitrarily large batches without failing due
    /// to memory or computation graph size constraints.
    ///
    /// # Algorithm Flow
    /// 1. Calculate maximum supported batch size (2^MAX_BATCH_SIZE_LOG2)
    /// 2. Split input data according to maximum batch size
    /// 3. Recursively call forward method on each sub-batch
    /// 4. Concatenate all sub-batch results into final result
    fn forward_divide_and_conquer(
        &mut self,
        shape: &[usize],
        x: &[u8],
        n_threads: usize,
    ) -> Vec<u8> {
        let batch_size = shape[0];

        // If batch size is less than or equal to maximum supported size, call original forward directly
        if batch_size <= MAX_BATCH_SIZE {
            return self.forward(shape, x, n_threads);
        }

        let input_size_per_sample = self.dim * Kind::BF16.size();

        let mut results = Vec::new();
        let mut processed = 0;

        while processed < batch_size {
            let current_batch_size = std::cmp::min(MAX_BATCH_SIZE, batch_size - processed);

            // Extract current batch data
            let start_idx = processed * input_size_per_sample;
            let end_idx = (processed + current_batch_size) * input_size_per_sample;
            let current_input = &x[start_idx..end_idx];

            // Recursively process current batch
            let current_shape = [current_batch_size, shape[1]];
            let current_output = self.forward(&current_shape, current_input, n_threads);

            // Collect results
            results.extend_from_slice(&current_output);

            processed += current_batch_size;
        }

        results
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use ek_ggml::Kind;
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
        assert_eq!(inp.kind, Kind::BF16);

        let inp_tch = TchTensor::from_tensor_view(&tv);

        let inp_data = inp.data.as_slice();
        let inp_tch_data = unsafe {
            std::slice::from_raw_parts(
                inp_tch.inner().data_ptr() as *const u8,
                inp_tch.inner().numel() * inp_tch.inner().kind().elt_size_in_bytes(),
            )
        };
        assert_eq!(inp_data.len(), inp_tch_data.len());
        assert_eq!(inp_data, inp_tch_data);

        let res = ffn.forward(&inp);

        assert_eq!(inp.data.len(), res.data.len());
        assert_eq!(inp.shape, res.shape);

        let truth = TchTensor::from_tensor_view(&gt_st.tensor("1-output").unwrap()).inner();

        let res_tch = TchTensor::from_raw(&res.data, &res.shape(), res.kind.into()).inner();

        let diff = (&res_tch - &truth).abs().sum(tch::Kind::BFloat16);
        diff.print();
    }

    #[test]
    fn test_ggml_large_batch_divide_and_conquer() {
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

        // Test batch size exceeding MAX_BATCH_SIZE_LOG2
        let large_batch_size = 2_usize.pow((super::MAX_BATCH_SIZE_LOG2 + 2) as u32); // 4 times larger than maximum supported size
        let large_input = ffn.rand_input(large_batch_size);

        // This call should trigger divide-and-conquer algorithm instead of panic
        let result = ffn.forward(&large_input);

        // Verify result shape and type
        assert_eq!(result.shape[0], large_batch_size as i64);
        assert_eq!(result.shape[1], 2048);
        assert_eq!(result.kind, Kind::BF16);

        // Verify result data length is correct
        let expected_data_len = large_batch_size * 2048 * Kind::BF16.size();
        assert_eq!(result.data.len(), expected_data_len);

        println!(
            "Successfully processed large batch of size {} using divide-and-conquer",
            large_batch_size
        );
    }
}
