use criterion::{BatchSize, Criterion};
use ek_computation::{
    backend::EkTensor,
    ffn::{
        expert_torch::TorchFFN,
        meta::{Expert, ExpertWeight},
    },
};
use once_cell::sync::OnceCell;

use crate::DEVICES;

const BATCH_SIZES: &[usize] = &[1, 4, 16, 64];
const FFN_COUNT: usize = 256;

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("torch queued ffn");

    for &batch_size in BATCH_SIZES {
        for &dev in DEVICES.keys() {
            let ffns = (0..FFN_COUNT)
                .map(|_| {
                    TorchFFN::new(
                        2048,
                        768,
                        OnceCell::new(),
                        ExpertWeight::from_rand_linear(
                            2048,
                            768,
                            ek_computation::backend::DType::BFloat16,
                            // we use cuda gpu to accelerate the ffn initialization here
                            ek_computation::backend::Device::CUDA(0),
                        ),
                        DEVICES[dev],
                    )
                })
                .collect::<Vec<_>>();
            let input = ffns[0].rand_input(batch_size).to_device(DEVICES[dev]);

            let setup = || {
                // Randomly select one of the FFNs to benchmark
                let idx = (rand::random::<u64>() % FFN_COUNT as u64) as usize;
                &ffns[idx]
            };

            group.bench_function(format!("batch={batch_size}, device={dev}"), |b| {
                b.iter_batched(
                    setup,
                    |ffn| {
                        let _ = std::hint::black_box(ffn.forward(&input));
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }
}
