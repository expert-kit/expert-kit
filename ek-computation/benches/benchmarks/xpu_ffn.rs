use criterion::{BatchSize, Criterion};
use ek_computation::{
    backend::{Device, EkTensor},
    ffn::{
        expert_torch::TorchFFN,
        meta::{Expert, ExpertWeight},
    },
};
use once_cell::sync::OnceCell;

use crate::DEVICES;

const BATCH_SIZES: &[usize] = &[1, 4, 8, 16, 64];

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("torch ffn");

    for &batch_size in BATCH_SIZES {
        for &dev in DEVICES.keys() {
            group.bench_function(format!("batch={batch_size}, device={dev}"), |b| {
                let ffn = TorchFFN::new(
                    2048,
                    768,
                    OnceCell::new(),
                    ExpertWeight::from_rand_linear(
                        2048,
                        768,
                        ek_computation::backend::DType::BFloat16,
                        DEVICES[dev],
                    ),
                    DEVICES[dev],
                );
                b.iter_batched(
                    || ffn.rand_input(batch_size).to_device(Device::CPU),
                    |input| {
                        let _ = std::hint::black_box(
                            ffn.forward(&input.to_device(DEVICES[dev]))
                                .to_device(Device::CPU),
                        );
                    },
                    BatchSize::NumBatches(1),
                );
            });
        }
    }
}
