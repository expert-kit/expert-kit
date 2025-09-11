use std::sync::OnceLock;

use criterion::{BatchSize, Criterion};
use ek_computation::{
    backend::{Device, EkTensor},
    ffn::{
        expert_ggml::GgmlFFN,
        expert_torch::TorchFFN,
        meta::{Expert, ExpertWeight},
    },
};

use crate::{BACKEND2DEVICES, DEVICES};

const BATCH_SIZES: &[usize] = &[1, 4, 8, 16, 32, 64, 128, 256, 512];

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("ffn w/o weight transfer");

    for &batch_size in BATCH_SIZES {
        for &backend in BACKEND2DEVICES.keys() {
            for &dev in &BACKEND2DEVICES[backend] {
                group.bench_with_input(
                    format!("batch={batch_size}, backend={backend} ({dev})"),
                    &batch_size,
                    |b, &batch_size| {
                        if backend == "ggml" {
                            let weight = ExpertWeight::from_rand_linear(
                                2048,
                                768,
                                ek_computation::backend::DType::BFloat16,
                                DEVICES[dev],
                            );
                            let ffn = GgmlFFN::new(2048, 768, weight, 8);
                            b.iter_batched(
                                || ffn.rand_input(batch_size).to_device(Device::CPU),
                                |input| {
                                    let _ = std::hint::black_box(
                                        ffn.forward(&input.to_device(DEVICES[dev]))
                                            .to_device(Device::CPU),
                                    );
                                },
                                BatchSize::PerIteration,
                            );
                        } else if backend == "torch" {
                            let weight = ExpertWeight::from_rand_linear(
                                2048,
                                768,
                                ek_computation::backend::DType::BFloat16,
                                DEVICES[dev],
                            );
                            let ffn =
                                TorchFFN::new(2048, 768, OnceLock::new(), weight, DEVICES[dev]);
                            b.iter_batched(
                                || ffn.rand_input(batch_size).to_device(Device::CPU),
                                |input| {
                                    let _ = std::hint::black_box(
                                        ffn.forward(&input.to_device(DEVICES[dev]))
                                            .to_device(Device::CPU),
                                    );
                                },
                                BatchSize::PerIteration,
                            );
                        }
                    },
                );
            }
        }
    }
}
