use std::{sync::OnceLock, time::Duration};

use criterion::{BatchSize, Criterion};
use ek_computation::{
    backend::{DType, Device, EkTensor, torch::TchTensor},
    ffn::{
        expert_torch::TorchFFN,
        meta::{Expert, ExpertWeight},
    },
};

use crate::DEVICES;

const BATCH_SIZES: &[usize] = &[1, 4, 8, 16, 64, 128, 256, 512];

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("torch ffn w/ weight transfer");
    group.measurement_time(Duration::from_secs(60));

    for &batch_size in BATCH_SIZES {
        for &dev in DEVICES.keys() {
            group.bench_function(format!("batch={batch_size}, device={dev}"), |b| {
                b.iter_batched(
                    || {
                        (
                            ExpertWeight::from_rand_linear(
                                2048,
                                768,
                                ek_computation::backend::DType::BFloat16,
                                Device::CPU,
                            ),
                            TchTensor::rand(vec![batch_size, 2048], DType::BFloat16, Device::CPU),
                        )
                    },
                    |(weight, input)| {
                        let ffn = TorchFFN::new(2048, 768, OnceLock::new(), weight, DEVICES[dev]);
                        let _ = std::hint::black_box(
                            ffn.forward(&input.to_device(DEVICES[dev]))
                                .to_device(Device::CPU)
                                .inner()
                                .numel(),
                        );
                    },
                    BatchSize::PerIteration,
                );
            });
        }
    }
}
