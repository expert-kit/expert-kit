use criterion::Criterion;
use ek_computation::backend::{DType, EkTensor, torch::TchTensor};

use crate::DEVICES;

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("xpu transfer");

    for &src_dev in DEVICES.keys() {
        for &dst_dev in DEVICES.keys() {
            if src_dev == dst_dev {
                continue; // Skip same device transfers
            }
            group.bench_function(format!("from {src_dev} to {dst_dev}"), |b| {
                let src_device = DEVICES[src_dev];
                let dst_device = DEVICES[dst_dev];
                let tensor = TchTensor::rand(vec![2048, 768], DType::BFloat16, src_device);
                b.iter(|| {
                    let _ = std::hint::black_box(tensor.to_device(dst_device));
                });
            });
        }
    }
}
