use ek_computation::ffn::expert_torch::w8a16_activate;

use criterion::Criterion;

use crate::DEVICES;

pub fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("activate w8a16");

    for &dev in DEVICES.keys() {
        group.bench_function(format!("device={dev}"), |b| {
            let tv1 = tch::Tensor::randn(vec![7168, 2048], (tch::Kind::Float, DEVICES[dev].into()));
            let tv2 = tch::Tensor::randn(vec![56, 16], (tch::Kind::Float, DEVICES[dev].into()));
            b.iter(|| {
                let _ = std::hint::black_box(w8a16_activate(&tv1, &tv2, 128));
            });
        });
    }
}
