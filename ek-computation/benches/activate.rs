use ek_computation::ffn::expert_torch::w8a16_activate;

use criterion::{criterion_group, criterion_main, Criterion};

pub fn stress(c: &mut Criterion) {
    let tv1 = tch::Tensor::randn(vec![7168, 2048], (tch::Kind::Float, tch::Device::Cpu));
    let tv2 = tch::Tensor::randn(vec![56, 16], (tch::Kind::Float, tch::Device::Cpu));
    
    c.bench_function("stress", |b| b.iter(|| {
        let _ = std::hint::black_box(w8a16_activate(&tv1, &tv2, 128));
    }));
}

criterion_group!(benches, stress);
criterion_main!(benches);
