mod benchmarks;

use std::{collections::HashMap, sync::LazyLock};

use criterion::{criterion_group, criterion_main};
use ek_computation::backend::Device;

pub static BACKEND2DEVICES: LazyLock<HashMap<&'static str, Vec<&'static str>>> =
    LazyLock::new(|| {
        let mut backends = HashMap::new();
        backends.insert("ggml", vec!["cpu"]);
        backends.insert("torch", vec!["cpu", "cuda:0"]);
        backends
    });

pub static DEVICES: LazyLock<HashMap<&'static str, Device>> = LazyLock::new(|| {
    let mut devices = HashMap::new();
    devices.insert("cpu", Device::CPU);
    devices.insert("cuda:0", Device::CUDA(0));
    devices
});

criterion_group!(
    benches,
    benchmarks::xpu_ffn_activate::bench,
    benchmarks::xpu_ffn_queue::bench,
    benchmarks::xpu_ffn_with_weight_transfer::bench,
    benchmarks::xpu_ffn::bench,
    benchmarks::xpu_transfer::bench,
);
criterion_main!(benches);
