use std::sync::{Arc, Mutex};

use crate::backend::{DType, Device, EkTensor, torch::TchTensor};

use super::{
    ExpertWeight,
    meta::{Expert, ExpertShape},
};
use ek_base::error::EKResult;
use once_cell::sync::OnceCell;
use tch::{
    self,
    nn::{self, Module},
};

pub struct TorchFFN {
    dim: usize,
    intermediate_dim: usize,
    module: OnceCell<Arc<Mutex<nn::Sequential>>>,
    weight: ExpertWeight<TchTensor>,
    device: Device,
}

pub fn w8a16_activate(x: &tch::Tensor, s: &tch::Tensor, block_size: i64) -> tch::Tensor {
    let shape = s.size();
    let x_shape = x.size();
    assert!(shape.len() == 2);
    assert!(x_shape.len() == 2);
    let m = shape[0];
    let n = shape[1];
    let pad = x_shape[0] % block_size;
    let s = s.reshape([shape[0], shape[1], 1]);
    let l = if pad > 0 {
        let t = tch::Tensor::zeros([pad, x_shape[1]], (x.kind(), x.device()));

        (tch::Tensor::cat(&[x, &t], 0))
            .reshape([m, block_size, n, block_size])
            .permute([0, 2, 1, 3])
            .reshape([m, n, block_size * block_size])
            .to_kind(tch::Kind::Float)
    } else {
        x.reshape([m, block_size, n, block_size])
            .permute([0, 2, 1, 3])
            .reshape([m, n, block_size * block_size])
            .to_kind(tch::Kind::Float)
    };

    (l * s)
        .to_kind(tch::Kind::BFloat16)
        .reshape([m, n, block_size, block_size])
        .permute([0, 2, 1, 3])
        .reshape(x_shape.clone())
}

unsafe impl Sync for TorchFFN {}

impl TorchFFN {
    pub fn load_module(&self) -> Arc<Mutex<nn::Sequential>> {
        let m = self.module.get_or_init(|| {
            tch::no_grad(|| {
                let w1_tensor = self
                    .weight
                    .up_w
                    .inner()
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16)
                    .to_device(self.device.into());
                let w2_tensor = self
                    .weight
                    .down_w
                    .inner()
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16)
                    .to_device(self.device.into());
                let w3_tensor = self
                    .weight
                    .gate_w
                    .inner()
                    .shallow_clone()
                    .to_kind(tch::Kind::BFloat16)
                    .to_device(self.device.into());
                let module = nn::seq().add_fn(move |x| {
                    let _up = x.matmul(&w1_tensor.transpose(0, 1));
                    let _gate = x.matmul(&w3_tensor.transpose(0, 1));
                    let _hidden = _up * _gate.silu();
                    _hidden.matmul(&w2_tensor.transpose(0, 1))
                });

                Arc::new(Mutex::new(module))
            })
        });
        m.clone()
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Expert<TchTensor> for TorchFFN {
    fn forward(&self, x: &TchTensor) -> TchTensor {
        let module = self.load_module();
        let guard = module.lock().unwrap();

        let res = guard.forward(&x.inner());
        TchTensor(res)
    }

    fn rand_input(&self, batch: usize) -> TchTensor {
        TchTensor::rand(vec![batch, self.dim], DType::BFloat16, Device::CPU)
    }
    fn shape(&self) -> ExpertShape {
        ExpertShape {
            hidden: self.dim,
            intermediate: self.intermediate_dim,
        }
    }

    fn backend(&self) -> std::string::String {
        "torch".to_string()
    }

    fn construct(x: crate::x::EKInstance, weight: ExpertWeight<TchTensor>) -> EKResult<Self> {
        let cell: OnceCell<Arc<Mutex<nn::Sequential>>> = OnceCell::new();
        let res = TorchFFN {
            intermediate_dim: x.intermediate,
            dim: x.hidden,
            module: cell,
            weight,
            device: x.device,
        };
        // res.load_module();

        Ok(res)
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use ek_base::utils::workspace_root;
    use safetensors::SafeTensors;
    use tch::IndexOp;

    use crate::{
        backend::{Device, EkTensor},
        ffn::{Expert, ExpertWeight, expert_torch::TorchFFN},
        x::{self, test_root},
    };

    use super::{TchTensor, w8a16_activate};

    #[test]
    fn test_io() {
        let rand_t = tch::Tensor::randn(vec![128, 128], (tch::Kind::Float, tch::Device::Cpu));
        let target = TchTensor::from(rand_t.copy());
        let bytes = target.serialize();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let tv = st.tensor("data").unwrap();
        let processed = TchTensor::from_tensor_view(&tv);
        assert!(processed.inner().sum(tch::Kind::Float) == rand_t.sum(tch::Kind::Float))
    }

    #[test]
    fn test_correctness() {
        let st_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.weight.safetensors");
        let st_bytes = fs::read(st_fp).unwrap();
        let st = SafeTensors::deserialize(&st_bytes).unwrap();
        let weight = ExpertWeight::from_safetensor(&st, Device::CPU).unwrap();
        let inst = x::EKInstance {
            hidden: 2048,
            intermediate: 768,
            backend: x::ExpertBackendType::Torch,
            device: Device::CPU,
        };
        let ffn = TorchFFN::construct(inst, weight).unwrap();

        let ground_truth_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.result.safetensors");
        let ground_truth_bytes = fs::read(ground_truth_fp).unwrap();
        let gt_st = SafeTensors::deserialize(&ground_truth_bytes).unwrap();

        let tv = gt_st.tensor("1-input").unwrap();
        let inp = TchTensor::from_tensor_view(&tv);

        let res = ffn.forward(&inp).inner();
        let truth = TchTensor::from_tensor_view(&gt_st.tensor("1-output").unwrap()).inner();

        let _vec1 = Vec::<f32>::try_from(res.i((0, 0..100))).unwrap();
        let _vec2 = Vec::<f32>::try_from(truth.i((0, 0..100))).unwrap();
        (res - truth).sum(tch::Kind::BFloat16).print();
    }

    #[test]
    fn test_fp8_dequant() {
        let st_fp = workspace_root()
            .join("ek-computation")
            .join("resources")
            .join("w8a16active-l0q_a_proj.safetensors");
        let st_bytes = fs::read(st_fp).unwrap();
        let st = SafeTensors::deserialize(&st_bytes).unwrap();
        let tv1 = st.tensor("src").unwrap();
        let tv2 = st.tensor("src_scale").unwrap();
        let expected = st.tensor("triton_dequanted").unwrap();
        let tv1 = TchTensor::from_tensor_view(&tv1).inner();
        let tv2 = TchTensor::from_tensor_view(&tv2).inner();
        let expected = TchTensor::from_tensor_view(&expected).inner();
        let res = w8a16_activate(&tv1, &tv2, 128);
        let diff = (res - expected)
            .sum(tch::Kind::Double)
            .abs()
            .double_value(&[]);
        assert!(diff < 0.2);
    }
}

#[cfg(test)]
mod bench_ffn {
    use super::TchTensor;
    use crate::{
        backend::{DType, Device, EkTensor},
        ffn::{Expert, ExpertWeight, expert_torch::TorchFFN},
    };
    use once_cell::sync::OnceCell;

    #[test]
    fn bench_transfer() {
        // Configs
        let round = 128;
        let cuda_device = Device::CUDA(0);
        let cpu_device = Device::CPU;

        let tensor = TchTensor::rand(vec![2048, 768], DType::BFloat16, cpu_device);

        // warm
        tensor.to_device(cuda_device);
        tensor.to_device(cpu_device);

        let mut to_cuda_durations = vec![std::time::Duration::new(0, 0)];
        let mut to_cpu_durations = vec![std::time::Duration::new(0, 0)];

        println!("Starting transfer benchmark...");

        for _ in 0..round {
            let now = std::time::Instant::now();
            let n = tensor.to_device(cuda_device);
            std::hint::black_box(n);
            to_cuda_durations.push(now.elapsed());

            let now = std::time::Instant::now();
            let m = tensor.to_device(cpu_device);
            std::hint::black_box(m);
            to_cpu_durations.push(now.elapsed());
        }

        // function to calculate mean and variance statistics
        fn calculate_stats(durations: &[std::time::Duration]) -> (f64, f64) {
            let times: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect(); // 转换为毫秒

            let mean = times.iter().sum::<f64>() / times.len() as f64;

            let variance =
                times.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / times.len() as f64;

            (mean, variance)
        }

        // output the results
        println!("\nAdditional Statistics:");
        println!("  Total rounds: {}", round);
        println!("  Tensor shape: [2048, 768]");
        println!("  Data type: BFloat16");
        println!(
            "  Tensor size: {:.2} MB",
            (2048 * 768 * 2) as f64 / 1024.0 / 1024.0
        );

        // calculate mean and variance for CPU -> CUDA transfer
        let (cuda_mean, cuda_variance) = calculate_stats(&to_cuda_durations);
        println!("\nCPU -> CUDA Transfer Statistics:");
        println!("  Average time: {:.3} ms", cuda_mean);
        println!("  Std: {:.3} ms", cuda_variance.sqrt());

        // calculate mean and variance for CUDA -> CPU transfer
        let (cpu_mean, cpu_variance) = calculate_stats(&to_cpu_durations);
        println!("\nCUDA -> CPU Transfer Statistics:");
        println!("  Average time: {:.3} ms", cpu_mean);
        println!("  Std: {:.3} ms", cpu_variance.sqrt());
    }

    // bench performace of torch FFN
    #[test]
    fn bench_torch_ffn_gpu() {
        // Configs
        let round = 1024;
        let batch_sizes: Vec<usize> = vec![1, 4, 16, 64];

        // generate a FFN with random weights
        let ffn = TorchFFN {
            dim: 2048,
            intermediate_dim: 768,
            weight: ExpertWeight::from_rand_linear(
                2048,
                768,
                crate::backend::DType::BFloat16,
                crate::backend::Device::CUDA(0),
            ),
            module: OnceCell::new(),
            device: Device::CUDA(0),
        };

        for batch_size in batch_sizes {
            let mut res: Vec<std::time::Duration> = vec![];

            let inp = ffn.rand_input(batch_size);
            let inp = inp.to_device(Device::CUDA(0));

            // warm up
            let _ = ffn.forward(&inp);

            for _ in 0..round {
                let now = std::time::Instant::now();
                let r = ffn.forward(&inp);
                // println!("🚩forward cost: {:?}", now.elapsed());
                res.push(now.elapsed());

                let _ = std::hint::black_box(r);
                // sleep for a while
                // std::thread::sleep(std::time::Duration::from_millis(200));
            }

            // basic info, batchsize dim intermediate ...
            println!();
            println!("🔥TorchFFN GPU forward benchmark:");
            println!("  Batch size: {}", batch_size);
            println!("  Hidden size: {}", ffn.dim);
            println!("  Intermediate size: {}", ffn.intermediate_dim);
            println!("  Data type: BFloat16");

            // summary
            let total_duration: std::time::Duration = res.iter().sum();
            let avg_duration = total_duration / round as u32;

            // calculate variance
            let avg_micros = avg_duration.as_micros() as f64;
            let variance = res
                .iter()
                .map(|d| {
                    let diff = d.as_micros() as f64 - avg_micros;
                    diff * diff
                })
                .sum::<f64>()
                / round as f64;
            let std_dev = variance.sqrt();

            // calculate per seq variance
            let avg_per_seq = avg_micros / batch_size as f64;
            let variance_per_seq = variance / (batch_size as f64 * batch_size as f64);
            let std_dev_per_seq = variance_per_seq.sqrt();

            println!(
                "⚠ TorchFFN forward {} times, avg: {:?} ± {:.2?} μs, total: {:?}",
                round, avg_duration, std_dev, total_duration
            );
            println!(
                "⚠ speed per seq: {:.2} ± {:.2} μs",
                avg_per_seq, std_dev_per_seq
            );
        }
    }

    // bench performace of torch FFN
    #[test]
    fn bench_torch_ffn_cpu() {
        // Configs
        let round = 1024;
        let batch_sizes: Vec<usize> = vec![1, 4, 16, 64];

        // generate a FFN with random weights
        let ffn = TorchFFN {
            dim: 2048,
            intermediate_dim: 768,
            weight: ExpertWeight::from_rand_linear(
                2048,
                768,
                crate::backend::DType::BFloat16,
                crate::backend::Device::CPU,
            ),
            module: OnceCell::new(),
            device: Device::CPU,
        };

        for batch_size in batch_sizes {
            let mut res: Vec<std::time::Duration> = vec![];

            let inp = ffn.rand_input(batch_size);
            // warm up
            let _ = ffn.forward(&inp);

            for _ in 0..round {
                let now = std::time::Instant::now();
                let r = ffn.forward(&inp);
                // println!("🚩forward cost: {:?}", now.elapsed());
                res.push(now.elapsed());

                let _ = std::hint::black_box(r);
                // sleep for a while
                // std::thread::sleep(std::time::Duration::from_millis(200));
            }

            println!();
            println!("🔥TorchFFN CPU forward benchmark:");
            println!("  Batch size: {}", batch_size);
            println!("  Hidden size: {}", ffn.dim);
            println!("  Intermediate size: {}", ffn.intermediate_dim);
            println!("  Data type: BFloat16");

            // summary
            let total_duration: std::time::Duration = res.iter().sum();
            let avg_duration = total_duration / round as u32;

            // calculate variance
            let avg_micros = avg_duration.as_micros() as f64;
            let variance = res
                .iter()
                .map(|d| {
                    let diff = d.as_micros() as f64 - avg_micros;
                    diff * diff
                })
                .sum::<f64>()
                / round as f64;
            let std_dev = variance.sqrt();

            // calculate per seq variance
            let avg_per_seq = avg_micros / batch_size as f64;
            let variance_per_seq = variance / (batch_size as f64 * batch_size as f64);
            let std_dev_per_seq = variance_per_seq.sqrt();

            println!(
                "⚠ TorchFFN forward {} times, avg: {:?} ± {:.2?} μs, total: {:?}",
                round, avg_duration, std_dev, total_duration
            );
            println!(
                "⚠ speed per seq: {:.2} ± {:.2} μs",
                avg_per_seq, std_dev_per_seq
            );
        }
    }

    #[test]
    fn bench_torch_ffn_queue_cpu() {
        // Configs
        let ffn_count = 256;
        let mut ffns = Vec::new();
        let round = 1;
        let batch_sizes: Vec<usize> = vec![1, 4, 16, 64];

        for _ in 0..ffn_count {
            let ffn = TorchFFN {
                dim: 2048,
                intermediate_dim: 768,
                weight: ExpertWeight::from_rand_linear(
                    2048,
                    768,
                    crate::backend::DType::BFloat16,
                    crate::backend::Device::CPU,
                ),
                module: OnceCell::new(),
                device: Device::CPU,
            };
            ffns.push(ffn);
        }

        for batch_size in batch_sizes {
            run_ffn_queue_benchmark(&ffns, batch_size, round);
        }
    }

    fn run_ffn_queue_benchmark(ffns: &[TorchFFN], batch_size: usize, round: usize) {
        let mut res: Vec<std::time::Duration> = vec![];

        // warm up all FFNs
        log::info!("Warming up {} FFNs...", ffns.len());
        for ffn in ffns {
            let inp = ffn.rand_input(batch_size);
            let _ = ffn.forward(&inp);
        }

        let inp = ffns[0].rand_input(batch_size);

        // randomly select a FFN and run forward
        for _ in 0..round {
            // randomly select a FFN
            let ffn_idx = rand::random::<u64>() % ffns.len() as u64;
            let ffn = &ffns[ffn_idx as usize];

            let now = std::time::Instant::now();
            let r = ffn.forward(&inp);
            res.push(now.elapsed());

            let _ = std::hint::black_box(r);
        }

        // output results
        println!();
        println!("🔥TorchFFN Queue CPU forward benchmark:");
        println!("  FFN count: {}", ffns.len());
        println!("  Batch size: {}", batch_size);
        println!("  Hidden size: {}", ffns[0].dim);
        println!("  Intermediate size: {}", ffns[0].intermediate_dim);
        println!("  Data type: BFloat16");

        // summary the results
        let total_duration: std::time::Duration = res.iter().sum();
        let avg_duration = total_duration / round as u32;
        println!(
            "⚠ TorchFFN queue forward {} times, avg: {:?}, total: {:?}",
            round, avg_duration, total_duration
        );
        println!(
            "⚠ speed per seq: {:?} μs",
            avg_duration.as_micros() as f64 / batch_size as f64
        );
    }
}

#[cfg(test)]
mod bench_ffn_concurrent {
    use super::*;
    use once_cell::sync::OnceCell;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn bench_torch_ffn_concurrent_vs_serial() {
        // Configs
        let ffn_count = 64;
        let batch_size = 1; // Size of each request
        let reqests_num = 512; // Reqests processed in per round
        let thread_num = 64; // Num of concurrent tasks, requests will be evenly distributed across these tasks

        // Randomly generate FFNs
        let mut ffns = Vec::new();
        for _ in 0..ffn_count {
            let ffn = TorchFFN {
                dim: 2048,
                intermediate_dim: 768,
                weight: ExpertWeight::from_rand_linear(
                    2048,
                    768,
                    crate::backend::DType::BFloat16,
                    crate::backend::Device::CPU,
                ),
                module: OnceCell::new(),
                device: Device::CPU,
            };
            ffns.push(ffn);
        }
        // Convert to Arc for sharing across threads
        let ffns: Vec<Arc<TorchFFN>> = ffns.into_iter().map(Arc::new).collect();

        println!("🔥 FFN Concurrent vs Serial Performance Comparison Test");
        println!("  FFN count: {}", ffn_count);
        println!("  Batch size: {}", batch_size);
        println!("  Requests num: {}", reqests_num);
        println!("  Thread num: {}", thread_num);
        println!();

        // Warm up all FFNs
        warm_up_ffns(&ffns, batch_size).await;

        // 1. Serial test
        let serial_results = run_serial_benchmark(&ffns, batch_size, reqests_num).await;

        // 2. Concurrent test
        let concurrent_results =
            run_concurrent_benchmark(&ffns, batch_size, reqests_num, thread_num).await;

        // 3. Compare results
        compare_results(&serial_results, &concurrent_results, batch_size, thread_num);
    }

    async fn warm_up_ffns(ffns: &[Arc<TorchFFN>], batch_size: usize) {
        println!("🚀 Warming up {} FFNs...", ffns.len());
        for ffn in ffns {
            let inp = ffn.rand_input(batch_size);
            let _ = ffn.forward(&inp);
        }
        println!("✅ Warmup completed");
        println!();
    }

    async fn run_serial_benchmark(
        ffns: &[Arc<TorchFFN>],
        batch_size: usize,
        requests_num: usize,
    ) -> Vec<Duration> {
        println!("📊 Starting serial test...");
        let mut results = Vec::new();
        let inp = ffns[0].rand_input(batch_size);

        for i in 0..requests_num {
            // Randomly select a FFN
            let ffn_idx = rand::random_range(0..ffns.len());
            let ffn = &ffns[ffn_idx];

            let start = Instant::now();
            let r = ffn.forward(&inp);
            let elapsed = start.elapsed();
            results.push(elapsed);

            let _ = std::hint::black_box(r);

            let _ = i;
            // if (i + 1) % 20 == 0 {
            //     println!("  Serial progress: {}/{}", i + 1, requests_num);
            // }
        }

        println!("✅ Serial test completed");
        results
    }

    async fn run_concurrent_benchmark(
        ffns: &[Arc<TorchFFN>],
        batch_size: usize,
        requests_num: usize,
        concurrent_tasks: usize,
    ) -> Vec<Duration> {
        println!("📊 Starting concurrent test...");
        // let mut all_results = Vec::new();

        // Ensure each task executes at least once with even distribution
        let base_reqs_num = requests_num / concurrent_tasks;
        let extra_reqs_num = requests_num % concurrent_tasks;

        println!(
            "  Base requests per task: {}, extra requests allocated to first {} tasks",
            base_reqs_num, extra_reqs_num
        );

        let mut join_set = JoinSet::new();

        // Launch multiple concurrent tasks
        for task_id in 0..concurrent_tasks {
            let ffns_clone = ffns.to_vec();
            let batch_size = batch_size;

            // First few tasks get an extra round
            let reqs_num = if task_id < extra_reqs_num {
                base_reqs_num + 1
            } else {
                base_reqs_num
            };

            // If base_rounds is 0 and this task has no extra rounds, execute at least once
            let reqs_num = if reqs_num == 0 { 1 } else { reqs_num };

            join_set.spawn(async move {
                let mut task_results = Vec::new();
                let inp = ffns_clone[0].rand_input(batch_size);

                for i in 0..reqs_num {
                    // Randomly select a FFN for this task
                    let ffn_idx = rand::random_range(0..ffns_clone.len());
                    let ffn = &ffns_clone[ffn_idx];

                    let start = Instant::now();
                    let r = ffn.forward(&inp);
                    let elapsed = start.elapsed();
                    task_results.push(elapsed);

                    let _ = std::hint::black_box(r);

                    let _ = i;
                    // if reqs_num > 10 && (i + 1) % 10 == 0 {
                    //     println!("  Concurrent task {} progress: {}/{}", task_id, i + 1, reqs_num);
                    // }
                }

                // println!("  Task {} completed: {} executions", task_id, reqs_num);
                task_results
            });
        }

        // join_all tasks at once
        let task_results = join_set.join_all().await;

        println!("✅ Concurrent test completed");
        task_results.into_iter().flatten().collect()
    }

    fn compare_results(
        serial_results: &[Duration],
        concurrent_results: &[Duration],
        batch_size: usize,
        concurrent_tasks: usize,
    ) {
        // Check if results are empty
        if serial_results.is_empty() {
            println!("❌ Serial test results are empty!");
            return;
        }
        if concurrent_results.is_empty() {
            println!("❌ Concurrent test results are empty!");
            return;
        }

        let serial_avg = serial_results.iter().sum::<Duration>() / serial_results.len() as u32;
        let concurrent_avg =
            concurrent_results.iter().sum::<Duration>() / concurrent_results.len() as u32;

        let serial_min = serial_results.iter().min().unwrap();
        let serial_max = serial_results.iter().max().unwrap();
        let concurrent_min = concurrent_results.iter().min().unwrap();
        let concurrent_max = concurrent_results.iter().max().unwrap();

        println!();
        println!("📈 Performance Comparison Results:");
        println!(
            "  Actual tests: {} serial rounds, {} concurrent rounds",
            serial_results.len(),
            concurrent_results.len()
        );
        println!("┌─────────────────────────────────────────┐");
        println!("│            Serial Test Results          │");
        println!("├─────────────────────────────────────────┤");
        println!(
            "│ Avg latency: {:>8} μs                │",
            serial_avg.as_micros()
        );
        println!(
            "│ Min latency: {:>8} μs                │",
            serial_min.as_micros()
        );
        println!(
            "│ Max latency: {:>8} μs                │",
            serial_max.as_micros()
        );
        println!(
            "│ Per sequence: {:>8.1} μs               │",
            serial_avg.as_micros() as f64 / batch_size as f64
        );
        println!("└─────────────────────────────────────────┘");

        println!();
        println!("┌─────────────────────────────────────────┐");
        println!(
            "│    Concurrent Test Results ({} tasks)   │",
            concurrent_tasks
        );
        println!("├─────────────────────────────────────────┤");
        println!(
            "│ Avg latency:    {:>8} μs             │",
            concurrent_avg.as_micros()
        );
        println!(
            "│ Min latency:    {:>8} μs             │",
            concurrent_min.as_micros()
        );
        println!(
            "│ Max latency:    {:>8} μs             │",
            concurrent_max.as_micros()
        );
        println!(
            "│ Per sequence:   {:>8.1} μs             │",
            concurrent_avg.as_micros() as f64 / batch_size as f64
        );
        println!("└─────────────────────────────────────────┘");

        println!();
        let slowdown_ratio = concurrent_avg.as_micros() as f64 / serial_avg.as_micros() as f64;
        println!("🎯 Key Metrics:");
        println!("  • Performance degradation: {:.2}x", slowdown_ratio);
        println!(
            "  • Concurrent latency increase: {} μs",
            concurrent_avg.as_micros() as i64 - serial_avg.as_micros() as i64
        );

        // Latency distribution analysis
        println!();
        println!("📊 Latency Distribution Analysis:");
        print_latency_distribution("Serial", serial_results);
        print_latency_distribution("Concurrent", concurrent_results);
    }

    fn print_latency_distribution(name: &str, results: &[Duration]) {
        let mut latencies: Vec<u64> = results.iter().map(|d| d.as_micros() as u64).collect();
        latencies.sort();

        let len = latencies.len();
        let p50 = latencies[len * 50 / 100];
        let p90 = latencies[len * 90 / 100];
        let p95 = latencies[len * 95 / 100];
        let p99 = latencies[len * 99 / 100];

        println!("  {} latency distribution:", name);
        println!(
            "    P50: {} μs, P90: {} μs, P95: {} μs, P99: {} μs",
            p50, p90, p95, p99
        );
    }
}
