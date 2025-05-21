#![feature(f16)]

mod bench;
use std::{
    fs::File,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use bench::{Benchmarker, ExpertBenchmark};
use clap::{Parser, ValueEnum};
use ek_computation::{
    backend::{DType, Device, ort::NDArrayTensor, torch::TchTensor},
    ffn::{
        expert_torch::TorchFFN,
        meta::{Expert, ExpertWeight},
    },
    x::ExpertBackendType,
};
use ek_computation::{
    ffn::{ExpertBackend, expert_ort},
    x,
};
use polars::prelude::{IntoLazy, ParquetWriter, col};
extern crate pretty_env_logger;
#[macro_use]
extern crate log;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum BenchmarkMode {
    ScanBatch,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_enum, default_value_t=BenchmarkMode::ScanBatch)]
    mode: BenchmarkMode,

    #[clap(value_enum, short, long, default_value_t=x::ExpertBackendType::Torch)]
    backend: x::ExpertBackendType,

    #[clap(long, value_delimiter = ',', default_values_t=vec![1,2,4,8,16,32,64,128,256])]
    batch_sizes: Vec<usize>,

    #[clap(short, long, default_value_t = 2048)]
    intermediate_dim: usize,

    #[clap(long, default_value_t = 7168)]
    hidden_dim: usize,

    #[clap(short, long, default_value_t = 8)]
    experts: usize,

    #[clap(short = 'i', long, default_value_t = 4)]
    iterations: usize,

    #[clap(short = 'r', long, default_value_t = 1)]
    repeats: usize,

    #[arg(long, value_name = "FILE")]
    onnx: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE", default_value_t = ("./data").to_string())]
    output_dir: String,

    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() {
    let m = Cli::parse();
    if m.verbose {
        unsafe {
            std::env::set_var("RUST_LOG", "Debug");
        }
    }
    pretty_env_logger::init();
    let expert_count = m.experts;
    let mut experts: Vec<ExpertBenchmark> = vec![];
    let instance = ek_computation::x::EKInstance {
        hidden: m.hidden_dim,
        intermediate: m.intermediate_dim,
        backend: m.backend,
    };

    info!("Creating {} expert models...", expert_count);
    for i in 0..expert_count {
        match m.backend {
            ExpertBackendType::Torch => {
                info!("Creating Torch expert {}", i);
                let rand_weight: ExpertWeight<TchTensor> = ExpertWeight::from_rand_linear(
                    m.hidden_dim,
                    m.intermediate_dim,
                    DType::Float,
                    Device::CPU,
                );

                let exp = TorchFFN::construct(instance, rand_weight).unwrap();
                experts.push(ExpertBenchmark(ExpertBackend::Torch(exp)));
            }
            ExpertBackendType::Onnx => {
                let rand_weight: ExpertWeight<NDArrayTensor<f32>> = ExpertWeight::from_rand_matmul(
                    m.hidden_dim,
                    m.intermediate_dim,
                    DType::Float,
                    Device::CPU,
                );
                log::info!("rand weight: {}", rand_weight);
                let exp = expert_ort::OnnxFFN::new(
                    m.hidden_dim as i64,
                    m.intermediate_dim as i64,
                    DType::Float,
                    rand_weight,
                )
                .unwrap();
                experts.push(ExpertBenchmark(ExpertBackend::OnnxF32(exp)));
            }
        }
    }
    let mut bencher = Benchmarker::new(experts);
    let mut df = bencher.iterations(1).scan_batch(&m.batch_sizes);
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let output_filename = format!(
        "{:?}_{}_{}_{}_{}.parquet",
        m.backend,
        m.intermediate_dim,
        m.hidden_dim,
        m.experts,
        ts.as_secs()
    );

    let output_path = std::path::Path::new(&m.output_dir).join(output_filename);
    info!("Writing results to: {}", output_path.to_str().unwrap());

    // Save detailed results to Parquet file
    let file = File::create(output_path).unwrap();
    ParquetWriter::new(file)
        .with_compression(polars::prelude::ParquetCompression::Snappy)
        .finish(&mut df)
        .unwrap();

    // Calculate and display summary statistics
    if m.repeats > 1 {
        let summary_by_experiment = bencher.calculate_summary(&df);
        println!("Per-experiment summary:");
        println!("{}", summary_by_experiment);
    }

    // Calculate and display final summary with statistics across all experiments
    let final_summary = bencher.calculate_final_summary(&df);
    println!("Final summary (across all experiments):");
    println!("{}", final_summary);
}
