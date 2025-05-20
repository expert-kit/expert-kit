#![feature(f16)]

mod bench;
use std::{
    fs::File,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use bench::{ExpertBenchmark, Benchmarker};
use clap::{Parser, ValueEnum};
use ek_computation::{ffn::ExpertBackend, x};
use ek_computation::{ffn::expert_torch::TorchFFN, x::ExpertBackendType};
use polars::prelude::ParquetWriter;
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

    #[clap(short, long, value_delimiter = ',', default_values_t=vec![1,2,4,8,16,32,64,128,256])]
    batch_sizes: Vec<usize>,

    #[clap(short, long, default_value_t = 2048)]
    dim: usize,

    #[clap(long, default_value_t = 7168)]
    hidden: usize,

    #[clap(short, long, default_value_t = 8)]
    experts: usize,
    
    #[clap(short = 'i', long, default_value_t = 4)]
    iterations: usize,
    
    #[clap(short = 'r', long, default_value_t = 1)]
    repeats: usize,

    #[arg(long, value_name = "FILE")]
    onnx: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE", default_value_t = ("./output").to_string())]
    output_dir: String,
}

fn main() {
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "Debug");
        }
    }
    pretty_env_logger::init();
    let cli_args = Cli::parse();

    let expert_count = cli_args.experts;
    let mut experts: Vec<ExpertBenchmark> = vec![];
    let instance = ek_computation::x::EKInstance {
        dim: cli_args.dim,
        hidden: cli_args.hidden,
        backend: cli_args.backend,
    };
    
    info!("Creating {} expert models...", expert_count);
    for i in 0..expert_count {
        match cli_args.backend {
            ExpertBackendType::Torch => {
                info!("Creating Torch expert {}", i);
                let exp = TorchFFN::new(instance);
                experts.push(ExpertBenchmark(ExpertBackend::Torch(exp)));
            }
            _ => todo!(),
            // ::Ort => {
            //     if cli_args.onnx.is_none() {
            //         panic!("Ort backend requires an onnx file");
            //     }
            //     let exp = expert_ort::OnnxFFN::new(cli_args.onnx.clone().unwrap(), cli_args.dim, cli_args.hidden);
            //     experts.push(GenericExpert::Ort(exp));
            // }
        }
    }
    
    info!("Setting up benchmark with {} iterations and {} experiment repeats", 
          cli_args.iterations, cli_args.repeats);
    
    let mut benchmarker = Benchmarker::new(experts);
    let mut results = benchmarker
        .iterations(cli_args.iterations)
        .repeats(cli_args.repeats)
        .scan_batch(&cli_args.batch_sizes);
    
    // Generate timestamp for file naming
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let output_filename = format!(
        "{:?}_{}_{}_{}_{}_r{}.parquet",
        cli_args.backend,
        cli_args.dim,
        cli_args.hidden,
        cli_args.experts,
        timestamp.as_secs(),
        cli_args.repeats
    );
    
    let output_path = std::path::Path::new(&cli_args.output_dir).join(output_filename);
    info!("Writing results to: {}", output_path.to_str().unwrap());
    
    // Save detailed results to Parquet file
    let file = File::create(output_path).unwrap();
    ParquetWriter::new(file)
        .with_compression(polars::prelude::ParquetCompression::Snappy)
        .finish(&mut results)
        .unwrap();
    
    // Calculate and display summary statistics
    if cli_args.repeats > 1 {
        let summary_by_experiment = benchmarker.calculate_summary(&results);
        println!("Per-experiment summary:");
        println!("{}", summary_by_experiment);
    }
    
    // Calculate and display final summary with statistics across all experiments
    let final_summary = benchmarker.calculate_final_summary(&results);
    println!("Final summary (across all experiments):");
    println!("{}", final_summary);
}