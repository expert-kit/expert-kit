#![feature(f16)]

mod bench;
use std::{
    fs::File,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use bench::{BenchmarkExpert, BenchmarkerImpl};
use clap::{Parser, ValueEnum};
use ek_computation::{ffn::expert_torch::TorchFFN, x::ExpertBackendType};
use ek_computation::{
    ffn::{ExpertBackend, expert_ort},
    x,
};
use polars::prelude::{IntoLazy, ParquetWriter, col};
extern crate pretty_env_logger;
#[macro_use]
extern crate log;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Mode {
    ScanBatch,
}
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[clap(short, long, value_enum, default_value_t=Mode::ScanBatch)]
    mode: Mode,

    #[clap(value_enum, short, long,default_value_t=x::ExpertBackendType::Torch)]
    backend: x::ExpertBackendType,

    #[clap(short, long, value_delimiter = ',',default_values_t=vec![1,2,4,8])]
    range: Vec<usize>,

    #[clap(short, long, default_value_t = 2048)]
    dim: usize,

    #[clap(long, default_value_t = 7168)]
    hidden: usize,

    #[clap(short, long, value_delimiter = ',', default_value_t = 10)]
    experts: usize,

    #[arg(long, value_name = "FILE")]
    onnx: Option<PathBuf>,

    #[arg(short, long, value_name = "FILE",default_value_t= ("./data/benchmark").to_string())]
    output_dir: String,
}

fn main() {
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "Debug");
        }
    }
    pretty_env_logger::init();
    let m = Cli::parse();

    let expert_count = m.experts;
    let mut experts: Vec<BenchmarkExpert> = vec![];
    let instance = ek_computation::x::EKInstance {
        dim: m.dim,
        hidden: m.hidden,
        backend: m.backend.into(),
    };
    for i in 0..expert_count {
        match m.backend {
            ExpertBackendType::Torch => {
                info!("create torch expert {}", i);
                let exp = TorchFFN::new(instance);
                experts.push(BenchmarkExpert(ExpertBackend::Torch(exp)));
            }
            _ => todo!(),
            // ::Ort => {
            //     if m.onnx.is_none() {
            //         panic!("Ort backend requires an onnx file");
            //     }
            //     let exp = expert_ort::OnnxFFN::new(m.onnx.clone().unwrap(), m.dim, m.hidden);
            //     experts.push(GenericExpert::Ort(exp));
            // }
        }
    }
    let mut bencher = BenchmarkerImpl::new(experts);
    let mut df = bencher.iterations(1).scan_batch(&m.range);
    let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let name = format!(
        "{:?}_{}_{}_{}_{}.parquet",
        m.backend,
        m.dim,
        m.hidden,
        m.experts,
        ts.as_secs()
    );
    let full_fp = std::path::Path::new(&m.output_dir).join(name);
    info!("write to: {}", full_fp.to_str().unwrap());
    let file = File::create(full_fp).unwrap();
    ParquetWriter::new(file)
        .with_compression(polars::prelude::ParquetCompression::Snappy)
        .finish(&mut df)
        .unwrap();
    let glance = df
        .lazy()
        .group_by([col("batch")])
        .agg([col("elapsed").mean()])
        .sort_by_exprs([col("batch")], Default::default())
        .collect()
        .unwrap();

    println!("{}", glance);
}
