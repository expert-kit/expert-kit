#![feature(random)]
use std::{mem::transmute, path::PathBuf};
mod db;
mod doctor;
mod model;
mod pretrain;
mod schedule;

mod onnx;
use db::execute_db;
use doctor::doctor_main;
use ek_base::config::get_ek_settings_base;
use ek_computation::{controller::controller_main, worker::worker_main};
use ek_db::weight_srv;

use clap::{Parser, Subcommand};
use model::execute_model;
use pretrain::{PretrainCommand, execute_pretrain};
use schedule::execute_schedule;
extern crate pretty_env_logger;

#[derive(Subcommand, Debug)]
enum Command {
    #[command(about = "check the environment")]
    Doctor {},

    #[command(about = "run expert-kit worker")]
    Worker {},

    #[command(about = "run expert-kit controller")]
    Controller {},

    #[command(about = "run expert-kit weight server")]
    WeightServer {
        #[arg(long, default_value_t = ("0.0.0.0").to_string())]
        host: String,
        #[arg(short, long, default_value_t = 6543)]
        port: u16,
        #[arg(long)]
        model: Vec<PathBuf>,
    },

    #[command(about = "safetensor pretrain weight manipulation")]
    Pretrain {
        #[command(subcommand)]
        command: PretrainCommand,
    },

    #[command(about = "low-level db operations")]
    DB {
        #[command(subcommand)]
        command: db::DBCommand,
    },

    #[command(about = "model operations")]
    Model {
        #[command(subcommand)]
        command: model::ModelCommand,
    },

    #[command(about = "schedule operations")]
    Schedule {
        #[command(subcommand)]
        command: schedule::ScheduleCommand,
    },

    #[command(about = "onnx operations")]
    Onnx {
        #[command(subcommand)]
        command: onnx::OnnxCommand,
    },
}

/// Expert Kit is an efficient foundation of Expert Parallelism (EP) for MoE model Inference on heterogenous hardware
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct RootCli {
    #[arg(long, default_value_t = false)]
    debug: bool,
    #[arg(long, global = true)]
    config: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 48)]
async fn main() {
    let cli = RootCli::parse();
    if cli.debug {
        unsafe { std::env::set_var("RUST_LOG", "debug") };
    }
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));
    let mut config_src = vec![];

    if let Ok(path) = std::env::var("EK_CONFIG") {
        config_src.push(path);
    }

    if let Some(path) = cli.config {
        config_src.push(path.to_string());
    }
    get_ek_settings_base(
        &config_src
            .as_slice()
            .iter()
            .map(|x| x.as_str())
            .collect::<Vec<_>>(),
    );
    let res = match cli.command {
        Command::Onnx { command } => onnx::execute_onnx(command).await,
        Command::Pretrain { command } => execute_pretrain(command).await,
        Command::Worker {} => worker_main().await,
        Command::Controller {} => controller_main().await,
        Command::Doctor {} => doctor_main().await,
        Command::WeightServer { host, port, model } => {
            let model: &[PathBuf] = unsafe { transmute(model.as_slice()) };
            weight_srv::server::listen(model, (host, port)).await
        }
        Command::DB { command } => execute_db(command).await,
        Command::Model { command } => execute_model(command).await,
        Command::Schedule { command } => execute_schedule(command).await,
    };
    if let Err(e) = res {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
