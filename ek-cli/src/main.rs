use std::{env, mem::transmute, path::PathBuf};
mod db;
mod doctor;
mod model;
mod pretrain;
mod schedule;

mod onnx;
mod affinity;
use db::execute_db;
use doctor::doctor_main;
use ek_base::config::get_ek_settings_base;
use ek_computation::{controller::controller_main, worker::worker_main};
use env_logger::fmt::default_kv_format;
use opentelemetry::{
    KeyValue, propagation::TextMapCompositePropagator, trace::TracerProvider as _,
};
use affinity::try_apply_cpu_affinity;
use std::io::Write;

use tokio::runtime::Runtime;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use ek_db::weight_srv;

use clap::{Parser, Subcommand};
use model::execute_model;
use opentelemetry_sdk::{
    Resource,
    propagation::{BaggagePropagator, TraceContextPropagator},
    trace::{RandomIdGenerator, Sampler, SdkTracerProvider},
};
use opentelemetry_semantic_conventions::{
    SCHEMA_URL,
    resource::{DEPLOYMENT_ENVIRONMENT_NAME, SERVICE_VERSION},
};
use pretrain::{PretrainCommand, execute_pretrain};
use schedule::execute_schedule;
use tracing::Level;

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

fn init_log() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .write_style(env_logger::WriteStyle::Auto)
        .target(env_logger::Target::Stderr)
        .format(|buf, record| {
            let level_color = buf.default_level_style(record.level());
            let timestamp = buf.timestamp();
            let level = record.level();
            let kv = record.key_values();
            let _ = write!(
                buf,
                "<{level_color}{level}{level_color:#}>({timestamp}) {} ",
                record.args(),
            );
            default_kv_format(buf, kv).unwrap();
            writeln!(buf).unwrap();
            Ok(())
        })
        .init();
}
fn resource(cmd: &'static str) -> Resource {
    Resource::builder()
        .with_service_name(cmd)
        .with_schema_url(
            [
                KeyValue::new(SERVICE_VERSION, env!("CARGO_PKG_VERSION")),
                KeyValue::new(DEPLOYMENT_ENVIRONMENT_NAME, "develop"),
            ],
            SCHEMA_URL,
        )
        .build()
}

fn init_tracer_provider(svc_name: &'static str) -> SdkTracerProvider {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()
        .unwrap();

    let provider = SdkTracerProvider::builder()
        // Customize sampling strategy
        .with_sampler(Sampler::AlwaysOn)
        // If export trace to AWS X-Ray, you can use XrayIdGenerator
        .with_id_generator(RandomIdGenerator::default())
        .with_resource(resource(svc_name))
        .with_batch_exporter(exporter)
        .build();
    let baggage_propagator = BaggagePropagator::new();
    let trace_context_propagator = TraceContextPropagator::new();
    let composite_propagator = TextMapCompositePropagator::new(vec![
        Box::new(baggage_propagator),
        Box::new(trace_context_propagator),
    ]);
    opentelemetry::global::set_text_map_propagator(composite_propagator);
    provider
}
fn init_tracing_subscriber(svc_name: &'static str) {
    let tracer_provider = init_tracer_provider(svc_name);
    let tracer = tracer_provider.tracer("tracing-otel-subscriber");
    tracing_subscriber::registry()
        .with(tracing_subscriber::filter::LevelFilter::from_level(
            Level::INFO,
        ))
        // .with(
        //     tracing_subscriber::fmt::layer()
        //         .with_thread_ids(true)
        //         .with_span_events(FmtSpan::NONE),
        // )
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .init();
}

fn get_command_name(cmd: &Command) -> &'static str {
    match cmd {
        Command::Worker {} => "worker",
        Command::Controller {} => "controller",
        _ => "others",
    }
}

const DEFAULT_THREAD_NUM:usize = 48;

/// Init tokio runtime based on command
fn init_tokio_runtime(command: &Command) -> Result<Runtime, std::io::Error> {
    match command {
        Command::Worker {} => {
            // Apply CPU affinity before creating runtime for worker
            let settings = ek_base::config::get_ek_settings();
            if let Err(e) = try_apply_cpu_affinity(&settings.worker) {
                log::warn!("Failed to apply CPU affinity before runtime creation: {}", e);
            } else {
                log::debug!("✅ CPU affinity applied before Tokio runtime creation");
            }
            
            // Determine worker thread count based on CPU affinity configuration
            let worker_threads = if let Some(advanced) = &settings.worker.advanced {
                if let Some(cpu_config) = &advanced.cpu_affinity {
                    cpu_config.cores.as_ref().map(|cores| cores.len()).unwrap_or_else(|| DEFAULT_THREAD_NUM)
                } else {
                    DEFAULT_THREAD_NUM
                }
            } else {
                DEFAULT_THREAD_NUM
            };
            
            log::info!("Creating Tokio runtime with {} worker threads", worker_threads);
            
            // TODO: hardcoded threadnum for now, need to be improved later
            // Create runtime with limited worker threads
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .max_blocking_threads(64)
                .enable_all()
                .build()
        }
        _ => {
            // Use default runtime for other commands
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(DEFAULT_THREAD_NUM)
                .enable_all()
                .build()
        }
    }
}

fn main() {
    let cli = RootCli::parse();
    if cli.debug {
        unsafe { std::env::set_var("RUST_LOG", "debug") };
    }
    let command_name = get_command_name(&cli.command);

    // Init config
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
    log::info!("config source: {:?}", config_src);
    let settings = ek_base::config::get_ek_settings();
    log::info!("settings: {:?}", settings);

    // Init log
    init_log();

    // Init tokio runtime (Prepare for cpu affinity settings)
    let tokio_rt = match init_tokio_runtime(&cli.command) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create Tokio runtime: {}", e);
            std::process::exit(1);
        }
    };

    let res = tokio_rt.block_on(async {
        // Must place tracing subscriber init in tokio runtime block
        init_tracing_subscriber(command_name);
        match cli.command {
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
        }
    });

    if let Err(e) = res {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
