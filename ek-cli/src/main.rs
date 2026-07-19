use std::{env, mem::transmute, path::PathBuf, process::Command as ProcessCommand};
mod db;
mod doctor;
mod model;
mod pretrain;
mod schedule;

mod weight_index;
use db::execute_db;
use doctor::doctor_main;
use ek_base::config::get_ek_settings_base;
use ek_computation::controller::controller_main;
use env_logger::fmt::default_kv_format;
use opentelemetry::{
    KeyValue, propagation::TextMapCompositePropagator, trace::TracerProvider as _,
};
use std::io::Write;
use weight_index::{WeightIndexCommand, execute_weight_index};

use tokio::runtime::Runtime;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use clap::{Parser, Subcommand};
use ek_db::weight_srv;
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
        /// Disable the expert index fast path (for ablation study baseline).
        /// Forces every request to use mmap + re-serialization regardless of
        /// whether an index exists in the cache directory.
        #[arg(long, default_value_t = false)]
        no_index: bool,
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

    #[command(about = "expert weight index operations")]
    Weight {
        #[command(subcommand)]
        command: WeightIndexCommand,
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

const DEFAULT_THREAD_NUM: usize = 6;

fn python_worker_command(config: Option<String>) -> Result<ProcessCommand, &'static str> {
    let Some(config) = config else {
        return Err("The Python Worker requires --config or EK_CONFIG");
    };
    let mut command = ProcessCommand::new("ek-worker");
    command.arg("--config").arg(config);
    Ok(command)
}

fn resolve_python_worker_config(
    cli_config: Option<String>,
    environment_config: Option<String>,
) -> Option<String> {
    cli_config.or(environment_config.filter(|value| !value.trim().is_empty()))
}

fn run_python_worker(config: Option<String>) -> ! {
    let mut command = match python_worker_command(config) {
        Ok(command) => command,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    };

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        let error = command.exec();
        eprintln!("Failed to start ek-worker: {error}");
        std::process::exit(1);
    }
    #[cfg(not(unix))]
    {
        match command.status() {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(error) => {
                eprintln!("Failed to start ek-worker: {error}");
                std::process::exit(1);
            }
        }
    }
}

/// Initialize the shared Tokio runtime for Rust commands.
fn init_tokio_runtime(_command: &Command) -> Result<Runtime, std::io::Error> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(DEFAULT_THREAD_NUM)
        .enable_all()
        .build()
}

fn main() {
    let cli = RootCli::parse();
    if cli.debug {
        unsafe { std::env::set_var("RUST_LOG", "debug") };
    }
    if matches!(&cli.command, Command::Worker {}) {
        let environment_config = std::env::var("EK_CONFIG").ok();
        run_python_worker(resolve_python_worker_config(
            cli.config.clone(),
            environment_config,
        ));
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
    log::info!("config source: {config_src:?}");
    let settings = ek_base::config::get_ek_settings();
    log::info!("settings: {settings:?}");

    // Init log
    init_log();

    // Init tokio runtime (Prepare for cpu affinity settings)
    let tokio_rt = match init_tokio_runtime(&cli.command) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create Tokio runtime: {e}");
            std::process::exit(1);
        }
    };

    let res = tokio_rt.block_on(async {
        // Must place tracing subscriber init in tokio runtime block
        init_tracing_subscriber(command_name);
        match cli.command {
            Command::Pretrain { command } => execute_pretrain(command).await,
            Command::Worker {} => unreachable!("Worker command is replaced before Tokio startup"),
            Command::Controller {} => controller_main().await,
            Command::Doctor {} => doctor_main().await,
            Command::WeightServer {
                host,
                port,
                model,
                no_index,
            } => {
                let model: &[PathBuf] = unsafe { transmute(model.as_slice()) };
                let cache_dir = if no_index {
                    None
                } else {
                    match &ek_base::config::get_ek_settings().weight.cache {
                        ek_base::config::OpenDALStorage::Fs(cfg) => Some(PathBuf::from(&cfg.path)),
                        _ => None,
                    }
                };
                weight_srv::server::listen(model, cache_dir, (host, port)).await
            }
            Command::Weight { command } => execute_weight_index(command).await,
            Command::DB { command } => execute_db(command).await,
            Command::Model { command } => execute_model(command).await,
            Command::Schedule { command } => execute_schedule(command).await,
        }
    });

    if let Err(e) = res {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod worker_launcher_tests {
    use super::*;

    #[test]
    fn builds_the_active_environment_worker_command() {
        let command = python_worker_command(Some("/tmp/worker.yaml".to_owned())).unwrap();

        assert_eq!(command.get_program(), "ek-worker");
        assert_eq!(
            command.get_args().collect::<Vec<_>>(),
            ["--config", "/tmp/worker.yaml"]
        );
    }

    #[test]
    fn requires_an_explicit_worker_config() {
        assert_eq!(
            python_worker_command(None).unwrap_err(),
            "The Python Worker requires --config or EK_CONFIG"
        );
    }

    #[test]
    fn resolves_worker_config_from_environment() {
        assert_eq!(
            resolve_python_worker_config(None, Some("/tmp/from-environment.yaml".to_owned())),
            Some("/tmp/from-environment.yaml".to_owned())
        );
    }

    #[test]
    fn explicit_worker_config_overrides_environment() {
        assert_eq!(
            resolve_python_worker_config(
                Some("/tmp/from-argument.yaml".to_owned()),
                Some("/tmp/from-environment.yaml".to_owned())
            ),
            Some("/tmp/from-argument.yaml".to_owned())
        );
    }

    #[test]
    fn ignores_empty_environment_worker_config() {
        assert_eq!(
            resolve_python_worker_config(None, Some("  ".to_owned())),
            None
        );
    }
}
