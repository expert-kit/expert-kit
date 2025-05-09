use std::path::PathBuf;

use ek_db::weight_srv;

use clap::{Parser, Subcommand};
extern crate pretty_env_logger;

#[derive(Subcommand, Debug)]
enum Command {
    #[command()]
    WeightServer {
        #[arg(long, default_value_t = ("0.0.0.0").to_string())]
        host: String,
        #[arg(short, long, default_value_t = 6543)]
        port: u16,
        #[arg(long)]
        model: Vec<PathBuf>,
    },
}
/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct RootCli {
    #[command(subcommand)]
    command: Command,
}

#[tokio::main]
async fn main() {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("debug"));
    let cli = RootCli::parse();

    let res = match cli.command {
        Command::WeightServer { host, port, model } => {
            weight_srv::listen(&model, (host, port)).await
        }
    };
    if let Err(e) = res {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
