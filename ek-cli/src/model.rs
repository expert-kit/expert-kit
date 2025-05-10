use clap::Subcommand;
use ek_base::error::EKResult;
use ek_computation::state::writer::StateWriterImpl;
use log::info;

#[derive(Subcommand, Debug)]
pub enum ModelCommand {
    #[command(about = "update or create a model in meta db")]
    Upsert {
        #[arg(long, help = "weight server address (e.g. http://localhost:6543)")]
        weight_server: String,
        #[arg(long, help = "unique model name")]
        model_name: String,
    },
}

async fn upsert_model(weight_server: &str, model_name: &str) -> EKResult<()> {
    let writer = StateWriterImpl::new();
    writer.model_upsert(weight_server, model_name).await?;
    info!("model upsert successfully");
    Ok(())
}

pub async fn execute_model(cmd: ModelCommand) -> EKResult<()> {
    match cmd {
        ModelCommand::Upsert {
            weight_server,
            model_name,
        } => upsert_model(&weight_server, &model_name).await,
    }
}
