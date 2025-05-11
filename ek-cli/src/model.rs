use clap::Subcommand;
use ek_base::error::EKResult;
use ek_computation::state::writer::StateWriterImpl;
use log::info;

#[derive(Subcommand, Debug)]
pub enum ModelCommand {
    #[command(about = "update or create a model in meta db")]
    Upsert {
        #[arg(long, help = "unique model name")]
        name: String,
    },
}

async fn upsert_model(weight_server: &str, model_name: &str) -> EKResult<()> {
    let writer = StateWriterImpl::new();
    writer.model_upsert(weight_server, model_name).await?;
    info!("model upsert successfully");
    Ok(())
}

pub async fn execute_model(cmd: ModelCommand) -> EKResult<()> {
    let settings = ek_base::config::get_ek_settings();
    let weight_server =
        settings
            .weight
            .server
            .clone()
            .ok_or(ek_base::error::EKError::InvalidInput(
                "weight server not set".to_string(),
            ))?;

    match cmd {
        ModelCommand::Upsert { name } => upsert_model(&weight_server.addr, &name).await,
    }
}
