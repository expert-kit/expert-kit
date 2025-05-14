use clap::Subcommand;
use ek_base::{config::get_ek_settings, error::EKResult};
use ek_computation::state::writer::StateWriterImpl;
use ek_db::dal::op_from_settings;
use log::info;

#[derive(Subcommand, Debug)]
pub enum ModelCommand {
    #[command(about = "update or create a model in meta db")]
    Upsert {
        #[arg(long, help = "unique model name")]
        name: String,
    },
    CleanCache {
        #[arg(long, help = "model name")]
        name: String,
    },
}

async fn upsert_model(weight_server: &str, model_name: &str) -> EKResult<()> {
    let writer = StateWriterImpl::new();
    writer.model_upsert(weight_server, model_name).await?;
    info!("model upsert successfully");
    Ok(())
}

async fn clean_cache(model_name: &str) -> EKResult<()> {
    let settings = get_ek_settings();
    let cache_dir = settings.weight.cache.clone();
    let op = op_from_settings(&cache_dir);

    let f = op.list_with(model_name).recursive(true).await?;
    log::info!("found {} files in cache", f.len());

    for file in f {
        op.delete(format!("{}/{}", model_name, file.name()).as_str())
            .await?;
    }
    info!("model cache cleaned successfully");
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
        ModelCommand::CleanCache { name } => clean_cache(&name).await,
    }
}
