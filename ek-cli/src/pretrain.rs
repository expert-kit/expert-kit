use std::{collections::HashMap, path::PathBuf};

use clap::Subcommand;
use ek_base::error::EKResult;
use ek_db::safetensor::transformer::{TransformerModelDesc, TransformerPretrained};
use tokio::{
    fs::{self, File},
    io::AsyncWriteExt,
};

#[derive(Subcommand, Debug)]
pub enum PretrainCommand {
    #[command(about = "extract attention from pretrained weight")]
    ExtractAttn {
        #[arg(long, help = "input path")]
        input: String,
        #[arg(long, help = "output path")]
        output: String,
    },
}

pub async fn execute_pretrain(cmd: PretrainCommand) -> EKResult<()> {
    match cmd {
        PretrainCommand::ExtractAttn { input, output } => {
            extract_attn(input.as_str(), output.as_str()).await?;
        }
    }
    Ok(())
}

async fn extract_attn(input: &str, output: &str) -> EKResult<()> {
    let desc = TransformerModelDesc {
        root: input.to_string().into(),
        ..Default::default()
    };
    let output = PathBuf::from(output);
    if !output.exists() {
        log::info!("output path not exists, creating {}", output.display());
        fs::create_dir_all(output.clone()).await?;
    }
    let pretrained = TransformerPretrained::try_from_desc(&desc)?;
    let names = pretrained.layer_names_except_experts();
    log::info!("found layer except experts {}", names.len());
    let mut converted = vec![];
    let out_st_name = "converted.safetensors";
    let mut new_map = HashMap::new();

    for layer in names {
        log::info!("extracting layer {}", layer);
        let tv = pretrained.get_tensor(layer.as_str()).await?;
        converted.push((layer.clone(), tv));
        new_map.insert(layer.clone(), out_st_name);
    }

    let mut model_weight_map_fp = File::create(output.clone().join(desc.weight_map_name)).await?;
    let mut converted_st_fp = File::create(output.clone().join(out_st_name)).await?;
    let new_map_json = serde_json::json!({
        "weight_map": new_map,
    });
    model_weight_map_fp
        .write_all(new_map_json.to_string().as_bytes())
        .await?;

    let serialized = safetensors::tensor::serialize(converted, &None)?;
    converted_st_fp.write_all(&serialized).await?;

    log::info!("converted safetensors saved to {}", output.display());

    Ok(())
}
