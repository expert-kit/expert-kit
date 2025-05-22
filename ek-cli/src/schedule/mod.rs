use std::{path::PathBuf, random::random};

use clap::Subcommand;
use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use ek_computation::state::{
        io::StateReaderImpl,
        models::{NewExpert, NewInstance, NewNode},
        writer::StateWriterImpl,
    };
use ek_db::{safetensor::ExpertKey, weight_srv::client::WeightSrvClient};
use indicatif::ProgressBar;
use log::info;
use serde::Deserialize;
use tokio::task::JoinSet;

#[derive(Subcommand, Debug)]
pub enum ScheduleCommand {
    Static {
        #[arg(long, short, help = "file that contains worker nodes information")]
        inventory: PathBuf,
    },
    Rebalance,
}

pub async fn execute_schedule(cmd: ScheduleCommand) -> EKResult<()> {
    match cmd {
        ScheduleCommand::Static { inventory } => {
            execute_static_schedule(inventory).await?;
            Ok(())
        }

        ScheduleCommand::Rebalance => {
            execute_rebalance().await?;
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub id: String,
    pub address: String,
    pub channel: String,
    pub device: String,
}
#[derive(Debug, Clone, Deserialize)]
pub struct Inventory {
    pub nodes: Vec<Node>,
}

async fn upsert_nodes(inventory: PathBuf) -> EKResult<Vec<i32>> {
    if !inventory.exists() {
        log::error!("inventory file not exists");
        return Err(EKError::NotFound("inventory file".to_string()));
    }

    let writer = StateWriterImpl::new();
    let contents = tokio::fs::read(inventory).await?;
    let inventory = serde_yaml::from_slice::<Inventory>(&contents).map_err(|e| {
        log::error!("failed to parse inventory file: {}", e);
        EKError::InvalidInput("inventory file".to_string())
    })?;
    let mut node_ids = vec![];
    for node in inventory.nodes {
        let new_node = NewNode {
            hostname: node.id.clone(),
            device: node.device.clone(),
            config: serde_json::json!({
                "addr": node.address,
                "channel": node.channel,
            }),
        };
        let node = writer.node_upsert(new_node).await?;
        node_ids.push(node.id);
    }
    Ok(node_ids)
}

async fn execute_static_schedule(inventory: PathBuf) -> EKResult<()> {
    // implement the static scheduling logic here
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings.weight.server.as_ref().unwrap().addr.clone();
    info!(
        "Running static schedule for model: {}, instance: {}, weight server: {}",
        model_name, instance_name, ws_addr
    );
    let cli = WeightSrvClient::new(ws_addr);
    let vital = cli.load_meta_vital(&model_name).await?;
    info!("model info : {:?}", &vital);

    let reader = StateReaderImpl::new();
    let model = reader
        .model_by_name(&model_name)
        .await?
        .ok_or(EKError::NotFound("model not found".to_string()))?;

    let writer = StateWriterImpl::new();
    let node_ids = upsert_nodes(inventory).await?;

    let instance_obj = writer
        .instance_upsert(NewInstance {
            model_id: model.id,
            name: instance_name,
        })
        .await?;

    let mut experts = vec![];
    for layer in vital.moe_layers.0..vital.moe_layers.1 {
        for expert in 0..vital.routed_experts {
            experts.push(ExpertKey::new(model_name.clone(), layer, expert));
        }
    }
    log::info!("total experts to schedule {}", experts.len());

    let pb = ProgressBar::new(experts.len() as u64);

    writer.expert_del_by_instance(instance_obj.id).await?;

    let mut js = JoinSet::new();
    for e in experts {
        let e = e.clone();
        let node_ids = node_ids.clone();
        let p = pb.clone();
        js.spawn(async move {
            let writer = StateWriterImpl::new();
            let rand = random::<u16>();
            writer
                .expert_upsert(NewExpert {
                    instance_id: instance_obj.id,
                    node_id: node_ids[(rand % node_ids.len() as u16) as usize],
                    expert_id: e.as_object_key(),
                    replica: 1,
                    state: serde_json::json!({}),
                })
                .await
                .unwrap();
            p.inc(1);
        });
    }
    js.join_all().await;
    pb.finish();
    log::info!("all experts scheduled");

    Ok(())
}

async fn execute_rebalance() -> EKResult<()> {
    // implement the static scheduling logic here
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings.weight.server.as_ref().unwrap().addr.clone();
    info!(
        "Running static schedule for model: {}, instance: {}, weight server: {}",
        model_name, instance_name, ws_addr
    );
    let cli = WeightSrvClient::new(ws_addr);
    let vital = cli.load_meta_vital(&model_name).await?;
    info!("model info : {:?}", &vital);

    let reader = StateReaderImpl::new();
    let model = reader
        .model_by_name(&model_name)
        .await?
        .ok_or(EKError::NotFound("model not found".to_string()))?;

    let writer = StateWriterImpl::new();
    let node_ids = reader
        .active_nodes()
        .await?
        .into_iter()
        .map(|x| x.id)
        .collect::<Vec<_>>();

    let instance_obj = writer
        .instance_upsert(NewInstance {
            model_id: model.id,
            name: instance_name,
        })
        .await?;

    let mut experts = vec![];
    for layer in vital.moe_layers.0..vital.moe_layers.1 {
        for expert in 0..vital.routed_experts {
            experts.push(ExpertKey::new(model_name.clone(), layer, expert));
        }
    }
    log::info!("total experts to schedule {}", experts.len());

    let pb = ProgressBar::new(experts.len() as u64);

    writer.expert_del_by_instance(instance_obj.id).await?;

    let mut js = JoinSet::new();
    for e in experts {
        let e = e.clone();
        let node_ids = node_ids.clone();
        let p = pb.clone();
        js.spawn(async move {
            let writer = StateWriterImpl::new();
            let rand = random::<u16>();
            writer
                .expert_upsert(NewExpert {
                    instance_id: instance_obj.id,
                    node_id: node_ids[(rand % node_ids.len() as u16) as usize],
                    expert_id: e.as_object_key(),
                    replica: 1,
                    state: serde_json::json!({}),
                })
                .await
                .unwrap();
            p.inc(1);
        });
    }
    js.join_all().await;
    pb.finish();
    log::info!("all experts scheduled");

    Ok(())
}
