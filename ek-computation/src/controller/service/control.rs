use std::random::random;

use ek_base::{
    config::get_ek_settings,
    error::{EKError, EKResult},
};
use ek_db::{safetensor::ExpertKey, weight_srv::client::WeightSrvClient};
use tokio::task::JoinSet;

use crate::{
    controller::registry::get_registry,
    proto::ek::control::v1::{self},
    state::{
        io::StateReaderImpl,
        models::{NewExpert, NewInstance},
        writer::StateWriterImpl,
    },
};
pub struct PlanServiceImpl {}

impl Default for PlanServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanServiceImpl {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl v1::plan_service_server::PlanService for PlanServiceImpl {
    async fn rebalance(
        &self,
        _request: tonic::Request<v1::RebalanceReq>,
    ) -> Result<tonic::Response<v1::RebalanceResp>, tonic::Status> {
        execute_rebalance().await?;
        let registry = get_registry();
        registry.lock().await.reset().await?;
        let resp = v1::RebalanceResp {};
        Ok(tonic::Response::new(resp))
    }
}

async fn execute_rebalance() -> EKResult<()> {
    // implement the static scheduling logic here
    let settings = get_ek_settings();
    let model_name = settings.inference.model_name.clone();
    let instance_name = settings.inference.instance_name.clone();
    let ws_addr = settings.weight.server.as_ref().unwrap().addr.clone();
    log::info!(
        "Running static schedule for model: {}, instance: {}, weight server: {}",
        model_name,
        instance_name,
        ws_addr
    );
    let cli = WeightSrvClient::new(ws_addr);
    let vital = cli.load_meta_vital(&model_name).await?;
    log::info!("model info : {:?}", &vital);

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

    writer.expert_del_by_instance(instance_obj.id).await?;

    let mut js = JoinSet::new();
    for e in experts {
        let e = e.clone();
        let node_ids = node_ids.clone();
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
        });
    }
    js.join_all().await;
    log::info!("all experts scheduled");

    Ok(())
}
