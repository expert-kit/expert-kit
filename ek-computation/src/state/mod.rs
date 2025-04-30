use ek_base::error::EKResult;
use tonic::async_trait;

pub struct Node {
    pub id: i32,
    pub hostname: String,
    pub device: String,
    pub config: serde_json::Value,
}

pub struct Expert {
    pub id: i32,
    pub instance_id: i32,
    pub node_id: i32,
    pub expert_id: String,
    pub replica: i32,
    pub state: String,
}

pub struct Instance {
    pub id: i32,
    pub model_id: i32,
    pub name: String,
}

pub struct Model {
    pub id: i32,
    pub name: String,
    pub config: serde_json::Value,
}

#[async_trait]
trait StateReader {
    async fn node_by_hostname(hostname: String) -> EKResult<Option<Node>>;
    async fn instance_by_id(id: i32) -> EKResult<Option<Instance>>;
    async fn experts_by_node(node_id: i32) -> EKResult<Vec<Expert>>;
}

#[async_trait]
trait StateWriter {
    async fn add_instance(instance: &Instance) -> EKResult<()>;
    async fn add_model(instance: &Model) -> EKResult<()>;
    async fn add_expert(instance: &Expert) -> EKResult<()>;
    async fn add_node(instance: &Node) -> EKResult<()>;

    async fn del_instance(id: i32) -> EKResult<()>;
    async fn del_model(id: i32) -> EKResult<()>;
    async fn del_expert(id: i32) -> EKResult<()>;
    async fn del_node(id: i32) -> EKResult<()>;

    async fn upd_expert_state(distribution_id: i32, state: String) -> EKResult<()>;
}
