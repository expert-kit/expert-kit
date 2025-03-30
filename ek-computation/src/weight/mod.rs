use async_trait::async_trait;

pub struct TorchTensorWeight {
    pub tensor: tch::Tensor,
}
pub enum Weight {
    TorchTensor(TorchTensorWeight),
}

#[async_trait]
pub trait WeightDAL {
    async fn load_weight(&self, path: &str) -> Result<Weight, String>;
}
