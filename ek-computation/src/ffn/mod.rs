pub mod expert_ort;
pub mod expert_torch;

pub struct ExpertShape {
    pub dim: usize,
    pub hidden: usize,
}
pub trait Expert<T> {
    fn backend(&self) -> std::string::String;
    fn shape(&self) -> ExpertShape;
    fn rand_input(&self, batch: usize) -> T;
    fn forward(&self, x: &T) -> T;
}

#[derive(Clone, Copy)]
pub enum DType {
    Uint8,
    Int8,
    Int16,
    BFloat16,
    Float,
    Float8e4m3fn,
    Float8e4m3fnuz,
}

#[derive(Clone, Copy)]
pub enum Device {
    CPU,
}

pub trait EkTensor: Sized {
    fn rand(shape: Vec<usize>, dtype: DType, dev: Device) -> Self;
    fn cat(tensors: &[Self], dim: usize) -> Self;
    fn serialize(&self) -> Vec<u8>;
}

pub struct ExpertWeight<T>
where
    T: EkTensor,
{
    pub up_w: T,
    pub up_b: Option<T>,
    pub down_w: T,
    pub down_b: Option<T>,
    pub gate_w: T,
    pub gate_b: Option<T>,
}
impl<T: EkTensor> ExpertWeight<T> {
    pub fn rand(dim: usize, hidden: usize, dtype: DType, dev: Device) -> Self {
        return Self {
            down_w: T::rand(vec![dim, hidden], dtype, dev),
            down_b: None,
            up_w: T::rand(vec![hidden, dim], dtype, dev),
            up_b: None,
            gate_w: T::rand(vec![hidden, dim], dtype, dev),
            gate_b: None,
        };
    }
}
