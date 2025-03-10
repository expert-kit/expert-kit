

pub struct ExpertShape {
    pub dim: usize,
    pub hidden: usize,
}
pub trait Expert<T> {
    fn backend(&self) -> std::string::String;
    fn shape(&self) -> ExpertShape;
    fn rand_input(&self, batch: usize) -> T;
    fn forward(&self, x: T) -> T;
}
