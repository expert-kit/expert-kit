use std::ptr;

use ndarray::ArrayD;

pub trait Expert<T> {
    fn forward(&self, x: T) -> T;
}
