pub mod rdma_impl;
pub mod shared_mem;

pub use rdma_impl::{RdmaBytes, RdmaQueue, RdmaQueueError};
pub use shared_mem::{ShmBytes, ShmQueue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShmQueueError {
    Full,
    Empty,
}

impl std::error::Error for ShmQueueError {}

impl std::fmt::Display for ShmQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmQueueError::Full => write!(f, "Queue is full"),
            ShmQueueError::Empty => write!(f, "Queue is empty"),
        }
    }
}
