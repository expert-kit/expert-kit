pub mod rdma_impl;
pub mod rdma_tcp_exchange;
pub mod shared_mem;

pub use rdma_impl::{RdmaQueue, RdmaQueueError};
pub use rdma_tcp_exchange::{RdmaEndpointServer, RdmaEndpointClient, RdmaConnectionInfo, RdmaConnectionPair};
pub use shared_mem::ShmQueue;

/// Trait for types that can be sent over RDMA queue
#[expect(clippy::len_without_is_empty)]
pub trait GeneralShmQueueBytes {
    const CAPACITY: usize;

    fn write_to_slice(&self, slice: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Real len of the structure
    /// This is the actual length of the data, excluding any padding
    fn len(&self) -> usize;

    #[inline]
    fn aligned_size() -> usize {
        Self::CAPACITY.next_multiple_of(64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralShmQueueError {
    Full,
    Empty,
}

impl std::error::Error for GeneralShmQueueError {}

impl std::fmt::Display for GeneralShmQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneralShmQueueError::Full => write!(f, "Queue is full"),
            GeneralShmQueueError::Empty => write!(f, "Queue is empty"),
        }
    }
}
