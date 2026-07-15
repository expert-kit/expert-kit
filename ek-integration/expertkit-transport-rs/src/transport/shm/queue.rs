use anyhow::Result;
use nix::{
    fcntl::OFlag,
    sys::{
        mman::{self, MapFlags, ProtFlags},
        stat::Mode,
    },
    unistd,
};
use std::ffi::c_void;
use std::num::NonZero;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Queue error types
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

/// Metadata structure at the beginning of shared memory
#[repr(C)]
struct ShmQueueMeta {
    capacity: usize,
    head: usize,
    tail: usize,
    data_offset: usize,
    ready: bool,
}

/// Shared memory circular queue
pub struct ShmQueue {
    owned: bool,
    name: String,
    mmap: (NonNull<c_void>, usize),
    meta: *mut ShmQueueMeta,
    data: *mut u8,
    #[allow(unused)]
    capacity: usize,
    slot_size: usize,
}

impl ShmQueue {
    /// Create a new shared memory queue
    #[allow(unused)]
    pub fn new(name: &str, capacity: usize, slot_size: usize) -> Result<Self> {
        // Align slot size to 64 bytes
        let slot_size = slot_size.div_ceil(64) * 64;

        // Calculate layout
        let meta_size = std::mem::size_of::<ShmQueueMeta>();
        let meta_slots = meta_size.div_ceil(slot_size);
        let data_offset = 1 + meta_slots;
        let total_slots = capacity + 1 + meta_slots;
        let total_size = total_slots * slot_size;

        // Create shared memory
        let shm = mman::shm_open(
            name,
            OFlag::O_CREAT | OFlag::O_RDWR | OFlag::O_EXCL,
            Mode::S_IRUSR | Mode::S_IWUSR,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create shm {}: {}", name, e))?;

        unistd::ftruncate(&shm, total_size as i64)
            .map_err(|e| anyhow::anyhow!("Failed to truncate: {}", e))?;

        let mmap = unsafe {
            mman::mmap(
                None,
                NonZero::new(total_size).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &shm,
                0,
            )
            .map_err(|e| anyhow::anyhow!("Failed to mmap: {}", e))?
            .cast::<u8>()
        };

        // Initialize metadata
        let meta = unsafe { &mut *(mmap.as_ptr() as *mut ShmQueueMeta) };
        meta.capacity = capacity;
        meta.head = 0;
        meta.tail = 0;
        meta.data_offset = data_offset;
        meta.ready = true;

        let data = unsafe { mmap.as_ptr().add(data_offset * slot_size) };

        Ok(Self {
            owned: true,
            name: name.to_string(),
            mmap: (mmap.cast(), total_size),
            meta: meta as *mut ShmQueueMeta,
            data,
            capacity,
            slot_size,
        })
    }

    /// Open an existing shared memory queue
    pub fn open(name: &str, _capacity: usize, slot_size: usize) -> Option<Self> {
        // Align slot size to 64 bytes
        let slot_size = slot_size.div_ceil(64) * 64;

        // Open shared memory
        let shm = mman::shm_open(name, OFlag::O_RDWR, Mode::S_IRUSR | Mode::S_IWUSR).ok()?;

        // Get size
        use std::os::fd::AsRawFd;
        let stat = nix::sys::stat::fstat(shm.as_raw_fd()).ok()?;
        let total_size = stat.st_size as usize;

        if total_size < std::mem::size_of::<ShmQueueMeta>() {
            return None;
        }

        // Map memory
        let mmap = unsafe {
            mman::mmap(
                None,
                NonZero::new(total_size).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &shm,
                0,
            )
            .ok()?
            .cast::<u8>()
        };

        // Read metadata
        let meta = unsafe { &mut *(mmap.as_ptr() as *mut ShmQueueMeta) };

        // Wait for ready flag
        while !unsafe { std::ptr::read_volatile(&meta.ready) } {
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        let data_offset = meta.data_offset;
        let capacity = meta.capacity;
        let data = unsafe { mmap.as_ptr().add(data_offset * slot_size) };

        Some(Self {
            owned: false,
            name: name.to_string(),
            mmap: (mmap.cast(), total_size),
            meta: meta as *mut ShmQueueMeta,
            data,
            capacity,
            slot_size,
        })
    }

    /// Send data to the queue
    pub fn send(&mut self, item: &impl ShmQueueItem) -> Result<(), ShmQueueError> {
        let meta = unsafe { std::ptr::read_volatile(self.meta) };

        // Check if full
        if meta.head == (meta.tail + 1) % meta.capacity {
            return Err(ShmQueueError::Full);
        }

        // Write data to slot
        let slot_offset = meta.tail * self.slot_size;
        let slot =
            unsafe { std::slice::from_raw_parts_mut(self.data.add(slot_offset), self.slot_size) };
        item.write_to_slice(slot);

        // Update tail
        unsafe {
            (*self.meta).tail = (meta.tail + 1) % meta.capacity;
        }

        Ok(())
    }

    /// Receive data from the queue
    pub fn recv<T: ShmQueueItem>(&mut self) -> Result<T, ShmQueueError> {
        let meta = unsafe { std::ptr::read_volatile(self.meta) };

        // Check if empty
        if meta.head == meta.tail {
            return Err(ShmQueueError::Empty);
        }

        // Read data from slot
        let slot_offset = meta.head * self.slot_size;
        let slot =
            unsafe { std::slice::from_raw_parts(self.data.add(slot_offset), self.slot_size) };
        let item = T::from_bytes(slot);

        // Update head
        unsafe {
            (*self.meta).head = (meta.head + 1) % meta.capacity;
        }

        Ok(item)
    }
}

impl Drop for ShmQueue {
    fn drop(&mut self) {
        let (addr, len) = self.mmap;
        unsafe {
            let _ = mman::munmap(addr, len);
        }

        if self.owned {
            let _ = mman::shm_unlink(self.name.as_str());
        }
    }
}

unsafe impl Send for ShmQueue {}

/// Trait for items that can be sent/received through shared memory queue
pub trait ShmQueueItem {
    fn write_to_slice(&self, slice: &mut [u8]);
    fn from_bytes(bytes: &[u8]) -> Self;
}

/// Worker request structure
#[derive(Debug, Clone)]
pub struct ShmqWorkerReq {
    pub id: usize,
    pub expert_id: String,
    pub traceparent: String,
    pub input_tensor: Vec<u8>,
}

static REQ_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

impl ShmqWorkerReq {
    pub fn new(expert_id: &str, input_tensor: &[u8], traceparent: &str) -> Self {
        let id = REQ_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            id,
            expert_id: expert_id.to_string(),
            traceparent: traceparent.to_string(),
            input_tensor: input_tensor.to_vec(),
        }
    }
}

impl ShmQueueItem for ShmqWorkerReq {
    fn write_to_slice(&self, slice: &mut [u8]) {
        // Layout: id (8) + expert_id (64) + traceparent (128) + tensor_len (8) + tensor_data
        let id_bytes = self.id.to_le_bytes();
        slice[0..8].copy_from_slice(&id_bytes);

        // Expert ID as null-terminated 64-byte array
        let expert_id_bytes = self.expert_id.as_bytes();
        let expert_id_len = expert_id_bytes.len().min(63);
        slice[8..8 + expert_id_len].copy_from_slice(&expert_id_bytes[..expert_id_len]);
        slice[8 + expert_id_len..72].fill(0);

        // Traceparent as null-terminated 128-byte array
        let traceparent_bytes = self.traceparent.as_bytes();
        let traceparent_len = traceparent_bytes.len().min(127);
        slice[72..72 + traceparent_len].copy_from_slice(&traceparent_bytes[..traceparent_len]);
        slice[72 + traceparent_len..200].fill(0);

        // Tensor length
        let tensor_len_bytes = self.input_tensor.len().to_le_bytes();
        slice[200..208].copy_from_slice(&tensor_len_bytes);

        // Tensor data
        slice[208..208 + self.input_tensor.len()].copy_from_slice(&self.input_tensor);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        // Parse id
        let id = usize::from_le_bytes(bytes[0..8].try_into().unwrap());

        // Parse expert_id (null-terminated)
        let expert_id_bytes = &bytes[8..72];
        let null_pos = expert_id_bytes.iter().position(|&b| b == 0).unwrap_or(64);
        let expert_id = String::from_utf8_lossy(&expert_id_bytes[..null_pos]).to_string();

        // Parse traceparent (null-terminated)
        let traceparent_bytes = &bytes[72..200];
        let traceparent_null = traceparent_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(128);
        let traceparent =
            String::from_utf8_lossy(&traceparent_bytes[..traceparent_null]).to_string();

        // Parse tensor
        let tensor_len = usize::from_le_bytes(bytes[200..208].try_into().unwrap());
        let input_tensor = bytes[208..208 + tensor_len].to_vec();

        Self {
            id,
            expert_id,
            traceparent,
            input_tensor,
        }
    }
}

/// Worker response structure
#[derive(Debug, Clone)]
pub struct ShmqWorkerResp {
    pub id: usize,
    pub output_tensor: Vec<u8>,
}

impl ShmqWorkerResp {
    #[allow(unused)]
    pub fn new(id: usize, output_tensor: Vec<u8>) -> Self {
        Self { id, output_tensor }
    }
}

impl ShmQueueItem for ShmqWorkerResp {
    fn write_to_slice(&self, slice: &mut [u8]) {
        // Layout: id (8) + tensor_len (8) + tensor_data
        let id_bytes = self.id.to_le_bytes();
        slice[0..8].copy_from_slice(&id_bytes);

        let tensor_len_bytes = self.output_tensor.len().to_le_bytes();
        slice[8..16].copy_from_slice(&tensor_len_bytes);

        slice[16..16 + self.output_tensor.len()].copy_from_slice(&self.output_tensor);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let id = usize::from_le_bytes(bytes[0..8].try_into().unwrap());
        let tensor_len = usize::from_le_bytes(bytes[8..16].try_into().unwrap());
        let output_tensor = bytes[16..16 + tensor_len].to_vec();

        Self { id, output_tensor }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_basic() {
        let mut queue = ShmQueue::new("test_basic", 10, 1024).unwrap();

        let req1 = ShmqWorkerReq::new("expert_1", b"test_data_1", "");
        let req2 = ShmqWorkerReq::new("expert_2", b"test_data_2", "");

        queue.send(&req1).unwrap();
        queue.send(&req2).unwrap();

        let recv1: ShmqWorkerReq = queue.recv().unwrap();
        let recv2: ShmqWorkerReq = queue.recv().unwrap();

        assert_eq!(recv1.expert_id, "expert_1");
        assert_eq!(recv1.input_tensor, b"test_data_1");
        assert_eq!(recv2.expert_id, "expert_2");
        assert_eq!(recv2.input_tensor, b"test_data_2");

        assert!(matches!(
            queue.recv::<ShmqWorkerReq>(),
            Err(ShmQueueError::Empty)
        ));
    }

    #[test]
    fn test_queue_open() {
        let mut sender = ShmQueue::new("test_open", 10, 1024).unwrap();
        let mut receiver = ShmQueue::open("test_open", 10, 1024).unwrap();

        let req = ShmqWorkerReq::new("expert_test", b"shared_data", "");
        sender.send(&req).unwrap();

        let recv: ShmqWorkerReq = receiver.recv().unwrap();
        assert_eq!(recv.expert_id, "expert_test");
        assert_eq!(recv.input_tensor, b"shared_data");
    }
}
