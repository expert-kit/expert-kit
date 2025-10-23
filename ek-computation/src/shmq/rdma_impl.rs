use std::io;

// Import from the provided ibverbs library
use ibverbs::{
    CompletionQueue, Context, MemoryRegion, ProtectionDomain, QueuePair, QueuePairBuilder,
    QueuePairEndpoint, RemoteMemoryRegion, devices, ibv_qp_type,
};

use crate::shmq::GeneralShmQueueBytes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RdmaQueueError {
    Full,
    Empty,
    IoError,
}

impl std::error::Error for RdmaQueueError {}

impl std::fmt::Display for RdmaQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdmaQueueError::Full => write!(f, "Queue is full"),
            RdmaQueueError::Empty => write!(f, "Queue is empty"),
            RdmaQueueError::IoError => write!(f, "RDMA I/O error"),
        }
    }
}

impl From<RdmaQueueError> for io::Error {
    fn from(err: RdmaQueueError) -> Self {
        match err {
            RdmaQueueError::Full => io::Error::new(io::ErrorKind::WouldBlock, "Queue is full"),
            RdmaQueueError::Empty => io::Error::new(io::ErrorKind::WouldBlock, "Queue is empty"),
            RdmaQueueError::IoError => io::Error::other("RDMA I/O error"),
        }
    }
}

/// Metadata structure for the RDMA queue, stored at the beginning of the memory region
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct RdmaQueueMeta {
    capacity: usize,
    head: usize,
    tail: usize,
    data_offset: usize,
    ready: bool,
}

pub struct RdmaQueue<T> {
    // RDMA resources
    _context: Context,
    _pd: ProtectionDomain,
    _send_cq: CompletionQueue,
    _recv_cq: CompletionQueue,
    _qp_builder: QueuePairBuilder,
    _endpoint: QueuePairEndpoint,
    qp: Option<QueuePair>,

    // Local memory region containing queue metadata and data
    memory_region: MemoryRegion<Vec<u8>>,

    // Remote memory region (for accessing peer's queue)
    remote_region: Option<RemoteMemoryRegion>,

    // Queue configuration
    capacity: usize,
    is_sender: bool,

    // Work request ID counter
    wr_id_counter: u64,

    // Connection Status
    _is_connected: bool,

    _phantom: std::marker::PhantomData<T>,
}

// Helper function for offset calculation
fn offset_of_head() -> usize {
    let dummy = RdmaQueueMeta::default();
    let base_ptr = &dummy as *const _ as usize;
    let field_ptr = &dummy.head as *const _ as usize;
    field_ptr - base_ptr
}

fn offset_of_tail() -> usize {
    let dummy = RdmaQueueMeta::default();
    let base_ptr = &dummy as *const _ as usize;
    let field_ptr = &dummy.tail as *const _ as usize;
    field_ptr - base_ptr
}

impl<T: GeneralShmQueueBytes> RdmaQueue<T> {
    /// Check if the queue is connected to a remote peer
    pub fn is_connected(&self) -> bool {
        self._is_connected
    }

    pub fn new(device_index: Option<usize>, capacity: usize, is_sender: bool) -> io::Result<Self> {
        // Get RDMA devices
        let devices = devices()?;
        if devices.is_empty() {
            return Err(io::Error::other("No RDMA devices found"));
        }

        // Select device
        let device = devices
            .get(device_index.unwrap_or(0))
            .ok_or_else(|| io::Error::other("Invalid device index"))?;

        // Open device context
        let context = device.open()?;

        // Create protection domain
        let pd = context.alloc_pd()?;

        // Create completion queues with larger size for high throughput
        let send_cq = context.create_cq(256, 0)?;
        let recv_cq = context.create_cq(256, 1)?;

        // Calculate memory layout
        let meta_size = std::mem::size_of::<RdmaQueueMeta>();
        let data_offset = meta_size.next_multiple_of(T::aligned_size());
        let total_size = data_offset + capacity * T::aligned_size();

        // Allocate memory region
        let mut memory_region = pd.allocate(total_size)?;

        // Initialize metadata (both sender and receiver need initialized metadata)
        let meta = RdmaQueueMeta {
            capacity,
            head: 0,
            tail: 0,
            data_offset,
            ready: true,
        };

        // Write metadata to the beginning of the memory region
        let meta_bytes =
            unsafe { std::slice::from_raw_parts(&meta as *const _ as *const u8, meta_size) };
        memory_region.inner()[..meta_size].copy_from_slice(meta_bytes);

        // Create queue pair
        let mut qp_builder = pd.create_qp(&send_cq, &recv_cq, ibv_qp_type::IBV_QPT_RC)?;

        qp_builder
            .set_gid_index(0)
            .set_max_send_wr(256)
            .set_max_recv_wr(256)
            .set_max_send_sge(1)
            .set_max_recv_sge(1)
            .allow_remote_rw();

        let prepared_qp = qp_builder.build()?;
        let endpoint = prepared_qp.endpoint()?;

        Ok(Self {
            _context: context,
            _pd: pd,
            _send_cq: send_cq,
            _recv_cq: recv_cq,
            _qp_builder: qp_builder,
            _endpoint: endpoint,
            qp: None,
            memory_region,
            remote_region: None,
            capacity,
            is_sender,
            wr_id_counter: 1,
            _is_connected: false,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the local endpoint information for connection establishment
    pub fn endpoint(&self) -> io::Result<QueuePairEndpoint> {
        Ok(self._endpoint.clone())
    }

    /// Get the local memory region information for sharing with remote peer
    pub fn memory_region(&self) -> RemoteMemoryRegion {
        self.memory_region.remote()
    }

    /// Connect to remote peer
    pub fn connect(
        &mut self,
        remote_endpoint: QueuePairEndpoint,
        remote_region: RemoteMemoryRegion,
    ) -> io::Result<()> {
        // Store remote memory region
        self.remote_region = Some(remote_region);

        // Complete the QP handshake
        if !self.is_connected() {
            let prepared_qp = self._qp_builder.build()?;

            let result = prepared_qp.handshake(remote_endpoint);
            let qp = result.map_err(|e| io::Error::other(format!("QP handshake failed: {}", e)))?;

            // Mark as connected
            self.qp = Some(qp);
            self._is_connected = true;
            log::info!("🚀Connected to remote peer: {:?}", remote_endpoint);
            Ok(())
        } else {
            Err(io::Error::other(
                "Queue pair already connected or not prepared",
            ))
        }
    }

    /// Send an item to the queue (sender side) - Push-based architecture
    pub fn send(&mut self, item: &T) -> Result<(), RdmaQueueError> {
        if !self.is_sender || !self.is_connected() {
            return Err(RdmaQueueError::IoError);
        }
        let start_time = std::time::Instant::now();
        let mut t = std::time::Instant::now();
        let remote_region = self.remote_region.clone().ok_or(RdmaQueueError::IoError)?;
        log::debug!("🚀 1. region clone: {:?}", t.elapsed());

        // Read current metadata from remote memory (receiver's queue)
        t = std::time::Instant::now();
        let meta = self.read_remote_meta(&remote_region)?;
        log::debug!("🚀 2. read remote meta: {:?}", t.elapsed());

        // Check if metadata is valid and queue is full
        if meta.capacity == 0 || !meta.ready {
            return Err(RdmaQueueError::IoError);
        }
        if (meta.tail + 1) % meta.capacity == meta.head {
            return Err(RdmaQueueError::Full);
        }

        // Write item data directly to remote memory
        t = std::time::Instant::now();
        log::debug!("🚀 3. start write remote data");
        let item_offset = meta.data_offset + meta.tail * T::aligned_size();
        self.write_remote_data(&remote_region, item_offset, item)?;
        log::debug!("🚀 - 3 end write remote data: {:?}\n", t.elapsed());

        // Update tail pointer in remote memory atomically
        t = std::time::Instant::now();
        log::debug!("🚀 4. start update remote tail");
        let new_tail = (meta.tail + 1) % meta.capacity;
        self.update_remote_tail(&remote_region, new_tail)?;
        log::debug!("🚀 - 4 end update remote tail: {:?}\n", t.elapsed());

        log::debug!("🚀 Total send time: {:?}\n", start_time.elapsed());
        Ok(())
    }

    /// Receive an item from the queue (receiver side) - Poll local memory
    pub fn recv(&mut self) -> Result<T, RdmaQueueError> {
        if self.is_sender || !self.is_connected() {
            return Err(RdmaQueueError::IoError);
        }

        // Read metadata from local memory (no RDMA needed!)
        let meta = self.read_local_meta();

        // Check if queue is empty
        if meta.head == meta.tail {
            return Err(RdmaQueueError::Empty);
        }

        // Read item data from local memory
        let item_offset = meta.data_offset + meta.head * T::aligned_size();
        let item_data = &self.memory_region.inner()[item_offset..item_offset + T::CAPACITY];
        let item = T::from_bytes(item_data);

        // Update head pointer in local memory
        let new_head = (meta.head + 1) % meta.capacity;
        self.update_local_head(new_head)?;

        Ok(item)
    }

    /// Read metadata from local memory
    fn read_local_meta(&mut self) -> RdmaQueueMeta {
        let meta_size = std::mem::size_of::<RdmaQueueMeta>();
        let meta_bytes = &self.memory_region.inner()[..meta_size];
        unsafe { std::ptr::read(meta_bytes.as_ptr() as *const RdmaQueueMeta) }
    }

    /// Update head pointer in local memory
    fn update_local_head(&mut self, new_head: usize) -> Result<(), RdmaQueueError> {
        let head_offset = offset_of_head();
        let head_bytes = new_head.to_le_bytes();
        let memory = self.memory_region.inner();
        memory[head_offset..head_offset + std::mem::size_of::<usize>()]
            .copy_from_slice(&head_bytes);
        Ok(())
    }

    /// Write data directly to remote memory using RDMA write
    fn write_remote_data(
        &mut self,
        remote_region: &RemoteMemoryRegion,
        offset: usize,
        item: &T,
    ) -> Result<(), RdmaQueueError> {
        // Use a temporary buffer in memory region for writing
        let write_offset = std::mem::size_of::<RdmaQueueMeta>().next_multiple_of(64);

        // Write item directly to local buffer (zero-copy)
        let mut t = std::time::Instant::now();
        item.write_to_slice(
            &mut self.memory_region.inner()[write_offset..write_offset + T::CAPACITY],
        );
        log::debug!("🚀 3.1 write to local buffer: {:?}", t.elapsed());

        t = std::time::Instant::now();
        let local_slice = self
            .memory_region
            .slice(write_offset..write_offset + item.len());
        let remote_slice = remote_region.slice(offset..offset + item.len());
        log::debug!("🚀 3.2 prepare slices: {:?}", t.elapsed());

        // Issue RDMA write
        t = std::time::Instant::now();
        let wr_id = self.next_wr_id();
        let qp = self.qp.as_mut().ok_or(RdmaQueueError::IoError)?;
        qp.post_write(&[local_slice], remote_slice, wr_id, None)
            .map_err(|_| RdmaQueueError::IoError)?;
        log::debug!("🚀 3.3 post write: {:?}", t.elapsed());

        // Wait for completion
        t = std::time::Instant::now();
        self.wait_for_completion(wr_id)?;
        log::debug!("🚀 3.4 wait for completion: {:?}", t.elapsed());

        Ok(())
    }

    /// Update tail pointer in remote memory using RDMA write
    fn update_remote_tail(
        &mut self,
        remote_region: &RemoteMemoryRegion,
        new_tail: usize,
    ) -> Result<(), RdmaQueueError> {
        let tail_offset = offset_of_tail();
        let tail_size = std::mem::size_of::<usize>();

        // Prepare local data for writing
        let mut t = std::time::Instant::now();
        let write_offset =
            std::mem::size_of::<RdmaQueueMeta>().next_multiple_of(64) + T::aligned_size();
        let tail_bytes = new_tail.to_le_bytes();
        log::debug!("🚀 4.1 prepare tail bytes: {:?}", t.elapsed());

        t = std::time::Instant::now();
        self.memory_region.inner()[write_offset..write_offset + tail_size]
            .copy_from_slice(&tail_bytes);
        log::debug!("🚀 4.2 write to local buffer: {:?}", t.elapsed());

        t = std::time::Instant::now();
        let local_slice = self
            .memory_region
            .slice(write_offset..write_offset + tail_size);
        let remote_slice = remote_region.slice(tail_offset..tail_offset + tail_size);
        log::debug!("🚀 4.3 prepare slices: {:?}", t.elapsed());

        // Issue RDMA write
        t = std::time::Instant::now();
        let wr_id = self.next_wr_id();
        let qp = self.qp.as_mut().ok_or(RdmaQueueError::IoError)?;
        qp.post_write(&[local_slice], remote_slice, wr_id, None)
            .map_err(|_| RdmaQueueError::IoError)?;
        log::debug!("🚀 4.4 post write: {:?}", t.elapsed());

        // Wait for completion
        t = std::time::Instant::now();
        self.wait_for_completion(wr_id)?;
        log::debug!("🚀 4.5 wait for completion: {:?}", t.elapsed());

        Ok(())
    }

    /// Read metadata from remote memory using RDMA read
    fn read_remote_meta(
        &mut self,
        remote_region: &RemoteMemoryRegion,
    ) -> Result<RdmaQueueMeta, RdmaQueueError> {
        let meta_size = std::mem::size_of::<RdmaQueueMeta>();

        // Prepare local buffer for reading metadata
        let local_slice = self.memory_region.slice(0..meta_size);
        let remote_slice = remote_region.slice(0..meta_size);

        // Issue RDMA read
        let wr_id = self.next_wr_id();
        let qp = self.qp.as_mut().ok_or(RdmaQueueError::IoError)?;
        qp.post_read(&[local_slice], remote_slice, wr_id)
            .map_err(|_| RdmaQueueError::IoError)?;

        // Wait for completion
        self.wait_for_completion(wr_id)?;

        // Parse metadata
        let meta_bytes = &self.memory_region.inner()[0..meta_size];
        let meta = unsafe { std::ptr::read(meta_bytes.as_ptr() as *const RdmaQueueMeta) };

        Ok(meta)
    }

    /// Wait for a specific work request to complete with non-blocking retry
    fn wait_for_completion(&mut self, expected_wr_id: u64) -> Result<(), RdmaQueueError> {
        let mut completions = [Default::default(); 4]; // Handle multiple completions

        loop {
            // First try non-blocking poll
            match self._send_cq.wait(&mut completions, None) {
                Ok(completed) => {
                    if !completed.is_empty() {
                        for completion in completed {
                            if completion.wr_id() == expected_wr_id {
                                // Check if the completion was successful
                                // TODO: We'll need to check the actual API for proper status checking
                                return Ok(());
                            }
                        }
                    }
                }
                Err(_) => return Err(RdmaQueueError::IoError),
            }
        }
    }

    /// Get next work request ID
    fn next_wr_id(&mut self) -> u64 {
        let id = self.wr_id_counter;
        self.wr_id_counter += 1;
        id
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Try to send without blocking (returns immediately if would block)
    pub fn try_send(&mut self, item: &T) -> Result<(), RdmaQueueError> {
        // Same as send but could be optimized to not block on completion
        self.send(item)
    }

    /// Try to receive without blocking (returns immediately if empty)
    pub fn try_recv(&mut self) -> Result<T, RdmaQueueError> {
        // This is already non-blocking in the new architecture
        self.recv()
    }

    pub fn close(&mut self) {
        // Clean up RDMA resources in reverse order of creation
        // This is critical to avoid "Device or resource busy" errors

        if let Some(qp) = self.qp.take() {
            drop(qp);
        }
    }
}

impl<T> Drop for RdmaQueue<T> {
    fn drop(&mut self) {
        if let Some(qp) = self.qp.take() {
            drop(qp);
        }
    }
}

impl GeneralShmQueueBytes for u64 {
    const CAPACITY: usize = std::mem::size_of::<u64>();

    fn write_to_slice(&self, slice: &mut [u8]) {
        slice[..Self::CAPACITY].copy_from_slice(&self.to_le_bytes());
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        u64::from_le_bytes(bytes[..Self::CAPACITY].try_into().unwrap())
    }

    fn len(&self) -> usize {
        std::mem::size_of::<u64>()
    }
}

impl GeneralShmQueueBytes for String {
    const CAPACITY: usize = 256; // Fixed size for simplicity

    fn write_to_slice(&self, slice: &mut [u8]) {
        slice.fill(0); // Zero out the slice first
        let bytes = self.as_bytes();
        let len = bytes.len().min(Self::CAPACITY - 1);
        slice[..len].copy_from_slice(&bytes[..len]);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..end]).into_owned()
    }

    fn len(&self) -> usize {
        self.len()
    }
}
