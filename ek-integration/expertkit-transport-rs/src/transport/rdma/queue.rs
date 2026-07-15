use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

// Import from the ibverbs library (InfiniBand verbs)
use ibverbs::{
    CompletionQueue, Context, MemoryRegion, PreparedQueuePair, ProtectionDomain, QueuePair,
    QueuePairEndpoint, RemoteMemoryRegion, devices, ibv_qp_type,
};

/// Maximum tensor size (64 MB) - must match worker expectations
const MAX_TENSOR_SIZE: usize = 64 * 1024 * 1024;
const TRACEPARENT_SIZE: usize = 128;

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
    head: usize,        // Consumer pointer (receiver updates)
    tail: usize,        // Producer pointer (sender updates via RDMA)
    data_offset: usize, // Offset where data slots begin
    ready: bool,        // Initialization flag
}

/// Trait for types that can be sent over RDMA queue
/// Must be able to serialize to/from byte slices
pub trait GeneralShmQueueBytes {
    /// Maximum capacity in bytes (includes padding)
    const CAPACITY: usize;

    /// Serialize self into the given byte slice
    fn write_to_slice(&self, slice: &mut [u8]);

    /// Deserialize from byte slice
    fn from_bytes(bytes: &[u8]) -> Self;

    /// Actual length of the data (excluding padding)
    fn len(&self) -> usize;

    /// Get aligned size (cache-line aligned to 64 bytes)
    #[inline]
    fn aligned_size() -> usize {
        Self::CAPACITY.next_multiple_of(64)
    }
}

/// RDMA queue for high-performance remote memory access
/// Uses InfiniBand one-sided RDMA writes for zero-copy data transfer
pub struct RdmaQueue<T> {
    // RDMA resources
    _context: Context,                      // RDMA device context
    _pd: ProtectionDomain,                  // Protection domain for memory isolation
    _recv_cq: CompletionQueue,              // Receive completion queue (unused in one-sided RDMA)
    send_cq: CompletionQueue,               // Send completion queue
    qp: Option<QueuePair>,                  // Connected queue pair (None until connect() called)
    prepared_qp: Option<PreparedQueuePair>, // Pre-handshake queue pair
    endpoint: QueuePairEndpoint,            // Local endpoint info for connection exchange

    // Local memory region containing queue metadata and data slots
    memory_region: MemoryRegion<Vec<u8>>,

    // Remote memory region (peer's memory) for RDMA operations
    remote_region: Option<RemoteMemoryRegion>,

    // Queue configuration
    #[allow(unused)]
    capacity: usize, // Number of slots in the queue
    is_sender: bool, // Role: true = sender (controller), false = receiver (worker)

    // Work request ID counter for tracking RDMA operations
    wr_id_counter: u64,

    _phantom: std::marker::PhantomData<T>,
}

// Helper functions for field offset calculation
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
        self.qp.is_some()
    }

    /// Create a new RDMA queue
    /// * `device_index` - RDMA device to use (None = auto-select first available)
    /// * `capacity` - Number of queue slots (typically 16 for transport-rs)
    /// * `is_sender` - true for sender (controller), false for receiver (worker)
    pub fn new(device_index: Option<usize>, capacity: usize, is_sender: bool) -> io::Result<Self> {
        // Get available RDMA devices
        let devices = devices()?;
        if devices.is_empty() {
            return Err(io::Error::other("No RDMA devices found"));
        }

        // Select device (auto-select first if not specified)
        let device = devices
            .get(device_index.unwrap_or(0))
            .ok_or_else(|| io::Error::other("Invalid device index"))?;

        // Open device context
        let context = device.open()?;

        // Create protection domain for memory isolation
        let pd = context.alloc_pd()?;

        // Create completion queues with 256 entries for high throughput
        let send_cq = context.create_cq(256, 0)?;
        let recv_cq = context.create_cq(256, 1)?;

        // Calculate memory layout:
        // [RdmaQueueMeta][padding][data slot 0][data slot 1]...[data slot N-1]
        let meta_size = std::mem::size_of::<RdmaQueueMeta>();
        let data_offset = meta_size.next_multiple_of(T::aligned_size());
        let total_size = data_offset + capacity * T::aligned_size();

        // Allocate and register memory region with RDMA NIC
        let mut memory_region = pd.allocate(total_size)?;

        // Initialize metadata in the memory region
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

        // Create queue pair (QP) - the connection endpoint for RDMA
        let mut qp_builder = pd.create_qp(&send_cq, &recv_cq, ibv_qp_type::IBV_QPT_RC)?;

        // Configure QP parameters
        qp_builder
            .set_gid_index(0)
            .set_max_send_wr(256)
            .set_max_recv_wr(256)
            .set_max_send_sge(1)
            .set_max_recv_sge(1)
            .allow_remote_rw();

        // Build prepared QP (not yet connected)
        let prepared_qp = qp_builder.build()?;
        let endpoint = prepared_qp.endpoint()?;

        Ok(Self {
            _context: context,
            _pd: pd,
            _recv_cq: recv_cq,
            send_cq,
            prepared_qp: Some(prepared_qp),
            qp: None,
            endpoint,
            memory_region,
            remote_region: None,
            capacity,
            is_sender,
            wr_id_counter: 1,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the local endpoint information for connection establishment
    pub fn endpoint(&self) -> io::Result<QueuePairEndpoint> {
        Ok(self.endpoint)
    }

    /// Get the local memory region information for sharing with remote peer
    pub fn memory_region(&self) -> RemoteMemoryRegion {
        self.memory_region.remote()
    }

    /// Connect to remote peer
    ///
    /// Completes the RDMA connection handshake using the provided remote endpoint
    /// and memory region information.
    pub fn connect(
        &mut self,
        remote_endpoint: QueuePairEndpoint,
        remote_region: RemoteMemoryRegion,
    ) -> io::Result<()> {
        // Store remote memory region for future RDMA operations
        self.remote_region = Some(remote_region);

        // Complete the QP handshake (transition QP to RTS state)
        if !self.is_connected() {
            let prepared_qp = self
                .prepared_qp
                .take()
                .ok_or_else(|| io::Error::other("No prepared QP available"))?;

            let result = prepared_qp.handshake(remote_endpoint);
            let qp = result.map_err(|e| io::Error::other(format!("QP handshake failed: {}", e)))?;

            // Mark as connected
            self.qp = Some(qp);
            log::info!("Connected to remote peer via RDMA");
            Ok(())
        } else {
            Err(io::Error::other(
                "Queue pair already connected or not prepared",
            ))
        }
    }

    /// Send an item to the queue (sender side) - Push-based RDMA write
    pub fn send(&mut self, item: &T) -> Result<(), RdmaQueueError> {
        if !self.is_sender || !self.is_connected() {
            return Err(RdmaQueueError::IoError);
        }

        let remote_region = self.remote_region.clone().ok_or(RdmaQueueError::IoError)?;

        // Step 1: Read current metadata from remote memory (receiver's queue)
        let meta = self.read_remote_meta(&remote_region)?;

        // Step 2: Validate metadata and check if queue is full
        if meta.capacity == 0 || !meta.ready {
            return Err(RdmaQueueError::IoError);
        }
        if (meta.tail + 1) % meta.capacity == meta.head {
            return Err(RdmaQueueError::Full);
        }

        // Step 3: Write item data directly to remote memory
        let item_offset = meta.data_offset + meta.tail * T::aligned_size();
        self.write_remote_data(&remote_region, item_offset, item)?;

        // Step 4: Update tail pointer in remote memory atomically
        let new_tail = (meta.tail + 1) % meta.capacity;
        self.update_remote_tail(&remote_region, new_tail)?;

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

    /// Read metadata from local memory (receiver side)
    fn read_local_meta(&mut self) -> RdmaQueueMeta {
        let meta_size = std::mem::size_of::<RdmaQueueMeta>();
        let meta_bytes = &self.memory_region.inner()[..meta_size];
        unsafe { std::ptr::read(meta_bytes.as_ptr() as *const RdmaQueueMeta) }
    }

    /// Update head pointer in local memory (receiver side)
    fn update_local_head(&mut self, new_head: usize) -> Result<(), RdmaQueueError> {
        let head_offset = offset_of_head();
        let head_bytes = new_head.to_le_bytes();
        let memory = self.memory_region.inner();
        memory[head_offset..head_offset + std::mem::size_of::<usize>()]
            .copy_from_slice(&head_bytes);
        Ok(())
    }

    /// Write data directly to remote memory using RDMA write (sender side)
    fn write_remote_data(
        &mut self,
        remote_region: &RemoteMemoryRegion,
        offset: usize,
        item: &T,
    ) -> Result<(), RdmaQueueError> {
        // Use a temporary buffer in our memory region for the RDMA write
        let write_offset = std::mem::size_of::<RdmaQueueMeta>().next_multiple_of(64);

        // Serialize item to our local buffer
        item.write_to_slice(
            &mut self.memory_region.inner()[write_offset..write_offset + T::CAPACITY],
        );

        // Prepare memory slices for RDMA operation
        let local_slice = self
            .memory_region
            .slice(write_offset..write_offset + item.len());
        let remote_slice = remote_region.slice(offset..offset + item.len());

        // Issue RDMA write operation
        let wr_id = self.next_wr_id();
        let qp = self.qp.as_mut().ok_or(RdmaQueueError::IoError)?;
        qp.post_write(&[local_slice], remote_slice, wr_id, None)
            .map_err(|_| RdmaQueueError::IoError)?;

        // Wait for completion
        self.wait_for_completion(wr_id)?;

        Ok(())
    }

    /// Update tail pointer in remote memory using RDMA write (sender side)
    fn update_remote_tail(
        &mut self,
        remote_region: &RemoteMemoryRegion,
        new_tail: usize,
    ) -> Result<(), RdmaQueueError> {
        let tail_offset = offset_of_tail();
        let tail_size = std::mem::size_of::<usize>();

        // Prepare local data for writing tail pointer
        let write_offset =
            std::mem::size_of::<RdmaQueueMeta>().next_multiple_of(64) + T::aligned_size();
        let tail_bytes = new_tail.to_le_bytes();
        self.memory_region.inner()[write_offset..write_offset + tail_size]
            .copy_from_slice(&tail_bytes);

        // Prepare memory slices for RDMA operation
        let local_slice = self
            .memory_region
            .slice(write_offset..write_offset + tail_size);
        let remote_slice = remote_region.slice(tail_offset..tail_offset + tail_size);

        // Issue RDMA write operation
        let wr_id = self.next_wr_id();
        let qp = self.qp.as_mut().ok_or(RdmaQueueError::IoError)?;
        qp.post_write(&[local_slice], remote_slice, wr_id, None)
            .map_err(|_| RdmaQueueError::IoError)?;

        // Wait for completion
        self.wait_for_completion(wr_id)?;

        Ok(())
    }

    /// Read metadata from remote memory using RDMA read (sender side)
    fn read_remote_meta(
        &mut self,
        remote_region: &RemoteMemoryRegion,
    ) -> Result<RdmaQueueMeta, RdmaQueueError> {
        let meta_size = std::mem::size_of::<RdmaQueueMeta>();

        // Prepare local buffer for reading metadata
        let local_slice = self.memory_region.slice(0..meta_size);
        let remote_slice = remote_region.slice(0..meta_size);

        // Issue RDMA read operation
        let wr_id = self.next_wr_id();
        let qp = self.qp.as_mut().ok_or(RdmaQueueError::IoError)?;
        qp.post_read(&[local_slice], remote_slice, wr_id)
            .map_err(|_| RdmaQueueError::IoError)?;

        // Wait for completion
        self.wait_for_completion(wr_id)?;

        // Parse metadata from our local buffer
        let meta_bytes = &self.memory_region.inner()[0..meta_size];
        let meta = unsafe { std::ptr::read(meta_bytes.as_ptr() as *const RdmaQueueMeta) };

        Ok(meta)
    }

    /// Wait for a specific work request to complete
    ///
    /// Polls the completion queue until the expected work request is completed.
    fn wait_for_completion(&mut self, expected_wr_id: u64) -> Result<(), RdmaQueueError> {
        let mut completions = [Default::default(); 4]; // Handle multiple completions

        loop {
            // Poll completion queue (blocking)
            match self.send_cq.wait(&mut completions, None) {
                Ok(completed) => {
                    if !completed.is_empty() {
                        for completion in completed {
                            if completion.wr_id() == expected_wr_id {
                                // Found our completion - check status and return
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
    #[allow(unused)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for RdmaQueue<T> {
    fn drop(&mut self) {
        if let Some(qp) = self.qp.take() {
            drop(qp);
        }
    }
}

// ============================================================================
// Wire Format Implementations
// These MUST match the format expected by workers (ek-computation)
// ============================================================================

/// Worker request structure - sent from controller to worker
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShmqWorkerReq {
    id: usize,
    expert_id: [u8; 64], // Fixed-size null-terminated string
    traceparent: [u8; TRACEPARENT_SIZE],
    input_tensor: Vec<u8>,
}

impl ShmqWorkerReq {
    pub fn new(expert_id: &str, input_tensor: &[u8], traceparent: &str) -> Self {
        static ID: AtomicUsize = AtomicUsize::new(1);

        assert!(expert_id.len() < 64, "expert_id too long");
        assert!(traceparent.len() < TRACEPARENT_SIZE, "traceparent too long");
        assert!(
            input_tensor.len() <= MAX_TENSOR_SIZE,
            "input_tensor too large"
        );

        let mut expert_id_array = [0u8; 64];
        let expert_id_bytes = expert_id.as_bytes();
        let copy_len = std::cmp::min(expert_id_bytes.len(), 63);
        expert_id_array[..copy_len].copy_from_slice(&expert_id_bytes[..copy_len]);

        let mut traceparent_array = [0u8; TRACEPARENT_SIZE];
        let traceparent_bytes = traceparent.as_bytes();
        let traceparent_len = std::cmp::min(traceparent_bytes.len(), TRACEPARENT_SIZE - 1);
        traceparent_array[..traceparent_len].copy_from_slice(&traceparent_bytes[..traceparent_len]);

        Self {
            id: ID.fetch_add(1, Ordering::SeqCst),
            expert_id: expert_id_array,
            traceparent: traceparent_array,
            input_tensor: input_tensor.to_vec(),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    #[allow(unused)]
    pub fn expert_id(&self) -> String {
        let end = self.expert_id.iter().position(|&b| b == 0).unwrap_or(64);
        String::from_utf8(self.expert_id[..end].to_vec()).unwrap()
    }

    #[allow(unused)]
    pub fn input_tensor(&self) -> &[u8] {
        &self.input_tensor
    }

    #[allow(unused)]
    pub fn traceparent(&self) -> String {
        let end = self
            .traceparent
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(TRACEPARENT_SIZE);
        String::from_utf8(self.traceparent[..end].to_vec()).unwrap()
    }
}

impl GeneralShmQueueBytes for ShmqWorkerReq {
    const CAPACITY: usize = std::mem::size_of::<usize>()
        + 64
        + TRACEPARENT_SIZE
        + std::mem::size_of::<usize>()
        + MAX_TENSOR_SIZE;

    fn write_to_slice(&self, slice: &mut [u8]) {
        let mut offset = 0;

        // Write id (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.id.to_le_bytes());
        offset += 8;

        // Write expert_id (64 bytes)
        slice[offset..offset + 64].copy_from_slice(&self.expert_id);
        offset += 64;

        // Write traceparent (128 bytes)
        slice[offset..offset + TRACEPARENT_SIZE].copy_from_slice(&self.traceparent);
        offset += TRACEPARENT_SIZE;

        // Write input_tensor length (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.input_tensor.len().to_le_bytes());
        offset += 8;

        // Write input_tensor data
        let tensor_len = self.input_tensor.len();
        slice[offset..offset + tensor_len].copy_from_slice(&self.input_tensor);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let id = usize::from_le_bytes(bytes[..std::mem::size_of::<usize>()].try_into().unwrap());
        let expert_id = bytes[std::mem::size_of::<usize>()..std::mem::size_of::<usize>() + 64]
            .try_into()
            .unwrap();
        let traceparent_start = std::mem::size_of::<usize>() + 64;
        let traceparent = bytes[traceparent_start..traceparent_start + TRACEPARENT_SIZE]
            .try_into()
            .unwrap();
        let input_tensor_len = usize::from_le_bytes(
            bytes[traceparent_start + TRACEPARENT_SIZE
                ..traceparent_start + TRACEPARENT_SIZE + std::mem::size_of::<usize>()]
                .try_into()
                .unwrap(),
        );
        let input_tensor = bytes
            [traceparent_start + TRACEPARENT_SIZE + std::mem::size_of::<usize>()..]
            [..input_tensor_len]
            .to_vec();

        Self {
            id,
            expert_id,
            traceparent,
            input_tensor,
        }
    }

    fn len(&self) -> usize {
        std::mem::size_of::<usize>()
            + 64
            + TRACEPARENT_SIZE
            + std::mem::size_of::<usize>()
            + self.input_tensor.len()
    }
}

/// Worker response structure - sent from worker to controller
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShmqWorkerResp {
    id: usize,
    output_tensor: Vec<u8>,
}

impl ShmqWorkerResp {
    #[allow(unused)]
    pub fn new(id: usize, output_tensor: Vec<u8>) -> Self {
        assert!(
            output_tensor.len() <= MAX_TENSOR_SIZE,
            "output_tensor too large"
        );
        Self { id, output_tensor }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn output_tensor(&self) -> &[u8] {
        &self.output_tensor
    }
}

impl GeneralShmQueueBytes for ShmqWorkerResp {
    const CAPACITY: usize =
        std::mem::size_of::<usize>() + std::mem::size_of::<usize>() + MAX_TENSOR_SIZE;

    fn write_to_slice(&self, slice: &mut [u8]) {
        let mut offset = 0;

        // Write id (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.id.to_le_bytes());
        offset += 8;

        // Write output_tensor length (8 bytes)
        slice[offset..offset + 8].copy_from_slice(&self.output_tensor.len().to_le_bytes());
        offset += 8;

        // Write output_tensor data
        let tensor_len = self.output_tensor.len();
        slice[offset..offset + tensor_len].copy_from_slice(&self.output_tensor);
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        let id = usize::from_le_bytes(bytes[..std::mem::size_of::<usize>()].try_into().unwrap());
        let output_tensor_len = usize::from_le_bytes(
            bytes[std::mem::size_of::<usize>()
                ..std::mem::size_of::<usize>() + std::mem::size_of::<usize>()]
                .try_into()
                .unwrap(),
        );
        let output_tensor = bytes[std::mem::size_of::<usize>() + std::mem::size_of::<usize>()..]
            [..output_tensor_len]
            .to_vec();

        Self { id, output_tensor }
    }

    fn len(&self) -> usize {
        std::mem::size_of::<usize>() + std::mem::size_of::<usize>() + self.output_tensor.len()
    }
}
