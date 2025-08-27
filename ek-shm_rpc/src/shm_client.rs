use std::ffi::CString;
use libc::{ shm_open, mmap, close };
use libc::{ O_RDWR, PROT_READ, PROT_WRITE, MAP_SHARED };
use libc::{ c_void, size_t };
use std::ptr;
use tch::Tensor;
use ek_base::error::{ EKResult, EKError };
use crate::posix_sem::PosixSem;
use crate::shm_rpc_protocol::*;
use std::sync::{ Arc, Mutex };


#[derive(Clone)]
#[allow(dead_code)]
pub struct SharedMemoryHandle {
    shm_addr: *mut c_void,
    shm_size: size_t,
    shm_fd: i32,
}

pub struct SharedMemoryRpcClient {
    handle: Arc<SharedMemoryHandle>,
    req_sem: PosixSem,
    resp_sem: PosixSem,
    req_resp_size: usize,
    lock: Arc<Mutex<i32>>
}

unsafe impl Send for SharedMemoryRpcClient {}
unsafe impl Sync for SharedMemoryRpcClient {}

impl SharedMemoryRpcClient {
    pub fn new(
        shm_name: String,
        req_sem_name: String,
        resp_sem_name: String,
        shm_size: size_t,
    ) -> EKResult<Self> {
        let name_cstr = CString::new(shm_name)
            .map_err(|e| EKError::InvalidInput(format!("CString conversion failed: {}", e)))?;
        let shm_fd = unsafe {
            shm_open(
                name_cstr.as_ptr(),
                O_RDWR,
                0o600,
            )
        };

        if shm_fd == -1 {
            return Err(EKError::IoError(std::io::Error::last_os_error().into()));
        }

        let shm_addr = unsafe {
            mmap(
                ptr::null_mut(),
                shm_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                shm_fd,
                0,
            )
        };

        if shm_addr == libc::MAP_FAILED {
            unsafe { close(shm_fd) };
            return Err(EKError::IoError(std::io::Error::last_os_error().into()));
        }

        let req_sem = PosixSem::open(req_sem_name, false)?;
        let resp_sem = PosixSem::open(resp_sem_name, false)?;

        let lock = Arc::new(Mutex::new(0));

        Ok(Self {
            handle: Arc::new(SharedMemoryHandle{
                shm_addr,
                shm_size,
                shm_fd,
            }),
            req_sem,
            resp_sem,
            req_resp_size: shm_size as usize / 2,
            lock: lock
        })
    }

    pub fn check_connection(&self) -> bool {
        check_connection(self.handle.shm_addr as *mut u8)
    }

    pub fn forward(&self, input_t: &Tensor, expert_ids: &[Vec<String>]) -> EKResult<Tensor> {
        let shm_base = self.handle.shm_addr as *mut u8;
        let _unused = self.lock.lock().unwrap();
        self.req_sem.reset();
        self.resp_sem.reset();

        let input_contig = input_t.contiguous();
        let input_desc = tensor_to_descriptor(&input_contig);

        let (expert_ids_desc, element_size) = expert_ids_to_descriptor(expert_ids);

        let mut request = RpcRequest {
            method: RpcMethod::Forward,
            input: input_desc,
            expert_ids: expert_ids_desc,
        };

        let input_element_size = element_size_in_bytes(input_contig.kind());
        request.input.offset = size_of_request();
        request.expert_ids.offset = request.input.offset + (input_contig.numel() * input_element_size) as u64;

        unsafe {
            let req_ptr = shm_base.add(size_of_head() as usize) as *mut RpcRequest;
            std::ptr::write_volatile(req_ptr, request);
        }
        write_tensor(&request.input, shm_base, &input_contig);
        let expert_base = unsafe {
            shm_base.add(request.expert_ids.offset as usize)
        };
        write_expert_ids(expert_ids, expert_base, element_size);

        self.req_sem.post();
        self.resp_sem.wait();

        let resp_ptr = unsafe {
            shm_base.add(self.req_resp_size) as *const RpcResponse
        };
        let response = unsafe { ptr::read_volatile(resp_ptr) };

        if !response.success {
            return Err(EKError::InvalidInput("RPC call failed".into()));
        }

        let output = read_tensor(&response.output, shm_base);

        self.resp_sem.post();

        Ok(output)
    }
}
