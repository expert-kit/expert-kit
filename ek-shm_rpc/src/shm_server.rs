use libc::c_void;
use libc::{ O_RDWR, O_CREAT, O_TRUNC, MAP_SHARED, PROT_READ, PROT_WRITE };
use libc::{ S_IRUSR, S_IWUSR, off_t, size_t };
use std::ffi::CString;
use std::ptr;
use std::thread;
use std::sync::atomic::{ AtomicBool, Ordering };
use std::sync::Arc;
use tch::Tensor;
use ek_base::error::{ EKResult, EKError };
use crate::posix_sem::PosixSem;
use crate::shm_rpc_protocol::*;

struct SharedResources {
    shm_addr: *mut c_void,
    shm_size: size_t,
    shm_fd: i32,
    req_sem: Arc<PosixSem>,
    resp_sem: Arc<PosixSem>,
    running: Arc<AtomicBool>,
}

pub struct SharedMemoryData {
    pub input: Option<Arc<Tensor>>,
    pub output: Option<Arc<Tensor>>,
    pub expert_ids: Option<Arc<Vec<Vec<String>>>>
}

impl Clone for SharedMemoryData {
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            output: self.output.clone(),
            expert_ids: self.expert_ids.clone(),
        }
    }
}

unsafe impl Send for SharedMemoryData {}
unsafe impl Sync for SharedMemoryData {}

pub trait SharedMemoryService {
    fn forward(&self, data: &SharedMemoryData) -> SharedMemoryData;
}

pub struct SharedMemoryRpcServer {
    shm_name: String,
    req_sem_name: String,
    resp_sem_name: String,
    req_resp_size: size_t,
    resources: Arc<SharedResources>,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    shm_client: Arc<dyn SharedMemoryService>
}

impl Drop for SharedResources {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.shm_addr, self.shm_size);
            libc::close(self.shm_fd);
        }
    }
}

unsafe impl Send for SharedMemoryRpcServer {}
unsafe impl Sync for SharedMemoryRpcServer {}

impl SharedMemoryRpcServer {
    pub fn new<T: SharedMemoryService + 'static>(shm_name: String, shm_size: size_t,
               req_sem_name: String, resp_sem_name: String, shm_client: T) -> EKResult<Self> {
        let name_cstr = CString::new(shm_name.as_str()).expect("CString::new failed");
        let shm_fd = unsafe {
            libc::shm_open(
                name_cstr.as_ptr(),
                O_RDWR | O_CREAT | O_TRUNC,
                (S_IRUSR | S_IWUSR) as u32
            )
        };
        if shm_fd == -1 {
            return Err(EKError::IoError(std::io::Error::last_os_error().into()));
        }
        let shm_addr = unsafe {
            libc::ftruncate(shm_fd, shm_size as off_t);
            libc::mmap(
                ptr::null_mut(),
                shm_size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                shm_fd,
                0,
            )
        };
        if shm_addr == libc::MAP_FAILED {
            unsafe { libc::close(shm_fd) };
            return Err(EKError::IoError(std::io::Error::last_os_error().into()));
        }

        let req_sem = PosixSem::open(req_sem_name.clone(), true)?;
        let resp_sem = PosixSem::open(resp_sem_name.clone(), true)?;

        Ok(Self {
            shm_name,
            req_sem_name,
            resp_sem_name,
            req_resp_size: shm_size / 2,
            resources: Arc::new(SharedResources {
                shm_addr,
                shm_size,
                shm_fd,
                req_sem: Arc::new(req_sem),
                resp_sem: Arc::new(resp_sem),
                running: Arc::new(AtomicBool::new(true))
            }),
            thread_handle: None,
            shm_client: Arc::new(shm_client)
        })
    }

    fn forward_descriptor(&self, input_desc: &TensorDescriptor, shm_ptr: *mut u8,
                          expert_ids: &Vec<Vec<String>>) -> (Tensor, TensorDescriptor) {
        let input_tensor = read_tensor(input_desc, shm_ptr);
        let input_kind = input_tensor.kind();
        let shm_data = SharedMemoryData {
            input: Some(Arc::new(input_tensor)),
            output: None,
            expert_ids: Some(Arc::new(expert_ids.to_vec()))
        };

        let output_data_shm = self.shm_client.forward(&shm_data);
        let output_tensor_shm = output_data_shm.output.expect("missing output tensor");
        let output_tensor = output_tensor_shm.to_kind(input_kind);

        let output_desc = tensor_to_descriptor(&output_tensor);
        (output_tensor, output_desc)
    }

    pub fn start(&mut self) {
        let server = Arc::new(self.clone());
        let server_clone = Arc::clone(&server);

        self.thread_handle = Some(thread::spawn(move || {
            let inner: SharedMemoryRpcServer = server_clone.as_ref().clone();
            let resources = inner.resources.as_ref();
            resources.reset_semaphores();
            log::info!("shm mem server listen on: {}", server_clone.shm_name);
            while resources.running.load(Ordering::SeqCst) {
                let ret = resources.req_sem.timedwait(1);
                if ret == 1 {
                    continue;
                } else if ret == -1 {
                    break;
                }

                // 直接读取共享内存中的结构体
                let request = unsafe {
                    let req_ptr = resources.shm_addr.add(size_of_head() as usize) as *const RpcRequest;
                    std::ptr::read_volatile(req_ptr)
                };
                let response = match request.method {
                    RpcMethod::Forward => {
                        let shm_base = resources.shm_addr as *mut u8;
                        let expert_ids = read_expert_ids(
                            &request.expert_ids, shm_base
                        );
                        let (output_tensor_desc, output_desc) = inner.forward_descriptor(
                            &request.input,
                            shm_base,
                            &expert_ids
                        );
                        let output_tensor = output_tensor_desc.contiguous();
                        let output_offset = inner.req_resp_size as u64 + size_of_response();
                        let mut final_output_desc = output_desc;
                        final_output_desc.offset = output_offset;
                        if output_tensor.numel() == 0 {
                            log::error!("Attempted to copy empty tensor");
                            return;
                        }

                        if output_tensor.data_ptr().is_null() {
                            log::error!("Tensor has null data pointer");
                            return;
                        }
                        write_tensor(&final_output_desc, shm_base, &output_tensor);

                        RpcResponse {
                            success: true,
                            output: final_output_desc,
                        }
                    },
                    _ => panic!("shm not support or data is not initialized"),
                };
                let resp_ptr = unsafe {
                    resources.shm_addr.offset((inner.req_resp_size) as isize) as *mut RpcResponse
                };
                unsafe {
                    std::ptr::write_volatile(resp_ptr, response);
                    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                }
                resources.resp_sem.post();
            }
            log::info!("shm mem server stoped: {}", server_clone.shm_name);
        }));

        // update handshake
        let heartbeat_server = Arc::clone(&server);
        thread::spawn(move || {
            let interval = std::time::Duration::from_millis(HANDSHAKE_INTERVAL);

            loop {
                if !heartbeat_server.resources.running.load(Ordering::SeqCst) {
                    break;
                }
                write_head(heartbeat_server.resources.shm_addr as *mut u8);
                thread::sleep(interval);
            }
        });
    }
}

impl Clone for SharedResources {
    fn clone(&self) -> Self {
        Self {
            shm_addr: self.shm_addr,
            shm_size: self.shm_size,
            shm_fd: self.shm_fd,
            req_sem: self.req_sem.clone(),
            resp_sem: self.resp_sem.clone(),
            running: Arc::clone(&self.running),
        }
    }
}

impl SharedResources {
    fn reset_semaphores(&self) {
        self.req_sem.reset();
        self.resp_sem.reset();
    }
}

impl Clone for SharedMemoryRpcServer {
    fn clone(&self) -> Self {
        Self {
            shm_name: self.shm_name.clone(),
            req_sem_name: self.req_sem_name.clone(),
            resp_sem_name: self.resp_sem_name.clone(),
            req_resp_size: self.req_resp_size,
            resources: self.resources.clone(),
            thread_handle: None,
            shm_client: self.shm_client.clone()
        }
    }
}

impl Drop for SharedMemoryRpcServer {
    fn drop(&mut self) {
        self.resources.running.store(false, Ordering::SeqCst);
        self.resources.reset_semaphores();

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }

        unsafe {
            if let Ok(name) = CString::new(self.shm_name.as_str()) {
                libc::shm_unlink(name.as_ptr());
            }
        }
    }
}
