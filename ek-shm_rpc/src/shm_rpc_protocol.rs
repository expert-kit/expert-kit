use std::mem;
use tch::Tensor;
use std::ffi::CStr;
use std::time::{ SystemTime, UNIX_EPOCH };

pub const MAX_DIMS: usize = 8;
pub const DTYPE_FLOAT32: u32 = 0;
pub const DTYPE_INT32: u32 = 1;
pub const DTYPE_STRING_ARRAY: u32 = 2;
pub const DTYPE_BFLOAT: u32 = 3;

pub const HANDSHAKE_INTERVAL: u64 = 100;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum RpcMethod {
    Default = 0,
    Forward = 1
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorDescriptor {
    pub dtype: u32,
    pub ndim: u32,
    pub shape: [i32; MAX_DIMS],
    pub offset: u64,
    pub element_size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RpcHeader {
    pub time: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RpcRequest {
    pub method: RpcMethod,
    pub input: TensorDescriptor,
    pub expert_ids: TensorDescriptor,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RpcResponse {
    pub success: bool,
    pub output: TensorDescriptor,
}

pub fn size_of_head() -> u64 {
    mem::size_of::<RpcHeader>() as u64
}

pub fn size_of_request() -> u64 {
    size_of_head() + mem::size_of::<RpcRequest>() as u64
}

pub fn size_of_response() -> u64 {
    mem::size_of::<RpcResponse>() as u64
}

pub fn dtype_to_kind(dtype: u32) -> tch::Kind {
    match dtype {
        DTYPE_FLOAT32 => tch::Kind::Float,
        DTYPE_INT32 => tch::Kind::Int,
        DTYPE_BFLOAT => tch::Kind::BFloat16,
        _ => panic!("Unsupported dtype"),
    }
}

pub fn kind_to_dtype(kind: tch::Kind) -> u32 {
    match kind {
        tch::Kind::Float => DTYPE_FLOAT32,
        tch::Kind::Int => DTYPE_INT32,
        tch::Kind::BFloat16 => DTYPE_BFLOAT,
        _ => panic!("Unsupported tensor type"),
    }
}

pub fn element_size_in_bytes(kind: tch::Kind) -> usize {
   match kind {
        tch::Kind::Uint8 => 1,
        tch::Kind::Int8 => 1,
        tch::Kind::Int16 => 2,
        tch::Kind::Half => 2,
        tch::Kind::BFloat16 => 2,
        tch::Kind::Float => 4,
        tch::Kind::Int => 4,
        tch::Kind::Double => 8,
        tch::Kind::Int64 => 8,
        _ => panic!("Unsupported tensor type"),
    }
}

pub fn read_tensor(desc: &TensorDescriptor, shm_base: *mut u8) -> Tensor {
    let shape: Vec<i64> = desc.shape[..desc.ndim as usize]
        .iter()
        .map(|&x| x as i64)
        .collect();

    let num_elements = shape.iter().product::<i64>() as usize;
    let data_ptr = unsafe { shm_base.add(desc.offset as usize) };

    let kind = dtype_to_kind(desc.dtype);
    Tensor::from_data_size(
        unsafe {
            std::slice::from_raw_parts(
                data_ptr,
                num_elements * kind.elt_size_in_bytes(),
            )
        },
        &shape,
        kind
    )
}

pub fn write_tensor(desc: &TensorDescriptor, shm_base: *mut u8, input: &Tensor) {
    let input_data = input.data_ptr() as *const u8;

    unsafe {
        let input_ptr = shm_base.add(desc.offset as usize) as *mut u8;
        std::ptr::copy_nonoverlapping(
            input_data,
            input_ptr,
            input.numel() * element_size_in_bytes(input.kind())
        );
   }
}

pub fn read_expert_ids(
        desc: &TensorDescriptor, shm_ptr: *mut u8
    ) -> Vec<Vec<String>> {
    let rows = desc.shape[0] as usize;
    let cols = desc.shape[1] as usize;
    let element_size = desc.element_size as usize;

    let base_ptr = unsafe { shm_ptr.add(desc.offset as usize) };
    let mut result = Vec::with_capacity(rows);

    for i in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for j in 0..cols {
            let offset = (i * cols * element_size) + (j * element_size);

            #[cfg(target_os = "linux")]
            let ptr = unsafe { base_ptr.add(offset) as *const u8 };
            #[cfg(not(target_os = "linux"))]
            let ptr = unsafe { base_ptr.add(offset) as *const i8 };

            // 读取以null终止的字符串
            let c_str = unsafe { CStr::from_ptr(ptr) };
            let s = c_str.to_str().unwrap().to_string();
            row.push(s);
        }
        result.push(row);
    }

    result
}

pub fn write_expert_ids(
    expert_ids: &[Vec<String>], base_ptr: *mut u8, element_size: usize) {
    let mut offset = 0;
    for row in expert_ids {
        for s in row {
            unsafe {
                let ptr = base_ptr.add(offset);
                let bytes = s.as_bytes();
                let len = bytes.len().min(element_size - 1);

                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    ptr,
                    len
                );

                ptr.add(len).write(0);

                if len < element_size - 1 {
                    std::ptr::write_bytes(
                        ptr.add(len + 1),
                        0,
                        element_size - len - 1
                    );
                }
            }
            offset += element_size;
        }
        let cols = row.len();
        let max_cols = expert_ids.iter().map(|v| v.len()).max().unwrap_or(0);
        if cols < max_cols {
            offset += (max_cols - cols) * element_size;
        }
    }
}

pub fn tensor_to_descriptor(tensor: &Tensor) -> TensorDescriptor {
    let mut desc = TensorDescriptor {
        dtype: kind_to_dtype(tensor.kind()),
        ndim: tensor.size().len() as u32,
        shape: [0; MAX_DIMS],
        offset: 0,
        element_size: element_size_in_bytes(tensor.kind()) as u32,
    };

    for (i, &dim) in tensor.size().iter().enumerate() {
        if i < MAX_DIMS {
            desc.shape[i] = dim as i32;
        }
    }

    desc
}

pub fn expert_ids_to_descriptor(expert_ids: &[Vec<String>]) -> (TensorDescriptor, usize) {
    let rows = expert_ids.len();
    let cols = expert_ids.iter().map(|v| v.len()).max().unwrap_or(0);

    let max_str_len = expert_ids.iter()
        .flat_map(|v| v.iter())
        .map(|s| s.len() + 1)
        .max()
        .unwrap_or(0);

    let desc = TensorDescriptor {
        dtype: DTYPE_STRING_ARRAY,
        ndim: 2,
        shape: [rows as i32, cols as i32, 0, 0, 0, 0, 0, 0],
        offset: 0,
        element_size: max_str_len as u32,
    };

    (desc, max_str_len)
}

pub fn write_head(shm_base: *mut u8) {
    let header = unsafe {
        &mut *(shm_base as *mut RpcHeader)
    };
    header.time = current_timestamp();
}

pub fn clear_head(shm_base: *mut u8) {
    let header = unsafe {
        &mut *(shm_base as *mut RpcHeader)
    };
    header.time = 0;
}

pub fn read_head(shm_base: *mut u8) -> RpcHeader {
    unsafe {
        *(shm_base as *const RpcHeader)
    }
}

pub fn check_connection(shm_base: *mut u8) -> bool {
    let head = read_head(shm_base);
    let now = current_timestamp();

    now - head.time < HANDSHAKE_INTERVAL * 2
}

pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
