use std::borrow::Borrow;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::Arc;

use ek_base::error::{EKError, EKResult};
use safetensors::tensor::TensorView;

use crate::tch_safetensors::{dtype_to_tch_kind, tch_kind_to_dtype};
use crate::x;

use super::{DType, Device, EkTensor, Expert, ExpertShape, ExpertWeight, FromSafeTensor};

// FFI declarations for CANN library
#[allow(non_camel_case_types)]
type aclrtStream = *mut c_void;
#[allow(non_camel_case_types)]
type aclTensor = *mut c_void;
#[allow(non_camel_case_types)]
type aclOpExecutor = *mut c_void;
#[allow(non_camel_case_types)]
type aclDataType = i32;

// Constants from CANN library
const ACL_SUCCESS: i32 = 0;
const ACL_FORMAT_ND: i32 = 2;
const ACL_FLOAT: aclDataType = 0;
const ACL_FLOAT16: aclDataType = 1;
const ACL_INT8: aclDataType = 2;
const ACL_INT16: aclDataType = 4;
const ACL_UINT8: aclDataType = 5;
const ACL_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const ACL_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const ACL_MEM_MALLOC_HUGE_FIRST: i32 = 0;

// FFI for CANN library
unsafe extern "C" {
    fn aclInit(config_path: *const i8) -> i32;
    fn aclFinalize() -> i32;
    fn aclrtSetDevice(device_id: i32) -> i32;
    fn aclrtResetDevice(device_id: i32) -> i32;
    fn aclrtCreateStream(stream: *mut aclrtStream) -> i32;
    fn aclrtDestroyStream(stream: aclrtStream) -> i32;
    fn aclrtSynchronizeStream(stream: aclrtStream) -> i32;
    fn aclrtMalloc(device_ptr: *mut *mut c_void, size: usize, memory_type: i32) -> i32;
    fn aclrtFree(device_ptr: *mut c_void) -> i32;
    fn aclrtMemcpy(
        dst: *mut c_void,
        dst_size: usize,
        src: *const c_void,
        src_size: usize,
        kind: i32,
    ) -> i32;
    fn aclCreateTensor(
        shape: *const i64,
        shape_size: u64,
        data_type: aclDataType,
        strides: *const i64,
        offset: i64,
        format: i32,
        storage_shape: *const i64,
        storage_shape_size: u64,
        data: *mut c_void,
    ) -> *mut aclTensor;
    fn aclDestroyTensor(tensor: *mut aclTensor) -> i32;
    fn aclnnFFNGetWorkspaceSize(
        self_tensor: *mut aclTensor,
        weight1: *mut aclTensor,
        weight2: *mut aclTensor,
        weight3: *const aclTensor,
        bias1: *const aclTensor,
        bias2: *const aclTensor,
        bias3: *const aclTensor,
        scale1: *const aclTensor,
        scale2: *const aclTensor,
        scale3: *const aclTensor,
        weight2_trans: *const aclTensor,
        weight1_trans: *const aclTensor,
        weight3_trans: *const aclTensor,
        offset2: *const aclTensor,
        activation: *const i8,
        gate_mode: i64,
        out: *mut aclTensor,
        workspace_size: *mut u64,
        executor: *mut *mut aclOpExecutor,
    ) -> i32;
    fn aclnnFFN(
        workspace: *mut c_void,
        workspace_size: u64,
        executor: *mut aclOpExecutor,
        stream: aclrtStream,
    ) -> i32;
}

// Tensor implementation for CANN
pub struct CannTensor {
    shape: Vec<usize>,
    dtype: DType,
    device_ptr: *mut c_void,
    acl_tensor: *mut aclTensor,
}

impl CannTensor {
    // Create tensor from raw data
    fn new(data: &[u8], shape: &[usize], dtype: DType) -> Self {
        let size = shape.iter().product::<usize>() * Self::get_dtype_size(dtype);
        let mut device_mem_ptr: *mut c_void = ptr::null_mut();
        
        unsafe {
            // Allocate device memory
            let ret = aclrtMalloc(&mut device_mem_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
            if ret != ACL_SUCCESS {
                panic!("Failed to allocate device memory: {}", ret);
            }
            
            // Copy data to device
            let ret = aclrtMemcpy(
                device_mem_ptr,
                size,
                data.as_ptr() as *const c_void,
                size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            );
            if ret != ACL_SUCCESS {
                aclrtFree(device_mem_ptr);
                panic!("Failed to copy data to device: {}", ret);
            }
            
            // Create ACL tensor
            let acl_tensor = Self::create_acl_tensor(device_mem_ptr, shape, dtype);
            
            Self {
                shape: shape.to_vec(),
                dtype,
                device_ptr: device_mem_ptr,
                acl_tensor,
            }
        }
    }
    
    // Create ACL tensor from device pointer
    unsafe fn create_acl_tensor(device_ptr: *mut c_void, shape: &[usize], dtype: DType) -> *mut aclTensor {
        // Convert shape to i64
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        
        // Calculate strides
        let mut strides: Vec<i64> = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = shape[i + 1] as i64 * strides[i + 1];
        }
        
        // Create tensor
        unsafe {
            aclCreateTensor(
                shape_i64.as_ptr(),
                shape.len() as u64,
                Self::dtype_to_acl(dtype),
                strides.as_ptr(),
                0,
                ACL_FORMAT_ND,
                shape_i64.as_ptr(),
                shape.len() as u64,
                device_ptr,
            )
        }
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // Convert DType to aclDataType
    fn dtype_to_acl(dtype: DType) -> aclDataType {
        match dtype {
            DType::Uint8 => ACL_UINT8,
            DType::Int8 => ACL_INT8,
            DType::Int16 => ACL_INT16,
            DType::BFloat16 => ACL_FLOAT16,
            DType::Float => ACL_FLOAT,
            DType::Float8e4m3fn => ACL_FLOAT16, // Map to closest available type
            DType::Float8e4m3fnuz => ACL_FLOAT16, // Map to closest available type
        }
    }
    
    // Get size of dtype in bytes
    fn get_dtype_size(dtype: DType) -> usize {
        match dtype {
            DType::Uint8 | DType::Int8 | DType::Float8e4m3fn | DType::Float8e4m3fnuz => 1,
            DType::Int16 | DType::BFloat16 => 2,
            DType::Float => 4,
        }
    }
    
    // Get raw pointer to ACL tensor
    pub fn acl_tensor(&self) -> *mut aclTensor {
        self.acl_tensor
    }
    
    // Get raw device pointer
    pub fn device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }
    
    // Helper function to calculate shape size
    fn get_shape_size(shape: &[usize]) -> usize {
        shape.iter().product()
    }
}

impl Drop for CannTensor {
    fn drop(&mut self) {
        unsafe {
            if !self.acl_tensor.is_null() {
                aclDestroyTensor(self.acl_tensor);
                self.acl_tensor = ptr::null_mut();
            }
            if !self.device_ptr.is_null() {
                aclrtFree(self.device_ptr);
                self.device_ptr = ptr::null_mut();
            }
        }
    }
}

impl Clone for CannTensor {
    fn clone(&self) -> Self {
        let size = Self::get_shape_size(&self.shape) * Self::get_dtype_size(self.dtype);
        let mut device_ptr: *mut c_void = ptr::null_mut();
        
        unsafe {
            // Allocate device memory
            let ret = aclrtMalloc(&mut device_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
            if ret != ACL_SUCCESS {
                panic!("Failed to allocate device memory: {}", ret);
            }
            
            // Copy data from original tensor
            let ret = aclrtMemcpy(
                device_ptr,
                size,
                self.device_ptr,
                size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            );
            if ret != ACL_SUCCESS {
                aclrtFree(device_ptr);
                panic!("Failed to copy data to device: {}", ret);
            }
            
            // Create ACL tensor
            let acl_tensor = Self::create_acl_tensor(device_ptr, &self.shape, self.dtype);
            
            Self {
                shape: self.shape.clone(),
                dtype: self.dtype,
                device_ptr,
                acl_tensor,
            }
        }
    }
}

impl EkTensor for CannTensor {
    fn rand(shape: Vec<usize>, dtype: DType, _dev: Device) -> Self {
        // Generate random data
        let size = Self::get_shape_size(&shape) * Self::get_dtype_size(dtype);
        let mut data = vec![0u8; size];
        
        // Simple random data generation
        for i in 0..size {
            data[i] = (i % 255) as u8;
        }
        
        Self::new(&data, &shape, dtype)
    }
    
    fn stack(tensors: &[Self], dim: usize) -> Self {
        // Check if tensors are empty
        if tensors.is_empty() {
            panic!("Cannot stack empty tensors");
        }
        
        // Get the shape of the first tensor
        let first = &tensors[0];
        let mut new_shape = first.shape.clone();
        new_shape[dim] = tensors.len() * new_shape[dim];
        
        // TODO: Implement stacking logic
        todo!("Stacking tensors is not implemented yet");
    }
    
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
    
    fn serialize(&self) -> Vec<u8> {
        let size = Self::get_shape_size(&self.shape) * Self::get_dtype_size(self.dtype);
        let mut data = vec![0u8; size];
        
        unsafe {
            let ret = aclrtMemcpy(
                data.as_mut_ptr() as *mut c_void,
                size,
                self.device_ptr,
                size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            );
            if ret != ACL_SUCCESS {
                panic!("Failed to copy data from device: {}", ret);
            }
        }
        
        data
    }
    
    fn from_raw(data: &[u8], shape: &[usize], dtype: DType) -> Self {
        Self::new(data, shape, dtype)
    }
    
    fn from_tensor_view(tv: &TensorView<'_>) -> Self {
        let shape: Vec<usize> = tv.shape().to_vec();
        let dtype = DType::from(tv.dtype());
        Self::from_raw(tv.data(), &shape, dtype)
    }
}

impl FromSafeTensor for CannTensor {}

// CANN FFN implementation
pub struct CannFFN {
    dim: usize,
    intermediate_dim: usize,
    weight: ExpertWeight<CannTensor>,
    device_id: i32,
    initialized: bool,
    stream: Option<aclrtStream>,
}

impl CannFFN {
    // Initialize CANN runtime
    fn init_cann(&mut self) -> EKResult<()> {
        // TODO: Current every FFN will initialize CANN runtime, may cause performance issue
        if self.initialized {
            return Ok(());
        }
        
        unsafe {
            // Initialize ACL
            let ret = aclInit(ptr::null());
            if ret != ACL_SUCCESS {
                return Err(EKError::BackendError(format!("Failed to initialize ACL: {}", ret)));
            }
            
            // Set device
            let ret = aclrtSetDevice(self.device_id);
            if ret != ACL_SUCCESS {
                aclFinalize();
                return Err(EKError::BackendError(format!("Failed to set device: {}", ret)));
            }
            
            // Create stream
            let mut stream: aclrtStream = ptr::null_mut();
            let ret = aclrtCreateStream(&mut stream);
            if ret != ACL_SUCCESS {
                aclrtResetDevice(self.device_id);
                aclFinalize();
                return Err(EKError::BackendError(format!("Failed to create stream: {}", ret)));
            }
            
            self.stream = Some(stream);
            self.initialized = true;
            Ok(())
        }
    }
    
    // Cleanup CANN resources
    fn cleanup_cann(&mut self) {
        if !self.initialized {
            return;
        }
        
        unsafe {
            if let Some(stream) = self.stream.take() {
                aclrtDestroyStream(stream);
            }
            aclrtResetDevice(self.device_id);
            aclFinalize();
        }
        
        self.initialized = false;
    }
}

// Implement Drop for CannFFN to ensure resources are cleaned up
impl Drop for CannFFN {
    fn drop(&mut self) {
        self.cleanup_cann();
    }
}

impl Expert<CannTensor> for CannFFN {
    fn backend(&self) -> std::string::String {
        "npu".to_string()
    }
    
    fn shape(&self) -> ExpertShape {
        ExpertShape {
            dim: self.dim,
            hidden: self.intermediate_dim,
        }
    }
    
    fn forward(&self, x: &CannTensor) -> CannTensor {
        // Create output tensor with same shape as input
        let result_shape = x.shape();
        let result = CannTensor::rand(result_shape.clone(), x.dtype, Device::CPU);
        
        unsafe {
            // Get the stream
            let stream = match self.stream {
                Some(s) => s,
                None => panic!("Stream not initialized"),
            };
            
            // Create executor
            let mut executor: *mut aclOpExecutor = ptr::null_mut();
            let mut workspace_size: u64 = 0;
            
            // Get workspace size
            let activation = CString::new("relu").unwrap();
            let ret = aclnnFFNGetWorkspaceSize(
                x.acl_tensor(),
                self.weight.up_w.acl_tensor(),
                self.weight.down_w.acl_tensor(),
                self.weight.gate_w.acl_tensor(),
                ptr::null(), // bias1
                ptr::null(), // bias2
                ptr::null(), // bias3
                ptr::null(), // scale1
                ptr::null(), // scale2
                ptr::null(), // scale3
                ptr::null(), // weight2_trans
                ptr::null(), // weight1_trans
                ptr::null(), // weight3_trans
                ptr::null(), // offset2
                activation.as_ptr() as *const i8,
                1, // gate_mode
                result.acl_tensor(),
                &mut workspace_size,
                &mut executor,
            );
            
            if ret != ACL_SUCCESS {
                panic!("Failed to get workspace size: {}", ret);
            }
            
            // Allocate workspace
            let mut workspace_ptr: *mut c_void = ptr::null_mut();
            if workspace_size > 0 {
                let ret = aclrtMalloc(&mut workspace_ptr, workspace_size as usize, ACL_MEM_MALLOC_HUGE_FIRST);
                if ret != ACL_SUCCESS {
                    panic!("Failed to allocate workspace: {}", ret);
                }
            }
            
            // Execute FFN
            let ret = aclnnFFN(
                workspace_ptr,
                workspace_size,
                executor,
                stream,
            );
            if ret != ACL_SUCCESS {
                if !workspace_ptr.is_null() {
                    aclrtFree(workspace_ptr);
                }
                panic!("Failed to execute FFN: {}", ret);
            }
            
            // Synchronize
            let ret = aclrtSynchronizeStream(stream);
            if ret != ACL_SUCCESS {
                if !workspace_ptr.is_null() {
                    aclrtFree(workspace_ptr);
                }
                panic!("Failed to synchronize stream: {}", ret);
            }
            
            // Free workspace
            if !workspace_ptr.is_null() {
                aclrtFree(workspace_ptr);
            }
        }
        
        result
    }
    
    fn rand_input(&self, batch: usize) -> CannTensor {
        CannTensor::rand(vec![batch, self.dim], DType::Float, Device::CPU)
    }
    
    fn construct(instance: x::EKInstance, weight: ExpertWeight<CannTensor>) -> EKResult<Self> {
        let mut ffn = CannFFN {
            dim: instance.dim,
            intermediate_dim: instance.hidden,
            weight,
            //TODO: Now hard code the device id
            device_id: 0, // Default device ID
            initialized: false,
            stream: None,
        };
        
        // Initialize CANN runtime
        ffn.init_cann()?;
        
        Ok(ffn)
    }
}

#[cfg(test)]
mod test {
    use crate::x;
    use ek_base::utils::workspace_root;
    use safetensors::SafeTensors;
    use tch::IndexOp;
    use test::Bencher;
    extern crate test;

    use crate::{
        ffn::{EkTensor, Expert, ExpertWeight, expert_torch::TorchFFN},
        x::{self, test_root},
    };

    use crate::ffn::expert_torch::TchTensor;

    fn test_correctness() {
        let st_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.weight.safetensors");
        let st_bytes = fs::read(st_fp).unwrap();
        let st = SafeTensors::deserialize(&st_bytes).unwrap();
        let weight = ExpertWeight::from_safetensor(&st).unwrap();
        let inst = x::EKInstance {
            dim: 2048,
            hidden: 768,
            backend: x::ExpertBackendType::Torch,
        };
        let ffn = CannFFN::construct(inst, weight).unwrap();

        let ground_truth_fp = test_root()
            .join("resources")
            .join("qwen3-l0e1.result.safetensors");
        let ground_truth_bytes = fs::read(ground_truth_fp).unwrap();
        let gt_st = SafeTensors::deserialize(&ground_truth_bytes).unwrap();

        let tv = gt_st.tensor("1-input").unwrap();
        let inp = TchTensor::from_tensor_view(&tv);

        let res = ffn.forward(&inp).inner();

        let truth = TchTensor::from_tensor_view(&gt_st.tensor("1-output").unwrap()).inner();

        let _vec1 = Vec::<f32>::try_from(res.i((0, 0..100))).unwrap();
        let _vec2 = Vec::<f32>::try_from(truth.i((0, 0..100))).unwrap();
        (res - truth).sum(tch::Kind::BFloat16).print();
    }
}