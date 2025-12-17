use anyhow::Result;
use log::{debug, warn};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use safetensors::{Dtype, SafeTensors, tensor::View};
use tch::{Kind, Tensor};

// ============================================================================
// Dtype Conversion (tch::Kind ↔ safetensors::Dtype)
// ============================================================================

/// Convert tch::Kind to safetensors Dtype
pub fn tch_kind_to_dtype(kind: Kind) -> Result<Dtype> {
    let dtype = match kind {
        Kind::Bool => Dtype::BOOL,
        Kind::Uint8 => Dtype::U8,
        Kind::Int8 => Dtype::I8,
        Kind::Int16 => Dtype::I16,
        Kind::Int => Dtype::I32,
        Kind::Int64 => Dtype::I64,
        Kind::BFloat16 => Dtype::BF16,
        Kind::Half => Dtype::F16,
        Kind::Float => Dtype::F32,
        Kind::Double => Dtype::F64,
        kind => return Err(anyhow::anyhow!("Unsupported kind: {:?}", kind)),
    };
    Ok(dtype)
}

/// Convert safetensors Dtype to tch::Kind
pub fn dtype_to_tch_kind(dtype: Dtype) -> Result<Kind> {
    let kind = match dtype {
        Dtype::BOOL => Kind::Bool,
        Dtype::U8 => Kind::Uint8,
        Dtype::I8 => Kind::Int8,
        Dtype::I16 => Kind::Int16,
        Dtype::I32 => Kind::Int,
        Dtype::I64 => Kind::Int64,
        Dtype::BF16 => Kind::BFloat16,
        Dtype::F16 => Kind::Half,
        Dtype::F32 => Kind::Float,
        Dtype::F64 => Kind::Double,
        dtype => return Err(anyhow::anyhow!("Unsupported dtype: {:?}", dtype)),
    };
    Ok(kind)
}

// ============================================================================
// SafeTensors Serialization (tch::Tensor ↔ bytes)
// ============================================================================

/// SafeView implementation for tch::Tensor
struct SafeView<'a> {
    tensor: &'a Tensor,
    shape: Vec<usize>,
    dtype: Dtype,
}

impl<'a> TryFrom<&'a Tensor> for SafeView<'a> {
    type Error = anyhow::Error;

    fn try_from(tensor: &'a Tensor) -> Result<Self> {
        if tensor.is_sparse() {
            return Err(anyhow::anyhow!("Cannot serialize sparse tensors"));
        }

        if !tensor.is_contiguous() {
            return Err(anyhow::anyhow!("Cannot serialize non-contiguous tensors"));
        }

        let dtype = tch_kind_to_dtype(tensor.kind())?;
        let shape = tensor.size().iter().map(|&x| x as usize).collect();
        Ok(Self {
            tensor,
            shape,
            dtype,
        })
    }
}

impl View for SafeView<'_> {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<'_, [u8]> {
        let mut data = vec![0; self.data_len()];
        let numel = self.tensor.numel();
        self.tensor.f_copy_data_u8(&mut data, numel).unwrap();
        data.into()
    }

    fn data_len(&self) -> usize {
        self.tensor.numel() * self.tensor.kind().elt_size_in_bytes()
    }
}

/// Serialize tch::Tensor to safetensors format
pub fn serialize_tch_tensor_2_safetensor(tensor: &Tensor) -> Result<Vec<u8>> {
    // Convert to CPU and make contiguous
    let cpu_tensor = tensor.to(tch::Device::Cpu).contiguous();

    // Create SafeView
    let view = SafeView::try_from(&cpu_tensor)?;

    // Serialize using safetensors
    let views = vec![("data", view)];
    safetensors::serialize(views, &None)
        .map_err(|e| anyhow::anyhow!("Failed to serialize tensor: {}", e))
}

/// Deserialize safetensors to tch::Tensor
pub fn deserialize_safetensor_2_tch_tensor(bytes: &[u8]) -> Result<Tensor> {
    let safetensors = SafeTensors::deserialize(bytes)?;
    let view = safetensors.tensor("data")?;

    let shape: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
    let kind = dtype_to_tch_kind(view.dtype())?;

    let tensor = Tensor::f_from_data_size(view.data(), &shape, kind)
        .map_err(|e| anyhow::anyhow!("Failed to create tensor: {:?}", e))?;

    Ok(tensor)
}

// ============================================================================
// PyTorch Tensor Conversion (PyTorch ↔ tch::Tensor)
// ============================================================================

/// Metadata extracted from PyTorch tensor
pub struct TensorMetadata {
    pub shape: Vec<i64>,
    pub tch_kind: tch::Kind,
    pub device_str: String,
    pub target_device: tch::Device,
    pub requires_grad: bool,
}

impl TensorMetadata {
    /// Extract tensor metadata from PyTorch tensor
    pub fn from_pytorch(py_tensor: &PyAny) -> PyResult<Self> {
        let t = std::time::Instant::now();

        // Extract shape
        let shape: Vec<i64> = py_tensor
            .getattr("shape")?
            .extract::<Vec<usize>>()?
            .iter()
            .map(|&s| s as i64)
            .collect();

        // Extract device
        let device_str: String = py_tensor.getattr("device")?.getattr("type")?.extract()?;
        let requires_grad: bool = py_tensor.getattr("requires_grad")?.extract()?;

        // Extract dtype from PyTorch tensor
        let dtype_obj = py_tensor.getattr("dtype")?;
        let dtype_str: String = format!("{:?}", dtype_obj);

        debug!(
            "[PyBinding-Time] 🔧 Extracted tensor metadata in {:?} μs",
            t.elapsed().as_micros()
        );

        // Map PyTorch dtype to tch::Kind
        let tch_kind = match dtype_str.as_str() {
            s if s.contains("float32") => tch::Kind::Float,
            s if s.contains("bfloat16") => tch::Kind::BFloat16,
            s if s.contains("float16") => tch::Kind::Half,
            s if s.contains("float64") => tch::Kind::Double,
            s if s.contains("int8") => tch::Kind::Int8,
            s if s.contains("int16") => tch::Kind::Int16,
            s if s.contains("int32") => tch::Kind::Int,
            s if s.contains("int64") => tch::Kind::Int64,
            s if s.contains("uint8") => tch::Kind::Uint8,
            s if s.contains("bool") => tch::Kind::Bool,
            _ => {
                warn!(
                    "[PyBinding] Unknown dtype: {}, defaulting to Float32",
                    dtype_str
                );
                tch::Kind::Float
            }
        };

        debug!(
            "[PyBinding] Received tensor: shape={:?}, dtype={:?}, device={}, requires_grad={}",
            shape, tch_kind, device_str, requires_grad
        );

        // Determine target device
        let target_device = if device_str == "cuda" {
            let device_index: i64 = py_tensor
                .getattr("device")?
                .getattr("index")?
                .extract()
                .unwrap_or(0);
            tch::Device::Cuda(device_index as usize)
        } else {
            tch::Device::Cpu
        };

        Ok(Self {
            shape,
            tch_kind,
            device_str,
            target_device,
            requires_grad,
        })
    }
}

/// Convert PyTorch tensor to tch::Tensor
pub fn pytorch_to_tch_tensor(py_tensor: &PyAny, metadata: &TensorMetadata) -> PyResult<Tensor> {
    // For CUDA tensors, we MUST copy to CPU first before extracting pointer
    let cpu_tensor = if metadata.device_str == "cuda" {
        debug!("[PyBinding] CUDA tensor detected, copying to CPU");
        py_tensor.call_method0("cpu")?
    } else {
        py_tensor
    };

    let t = std::time::Instant::now();
    debug!(
        "[PyBinding-Time] 🔧 Prepared tensor on CPU in {:?} μs",
        t.elapsed().as_micros()
    );

    // Ensure contiguous layout
    let t = std::time::Instant::now();
    let cpu_tensor = cpu_tensor.call_method0("contiguous")?;
    debug!(
        "[PyBinding-Time] 🔧 Made tensor contiguous in {:?} μs",
        t.elapsed().as_micros()
    );

    // Extract data into Rust-owned memory with proper dtype handling
    let t = std::time::Instant::now();
    let numel: usize = metadata.shape.iter().product::<i64>() as usize;
    let data_ptr = cpu_tensor.call_method0("data_ptr")?.extract::<usize>()?;

    debug!(
        "[PyBinding-Time] 🔧 Extracted data pointer in {:?} μs",
        t.elapsed().as_micros()
    );

    debug!(
        "[PyBinding] Extracting {} elements as {:?}",
        numel, metadata.tch_kind
    );

    // Create tch::Tensor directly from raw data with correct dtype
    let t = std::time::Instant::now();
    let input_tensor = unsafe {
        let element_size = metadata.tch_kind.elt_size_in_bytes();
        let data_slice = std::slice::from_raw_parts(data_ptr as *const u8, numel * element_size);

        Tensor::f_from_data_size(data_slice, &metadata.shape, metadata.tch_kind).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create tensor: {:?}", e))
        })?
    };

    debug!(
        "[PyBinding-Time] 🔧 Created tch::Tensor in {:?} μs",
        t.elapsed().as_micros()
    );

    debug!("[PyBinding] Created tch::Tensor: {:?}", input_tensor.size());

    Ok(input_tensor)
}

/// Convert tch::Tensor back to PyTorch tensor
pub fn tch_to_pytorch_tensor<'py>(
    py: Python<'py>,
    output_tensor: &Tensor,
    original_device_str: &str,
) -> PyResult<&'py PyAny> {
    let t = std::time::Instant::now();
    let torch = py.import("torch")?;
    let output_shape: Vec<i64> = output_tensor.size();
    let cpu_output = output_tensor.to(tch::Device::Cpu);

    debug!(
        "[PyBinding] Got output tensor: shape={:?}, dtype={:?}",
        output_shape,
        cpu_output.kind()
    );

    // Create PyTorch tensor based on output dtype
    let py_tensor = match cpu_output.kind() {
        tch::Kind::Half | tch::Kind::BFloat16 => {
            // Extract raw bytes directly
            let numel = cpu_output.numel();
            let element_size = cpu_output.kind().elt_size_in_bytes();
            let mut data = vec![0u8; numel * element_size];
            cpu_output.f_copy_data_u8(&mut data, numel).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to copy tensor data: {:?}",
                    e
                ))
            })?;

            // Create PyTorch tensor directly from raw bytes with correct dtype
            let dtype_name = if cpu_output.kind() == tch::Kind::Half {
                "float16"
            } else {
                "bfloat16"
            };
            let torch_dtype = torch.getattr(dtype_name)?;

            // Use frombuffer to create tensor from raw bytes
            let py_bytes = PyBytes::new(py, &data);
            let kwargs = PyDict::new(py);
            kwargs.set_item("dtype", torch_dtype)?;

            torch
                .call_method("frombuffer", (py_bytes,), Some(kwargs))?
                .call_method1("reshape", (output_shape,))?
                .call_method0("clone")? // Clone to get a writable tensor
        }
        tch::Kind::Float => {
            let data: Vec<f32> = cpu_output.view([-1]).try_into().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to extract data: {:?}",
                    e
                ))
            })?;
            torch
                .call_method1("tensor", (data,))?
                .call_method1("reshape", (output_shape,))?
        }
        tch::Kind::Double => {
            let data: Vec<f64> = cpu_output.view([-1]).try_into().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to extract data: {:?}",
                    e
                ))
            })?;
            torch
                .call_method1("tensor", (data,))?
                .call_method1("reshape", (output_shape,))?
        }
        _ => {
            // For other types, convert to float32 as fallback
            let data: Vec<f32> = cpu_output
                .to_kind(tch::Kind::Float)
                .view([-1])
                .try_into()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to extract data: {:?}",
                        e
                    ))
                })?;
            torch
                .call_method1("tensor", (data,))?
                .call_method1("reshape", (output_shape,))?
        }
    };

    debug!(
        "[PyBinding-Time] 🔧 Converted back to PyTorch tensor in {:?} μs",
        t.elapsed().as_micros()
    );

    // Move to original device if needed
    let final_tensor = if original_device_str == "cuda" {
        py_tensor.call_method1("cuda", ())?
    } else {
        py_tensor
    };

    Ok(final_tensor)
}
