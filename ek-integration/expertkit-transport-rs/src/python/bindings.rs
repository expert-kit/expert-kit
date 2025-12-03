use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::client::ExpertKitClient as RustExpertKitClient;

const DEFAULT_THREAD_NUM: usize = 6;

/// High-level ExpertKit client with routing and batching
#[pyclass]
pub struct PyExpertKitClient {
    client: Option<RustExpertKitClient>,
    runtime: Option<tokio::runtime::Runtime>, // Shared runtime for all requests
}

#[pymethods]
impl PyExpertKitClient {
    #[new]
    fn new(controller_addr: String, timeout_sec: Option<f64>) -> PyResult<Self> {
        let timeout = timeout_sec.unwrap_or(2.0);

        // Create ONE shared Tokio runtime for all requests
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(DEFAULT_THREAD_NUM)
            .enable_all()
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create runtime: {}",
                    e
                ))
            })?;

        Ok(Self {
            client: Some(RustExpertKitClient::new(controller_addr, timeout)),
            runtime: Some(runtime),
        })
    }

    /// Connect to controller and fetch routing table
    fn connect(&mut self, py: Python) -> PyResult<()> {
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        py.allow_threads(|| {
            runtime
                .block_on(async { client.connect().await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Send batch of expert requests with automatic worker-level batching
    fn send_expert_batch<'py>(
        &self,
        py: Python<'py>,
        expert_ids: Vec<String>,
        tensor_data: Vec<&PyBytes>,
    ) -> PyResult<Vec<&'py PyBytes>> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        // Convert Python bytes to Vec<Vec<u8>>
        let tensors: Vec<Vec<u8>> = tensor_data.iter().map(|b| b.as_bytes().to_vec()).collect();

        // Use the shared runtime (not a new one each time!)
        let responses = py.allow_threads(|| {
            runtime
                .block_on(async { client.send_expert_batch(expert_ids, tensors).await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        // Convert back to Python bytes
        Ok(responses
            .into_iter()
            .map(|data| PyBytes::new(py, &data))
            .collect())
    }

    /// Forward expert computation
    fn forward_expert<'py>(
        &self,
        py: Python<'py>,
        expert_ids: Vec<Vec<String>>,
        hidden_state: &PyBytes,
    ) -> PyResult<&'py PyBytes> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        // Convert to Vec<u8>
        let hidden_state_bytes = hidden_state.as_bytes().to_vec();

        // Release GIL and do all processing in Rust
        let response_bytes = py.allow_threads(|| {
            runtime
                .block_on(async { client.forward_expert(expert_ids, hidden_state_bytes).await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        // Return as PyBytes
        Ok(PyBytes::new(py, &response_bytes))
    }

    /// Refresh routing table
    fn refresh_routing(&self, py: Python) -> PyResult<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        py.allow_threads(|| {
            runtime
                .block_on(async { client.refresh_routing().await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}
