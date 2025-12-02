use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::client::ExpertKitClient as RustExpertKitClient;
use crate::transport::{
    ExpertRequest, Transport, auto::AutoTransport, grpc::GrpcTransport, mock::MockTransport,
    shm::ShmTransport,
};

/// High-level ExpertKit client with routing and batching (Phase 4)
#[pyclass]
pub struct PyExpertKitClient {
    client: Option<RustExpertKitClient>,
    controller_addr: String,
    timeout_sec: f64,
    runtime: Option<tokio::runtime::Runtime>, // Shared runtime for all requests
}

#[pymethods]
impl PyExpertKitClient {
    #[new]
    fn new(controller_addr: String, timeout_sec: Option<f64>) -> PyResult<Self> {
        let timeout = timeout_sec.unwrap_or(2.0);

        // Create ONE shared Tokio runtime for all requests
        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e))
        })?;

        Ok(Self {
            client: Some(RustExpertKitClient::new(controller_addr.clone(), timeout)),
            controller_addr,
            timeout_sec: timeout,
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

    /// Get routing table version
    fn get_routing_version(&self, py: Python) -> PyResult<u64> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        py.allow_threads(|| Ok(runtime.block_on(async { client.get_routing_version().await })))
    }
}
