use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::client::ExpertKitClient as RustExpertKitClient;
use crate::observability::init_tracing;
use crate::utils::{TensorMetadata, pytorch_to_tch_tensor, tch_to_pytorch_tensor};

const DEFAULT_THREAD_NUM: usize = 16;

/// Default timeout for requests in seconds.
/// Must be long enough for the controller fallback path (which includes
/// routing refresh + retry) to complete without premature abort.
const DEFAULT_TIMEOUT_SEC: f64 = 30.0;

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
        if env_logger::try_init().is_ok() {
            log::info!("Logger initialized");
        }

        let timeout = timeout_sec.unwrap_or(DEFAULT_TIMEOUT_SEC);

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
        {
            let _entered = runtime.enter();
            init_tracing();
        }

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

    /// Forward expert computation with automatic failover
    /// Uses direct worker path with retry, falls back to controller if needed
    #[pyo3(signature = (expert_ids, hidden_state, use_fallback=true))]
    fn forward_expert<'py>(
        &self,
        py: Python<'py>,
        expert_ids: Vec<Vec<String>>,
        hidden_state: &PyAny,
        use_fallback: bool,
    ) -> PyResult<PyObject> {
        let t = std::time::Instant::now();
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        // Extract tensor metadata from Python
        let metadata = TensorMetadata::from_pytorch(hidden_state)?;

        log::debug!(
            "[PyBinding-Time] 🚗 Runtime get and extracted tensor metadata in {:?} μs",
            t.elapsed().as_micros()
        );

        // Convert PyTorch tensor to tch::Tensor
        let t = std::time::Instant::now();
        let input_tensor = pytorch_to_tch_tensor(hidden_state, &metadata)?;
        log::debug!(
            "[PyBinding-Time] 🚗 Converted PyTorch tensor to tch::Tensor in {:?} μs",
            t.elapsed().as_micros()
        );

        // Release GIL and process
        let t = std::time::Instant::now();
        let output_tensor = py.allow_threads(|| {
            runtime
                .block_on(async {
                    if use_fallback {
                        // Use fallback version for maximum resilience
                        client
                            .forward_expert_tensor_with_fallback(expert_ids, input_tensor)
                            .await
                    } else {
                        // Direct path only (still has retry logic)
                        client.forward_expert_tensor(expert_ids, input_tensor).await
                    }
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        log::debug!(
            "[PyBinding-Time] 🚗 Processed tensor in {:?} μs",
            t.elapsed().as_micros()
        );

        // Convert tch::Tensor back to PyTorch tensor
        let t = std::time::Instant::now();
        let final_tensor = tch_to_pytorch_tensor(py, &output_tensor, &metadata.device_str)?;
        log::debug!(
            "[PyBinding-Time] 🚗 Converted tch::Tensor back to PyTorch tensor in {:?} μs",
            t.elapsed().as_micros()
        );

        Ok(final_tensor.into())
    }

    /// Reset all worker stats (RTT, throughput, inflight).
    /// Call between benchmark iterations to avoid stale EMA bias.
    fn reset_stats(&self, py: Python) -> PyResult<()> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Client not initialized"))?;

        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;

        py.allow_threads(|| {
            runtime.block_on(async { client.reset_stats().await });
            Ok(())
        })
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
