// Allow PyO3 macro-generated unsafe patterns
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod client;
mod lb;
mod observability;
mod python;
mod routing;
mod transport;
mod utils;

use python::bindings::PyExpertKitClient;

#[pyfunction]
fn flush_tracing() -> PyResult<()> {
    observability::flush_tracing().map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pyfunction]
fn shutdown_tracing() -> PyResult<()> {
    observability::shutdown_tracing().map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// ExpertKit Transport Library - Rust FFI Module
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the transport client classes
    m.add_class::<PyExpertKitClient>()?;
    m.add_function(wrap_pyfunction!(flush_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(shutdown_tracing, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
