// FIXME: suppress warnings, should update py03 later
#![allow(non_local_definitions, unsafe_op_in_unsafe_fn)]
use pyo3::prelude::*;

mod client;
mod python;
mod routing;
mod trace;
mod transport;
mod utils;

use python::bindings::PyExpertKitClient;
use trace::PyExpertKitTracer;

/// ExpertKit Transport Library - Rust FFI Module
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the transport client classes
    m.add_class::<PyExpertKitClient>()?;
    m.add_class::<PyExpertKitTracer>()?;
    m.add_function(wrap_pyfunction!(trace::shutdown_tracer_provider, m)?)?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
