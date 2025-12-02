use pyo3::prelude::*;

mod client;
mod python;
mod routing;
mod transport;

use python::bindings::PyExpertKitClient;

/// ExpertKit Transport Library - Rust FFI Module
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the transport client classes
    m.add_class::<PyExpertKitClient>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
