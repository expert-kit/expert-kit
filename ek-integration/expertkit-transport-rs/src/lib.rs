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

/// ExpertKit Transport Library - Rust FFI Module
#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register the transport client classes
    m.add_class::<PyExpertKitClient>()?;

    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
