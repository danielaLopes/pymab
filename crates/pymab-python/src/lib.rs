//! Private Python bindings for the PyMAB Rust core.

use pyo3::prelude::*;

mod environment;
mod error;
mod experiment;
mod policy;

/// Return whether the compiled native extension loaded successfully.
#[pyfunction]
#[must_use]
fn native_available() -> bool {
    true
}

/// Return the version of the linked Rust core.
#[pyfunction]
#[must_use]
fn core_version() -> &'static str {
    pymab::version()
}

/// Return the versioned native random-stream scheme identifier.
#[pyfunction]
#[must_use]
fn rng_scheme_version() -> &'static str {
    pymab::rng_scheme_version()
}

/// Native implementation details used by the public Python package.
#[pymodule]
mod _pymab {
    #[pymodule_export]
    use super::{core_version, native_available, rng_scheme_version};

    #[pymodule_export]
    use crate::environment::NativeEnvironment;

    #[pymodule_export]
    use crate::experiment::{NativeExperiment, NativeExperimentResult};

    #[pymodule_export]
    use crate::policy::NativePolicy;
}
