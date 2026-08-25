//! Mapping from typed core failures to stable Python exception families.

use pyo3::exceptions::{PyArithmeticError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::PyErr;

pub(crate) fn to_python(error: pymab::error::PyMabError) -> PyErr {
    let message = error.to_string();
    match error {
        pymab::error::PyMabError::Configuration { .. }
        | pymab::error::PyMabError::Validation { .. } => PyValueError::new_err(message),
        pymab::error::PyMabError::Compatibility { .. } => PyTypeError::new_err(message),
        pymab::error::PyMabError::Numerical { .. } => PyArithmeticError::new_err(message),
        pymab::error::PyMabError::Internal { .. } => PyRuntimeError::new_err(message),
        _ => PyRuntimeError::new_err(message),
    }
}
