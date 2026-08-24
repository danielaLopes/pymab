//! Typed errors returned by the Rust core.

use std::error::Error;
use std::fmt::{self, Display, Formatter};

/// Stable high-level categories for errors returned by PyMAB.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ErrorCode {
    /// A policy or experiment configuration is invalid.
    Configuration,
    /// Runtime input violates a public data contract.
    Validation,
    /// Individually valid components cannot be used together.
    Compatibility,
    /// A numerical operation could not produce a valid result.
    Numerical,
    /// An invariant inside PyMAB was violated.
    Internal,
}

/// Errors returned by public Rust APIs.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum PyMabError {
    /// A named configuration value is invalid.
    Configuration { field: String, message: String },
    /// A named runtime value is invalid.
    Validation { field: String, message: String },
    /// Two components cannot be combined.
    Compatibility { component: String, message: String },
    /// A numerical operation failed safely.
    Numerical { operation: String, message: String },
    /// An internal invariant was violated.
    Internal { message: String },
}

impl PyMabError {
    /// Construct a configuration error.
    pub fn configuration(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Configuration {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Construct a validation error.
    pub fn validation(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Construct a compatibility error.
    pub fn compatibility(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Compatibility {
            component: component.into(),
            message: message.into(),
        }
    }

    /// Construct a numerical error.
    pub fn numerical(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Numerical {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Construct an internal error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Return the stable high-level category for this error.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::Configuration { .. } => ErrorCode::Configuration,
            Self::Validation { .. } => ErrorCode::Validation,
            Self::Compatibility { .. } => ErrorCode::Compatibility,
            Self::Numerical { .. } => ErrorCode::Numerical,
            Self::Internal { .. } => ErrorCode::Internal,
        }
    }
}

impl Display for PyMabError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration { field, message } => {
                write!(formatter, "invalid configuration `{field}`: {message}")
            }
            Self::Validation { field, message } => {
                write!(formatter, "invalid `{field}`: {message}")
            }
            Self::Compatibility { component, message } => {
                write!(formatter, "incompatible `{component}`: {message}")
            }
            Self::Numerical { operation, message } => {
                write!(
                    formatter,
                    "numerical failure during `{operation}`: {message}"
                )
            }
            Self::Internal { message } => write!(formatter, "internal PyMAB error: {message}"),
        }
    }
}

impl Error for PyMabError {}

/// Result type used by the Rust core.
pub type Result<T> = std::result::Result<T, PyMabError>;
