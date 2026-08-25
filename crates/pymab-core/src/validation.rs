//! Validation helpers shared by public Rust APIs.

use crate::error::{PyMabError, Result};
use crate::types::RewardDomain;

/// Return a finite value or a typed validation error.
pub fn finite(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyMabError::validation(name, "must be finite"))
    }
}

/// Return a finite, strictly positive configuration value.
pub fn strictly_positive(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(PyMabError::configuration(
            name,
            "must be finite and greater than zero",
        ))
    }
}

/// Return a finite probability in the inclusive interval `[0, 1]`.
pub fn probability(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(PyMabError::configuration(
            name,
            "must be between zero and one inclusive",
        ))
    }
}

/// Validate a reward against a mathematical domain.
pub fn reward(name: &str, value: f64, domain: RewardDomain) -> Result<f64> {
    finite(name, value)?;
    let valid = match domain {
        RewardDomain::Real => true,
        RewardDomain::UnitInterval => (0.0..=1.0).contains(&value),
        RewardDomain::Binary => value == 0.0 || value == 1.0,
    };
    if valid {
        Ok(value)
    } else {
        Err(PyMabError::validation(
            name,
            match domain {
                RewardDomain::Real => "must be a finite real number",
                RewardDomain::UnitInterval => "must be between zero and one inclusive",
                RewardDomain::Binary => "must be binary (exactly zero or one)",
            },
        ))
    }
}
