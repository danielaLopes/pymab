//! Shared domain types used by policies and experiments.

use crate::error::{PyMabError, Result};
use crate::validation::finite;

/// Mathematical support required by an environment or policy.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum RewardDomain {
    /// Any finite real value.
    Real,
    /// A finite value in the inclusive interval `[0, 1]`.
    UnitInterval,
    /// Exactly `0.0` or `1.0`.
    Binary,
}

/// Primary objective optimized by a policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum PolicyObjective {
    /// Maximize cumulative reward over the horizon.
    #[default]
    CumulativeReward,
    /// Identify the best arm at the end of the experiment.
    BestArm,
}

/// Static compatibility information for a policy implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PolicyCapabilities {
    contextual: bool,
    reward_domains: &'static [RewardDomain],
    objective: PolicyObjective,
}

impl PolicyCapabilities {
    /// Construct policy capability metadata.
    #[must_use]
    pub const fn new(
        contextual: bool,
        reward_domains: &'static [RewardDomain],
        objective: PolicyObjective,
    ) -> Self {
        Self {
            contextual,
            reward_domains,
            objective,
        }
    }

    /// Return whether contexts are required.
    #[must_use]
    pub const fn contextual(self) -> bool {
        self.contextual
    }

    /// Return the policy objective.
    #[must_use]
    pub const fn objective(self) -> PolicyObjective {
        self.objective
    }

    /// Return the supported reward domains.
    #[must_use]
    pub const fn reward_domains(self) -> &'static [RewardDomain] {
        self.reward_domains
    }

    /// Return whether a reward domain is supported.
    #[must_use]
    pub fn supports(self, domain: RewardDomain) -> bool {
        self.reward_domains.contains(&domain)
    }
}

/// A checked zero-based arm index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ActionIndex(usize);

impl ActionIndex {
    /// Validate an index for a policy with `n_arms` arms.
    pub fn new(index: usize, n_arms: usize) -> Result<Self> {
        if n_arms == 0 {
            return Err(PyMabError::configuration(
                "n_arms",
                "must be greater than zero",
            ));
        }
        if index >= n_arms {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {n_arms})"),
            ));
        }
        Ok(Self(index))
    }

    /// Return the underlying zero-based index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Shape of an arm-by-feature contextual observation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ContextShape {
    n_arms: usize,
    n_features: usize,
    element_count: usize,
}

impl ContextShape {
    /// Construct a non-empty context shape.
    pub fn new(n_arms: usize, n_features: usize) -> Result<Self> {
        if n_arms == 0 {
            return Err(PyMabError::configuration(
                "n_arms",
                "must be greater than zero",
            ));
        }
        if n_features == 0 {
            return Err(PyMabError::configuration(
                "n_features",
                "must be greater than zero",
            ));
        }
        let element_count = n_arms.checked_mul(n_features).ok_or_else(|| {
            PyMabError::configuration("context_shape", "element count overflows usize")
        })?;
        Ok(Self {
            n_arms,
            n_features,
            element_count,
        })
    }

    /// Return the number of arms.
    #[must_use]
    pub const fn n_arms(self) -> usize {
        self.n_arms
    }

    /// Return the number of features per arm.
    #[must_use]
    pub const fn n_features(self) -> usize {
        self.n_features
    }

    /// Return the required length of a contiguous context buffer.
    #[must_use]
    pub const fn element_count(self) -> usize {
        self.element_count
    }

    /// Validate a row-major contiguous context buffer.
    pub fn validate_flat(self, values: &[f64]) -> Result<()> {
        if values.len() != self.element_count {
            return Err(PyMabError::validation(
                "context",
                format!(
                    "expected {} values for shape ({}, {}), received {}",
                    self.element_count,
                    self.n_arms,
                    self.n_features,
                    values.len()
                ),
            ));
        }
        for &value in values {
            finite("context", value)?;
        }
        Ok(())
    }
}
