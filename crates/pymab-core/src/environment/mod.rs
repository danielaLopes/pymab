//! Built-in classic and contextual bandit environments.

pub mod classic;
pub mod contextual;
pub mod dynamics;

pub use classic::BanditEnvironment;
pub use contextual::ContextProvider;
pub use dynamics::EnvironmentDynamics;

use crate::distribution::RewardModel;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{ContextShape, RewardDomain};
use contextual::{
    BuiltInContextProvider, LinearContextualEnvironment, LogisticContextualEnvironment,
};

/// Monomorphic dispatch over environments supported by the native runner.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BuiltInEnvironment {
    /// Classic non-contextual environment.
    Classic(BanditEnvironment),
    /// Linear contextual environment.
    Linear(LinearContextualEnvironment<BuiltInContextProvider>),
    /// Logistic contextual environment.
    Logistic(LogisticContextualEnvironment<BuiltInContextProvider>),
}

impl BuiltInEnvironment {
    /// Return whether the environment requires contextual observations.
    #[must_use]
    pub fn contextual(&self) -> bool {
        !matches!(self, Self::Classic(_))
    }

    /// Return the number of arms.
    #[must_use]
    pub fn n_arms(&self) -> usize {
        match self {
            Self::Classic(value) => value.n_arms(),
            Self::Linear(value) => value.shape().n_arms(),
            Self::Logistic(value) => value.shape().n_arms(),
        }
    }

    /// Return the context shape for contextual environments.
    #[must_use]
    pub fn context_shape(&self) -> Option<ContextShape> {
        match self {
            Self::Classic(_) => None,
            Self::Linear(value) => Some(value.shape()),
            Self::Logistic(value) => Some(value.shape()),
        }
    }

    /// Return the reward domain.
    #[must_use]
    pub fn reward_domain(&self) -> RewardDomain {
        match self {
            Self::Classic(value) => value.reward_domain(),
            Self::Linear(value) => value.reward_model().domain(),
            Self::Logistic(value) => value.reward_model().domain(),
        }
    }

    /// Advance classic dynamics. Contextual environments are stateless here.
    pub fn advance(&mut self, step: u64, rng: &mut NativeRng) -> Result<()> {
        match self {
            Self::Classic(value) => value.advance(step, rng),
            Self::Linear(_) | Self::Logistic(_) => Err(PyMabError::compatibility(
                "environment",
                "contextual environments do not expose classic dynamics",
            )),
        }
    }

    /// Sample one context matrix.
    pub fn context(&self, rng: &mut NativeRng) -> Result<Vec<f64>> {
        match self {
            Self::Classic(_) => Err(PyMabError::compatibility(
                "environment",
                "classic environments do not produce context",
            )),
            Self::Linear(value) => value.context(rng),
            Self::Logistic(value) => value.context(rng),
        }
    }

    /// Compute one expected reward per arm.
    pub fn expected_rewards(&self, context: Option<&[f64]>) -> Result<Vec<f64>> {
        match (self, context) {
            (Self::Classic(value), None) => Ok(value.expected_rewards().to_vec()),
            (Self::Classic(_), Some(_)) => Err(PyMabError::compatibility(
                "context",
                "classic environments do not accept context",
            )),
            (Self::Linear(_), None) | (Self::Logistic(_), None) => Err(PyMabError::compatibility(
                "context",
                "contextual environment requires context",
            )),
            (Self::Linear(value), Some(context)) => value.expected_rewards(context),
            (Self::Logistic(value), Some(context)) => value.expected_rewards(context),
        }
    }

    /// Sample one potential reward per arm.
    pub fn sample_rewards(&self, context: Option<&[f64]>, rng: &mut NativeRng) -> Result<Vec<f64>> {
        match (self, context) {
            (Self::Classic(value), None) => value.sample_rewards(rng),
            (Self::Classic(_), Some(_)) => Err(PyMabError::compatibility(
                "context",
                "classic environments do not accept context",
            )),
            (Self::Linear(_), None) | (Self::Logistic(_), None) => Err(PyMabError::compatibility(
                "context",
                "contextual environment requires context",
            )),
            (Self::Linear(value), Some(context)) => value.sample_rewards(context, rng),
            (Self::Logistic(value), Some(context)) => value.sample_rewards(context, rng),
        }
    }
}
