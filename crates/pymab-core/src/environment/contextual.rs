//! Built-in context providers and contextual environments.

use rand_distr::{Distribution, Normal};

use crate::distribution::{BuiltInRewardModel, RewardModel};
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{ContextShape, RewardDomain};
use crate::validation::finite;

/// Cloneable source of row-major contextual feature matrices.
pub trait ContextProvider {
    /// Context matrix shape.
    fn shape(&self) -> ContextShape;

    /// Sample one row-major context matrix.
    fn sample(&self, rng: &mut NativeRng) -> Result<Vec<f64>>;
}

/// Deterministic context provider.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedContextProvider {
    shape: ContextShape,
    values: Vec<f64>,
}

impl FixedContextProvider {
    /// Construct a fixed provider from one shared feature row or a full matrix.
    pub fn new(n_arms: usize, n_features: usize, values: Vec<f64>) -> Result<Self> {
        let shape = ContextShape::new(n_arms, n_features)?;
        let values = if values.len() == n_features {
            values.repeat(n_arms)
        } else {
            values
        };
        shape.validate_flat(&values)?;
        Ok(Self { shape, values })
    }

    /// Return fixed row-major values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl ContextProvider for FixedContextProvider {
    fn shape(&self) -> ContextShape {
        self.shape
    }

    fn sample(&self, _rng: &mut NativeRng) -> Result<Vec<f64>> {
        Ok(self.values.clone())
    }
}

/// Independent Gaussian context provider.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianContextProvider {
    shape: ContextShape,
    mean: f64,
    std: f64,
}

impl GaussianContextProvider {
    /// Construct a Gaussian context provider.
    pub fn new(n_arms: usize, n_features: usize, mean: f64, std: f64) -> Result<Self> {
        let shape = ContextShape::new(n_arms, n_features)?;
        if !mean.is_finite() {
            return Err(PyMabError::configuration("mean", "must be finite"));
        }
        if !std.is_finite() || std < 0.0 {
            return Err(PyMabError::configuration(
                "std",
                "must be finite and non-negative",
            ));
        }
        Ok(Self { shape, mean, std })
    }
}

impl ContextProvider for GaussianContextProvider {
    fn shape(&self) -> ContextShape {
        self.shape
    }

    fn sample(&self, rng: &mut NativeRng) -> Result<Vec<f64>> {
        if self.std == 0.0 {
            return Ok(vec![self.mean; self.shape.element_count()]);
        }
        let distribution = Normal::new(self.mean, self.std).map_err(|error| {
            PyMabError::configuration("std", format!("invalid Gaussian scale: {error}"))
        })?;
        Ok(distribution
            .sample_iter(rng)
            .take(self.shape.element_count())
            .collect())
    }
}

/// Monomorphic dispatch over built-in context providers.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BuiltInContextProvider {
    /// Deterministic fixed contexts.
    Fixed(FixedContextProvider),
    /// Independent Gaussian contexts.
    Gaussian(GaussianContextProvider),
}

impl ContextProvider for BuiltInContextProvider {
    fn shape(&self) -> ContextShape {
        match self {
            Self::Fixed(value) => value.shape(),
            Self::Gaussian(value) => value.shape(),
        }
    }

    fn sample(&self, rng: &mut NativeRng) -> Result<Vec<f64>> {
        match self {
            Self::Fixed(value) => value.sample(rng),
            Self::Gaussian(value) => value.sample(rng),
        }
    }
}

fn validate_theta(shape: ContextShape, theta: &[f64]) -> Result<()> {
    if theta.len() != shape.element_count() {
        return Err(PyMabError::configuration(
            "theta",
            format!(
                "expected {} values for shape ({}, {}), received {}",
                shape.element_count(),
                shape.n_arms(),
                shape.n_features(),
                theta.len()
            ),
        ));
    }
    for &value in theta {
        finite("theta", value)
            .map_err(|_| PyMabError::configuration("theta", "must contain only finite values"))?;
    }
    Ok(())
}

fn dot_rows(shape: ContextShape, context: &[f64], theta: &[f64]) -> Result<Vec<f64>> {
    shape.validate_flat(context)?;
    Ok(context
        .chunks_exact(shape.n_features())
        .zip(theta.chunks_exact(shape.n_features()))
        .map(|(context_row, theta_row)| {
            context_row
                .iter()
                .zip(theta_row)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect())
}

/// Contextual environment with linear expected rewards.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearContextualEnvironment<P: ContextProvider> {
    shape: ContextShape,
    theta: Vec<f64>,
    context_provider: P,
    reward_model: BuiltInRewardModel,
}

impl<P: ContextProvider> LinearContextualEnvironment<P> {
    /// Construct a linear contextual environment.
    pub fn new(
        n_arms: usize,
        n_features: usize,
        theta: Vec<f64>,
        context_provider: P,
        reward_model: BuiltInRewardModel,
    ) -> Result<Self> {
        let shape = ContextShape::new(n_arms, n_features)?;
        validate_theta(shape, &theta)?;
        if context_provider.shape() != shape {
            return Err(PyMabError::compatibility(
                "context_provider",
                "shape does not match the environment",
            ));
        }
        if reward_model.domain() == RewardDomain::Binary {
            return Err(PyMabError::compatibility(
                "reward_model",
                "use LogisticContextualEnvironment for binary rewards",
            ));
        }
        Ok(Self {
            shape,
            theta,
            context_provider,
            reward_model,
        })
    }

    /// Context shape.
    #[must_use]
    pub const fn shape(&self) -> ContextShape {
        self.shape
    }

    /// Sample one context matrix.
    pub fn context(&self, rng: &mut NativeRng) -> Result<Vec<f64>> {
        let context = self.context_provider.sample(rng)?;
        self.shape.validate_flat(&context)?;
        Ok(context)
    }

    /// Compute one expected reward per arm.
    pub fn expected_rewards(&self, context: &[f64]) -> Result<Vec<f64>> {
        let means = dot_rows(self.shape, context, &self.theta)?;
        self.reward_model.validate_means(&means)?;
        Ok(means)
    }

    /// Sample one potential reward per arm.
    pub fn sample_rewards(&self, context: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.reward_model
            .sample(&self.expected_rewards(context)?, rng)
    }

    /// Return model parameters in row-major order.
    #[must_use]
    pub fn theta(&self) -> &[f64] {
        &self.theta
    }
}

/// Contextual Bernoulli environment with a clipped logistic link.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticContextualEnvironment<P: ContextProvider> {
    shape: ContextShape,
    theta: Vec<f64>,
    context_provider: P,
    reward_model: BuiltInRewardModel,
}

impl<P: ContextProvider> LogisticContextualEnvironment<P> {
    /// Construct a logistic contextual environment.
    pub fn new(
        n_arms: usize,
        n_features: usize,
        theta: Vec<f64>,
        context_provider: P,
        reward_model: BuiltInRewardModel,
    ) -> Result<Self> {
        let shape = ContextShape::new(n_arms, n_features)?;
        validate_theta(shape, &theta)?;
        if context_provider.shape() != shape {
            return Err(PyMabError::compatibility(
                "context_provider",
                "shape does not match the environment",
            ));
        }
        if reward_model.domain() != RewardDomain::Binary {
            return Err(PyMabError::compatibility(
                "reward_model",
                "logistic environments require a binary reward model",
            ));
        }
        Ok(Self {
            shape,
            theta,
            context_provider,
            reward_model,
        })
    }

    /// Context shape.
    #[must_use]
    pub const fn shape(&self) -> ContextShape {
        self.shape
    }

    /// Sample one context matrix.
    pub fn context(&self, rng: &mut NativeRng) -> Result<Vec<f64>> {
        let context = self.context_provider.sample(rng)?;
        self.shape.validate_flat(&context)?;
        Ok(context)
    }

    /// Compute one Bernoulli probability per arm.
    pub fn expected_rewards(&self, context: &[f64]) -> Result<Vec<f64>> {
        Ok(dot_rows(self.shape, context, &self.theta)?
            .into_iter()
            .map(|logit| 1.0 / (1.0 + (-logit.clamp(-35.0, 35.0)).exp()))
            .collect())
    }

    /// Sample one potential binary reward per arm.
    pub fn sample_rewards(&self, context: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.reward_model
            .sample(&self.expected_rewards(context)?, rng)
    }

    /// Return model parameters in row-major order.
    #[must_use]
    pub fn theta(&self) -> &[f64] {
        &self.theta
    }
}
