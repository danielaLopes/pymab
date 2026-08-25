//! Support-aware evolution strategies for true arm means.

use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::RewardDomain;
use crate::validation::probability;

/// Strategy for evolving true arm means.
pub trait EnvironmentDynamics {
    /// Return whether this strategy supports a reward domain.
    fn supports(&self, domain: RewardDomain) -> bool;

    /// Return evolved means for one step without mutating the input.
    fn apply(&self, means: &[f64], step: u64, rng: &mut NativeRng) -> Result<Vec<f64>>;
}

/// Dynamics that leave arm means unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StationaryDynamics;

impl EnvironmentDynamics for StationaryDynamics {
    fn supports(&self, _domain: RewardDomain) -> bool {
        true
    }

    fn apply(&self, means: &[f64], _step: u64, _rng: &mut NativeRng) -> Result<Vec<f64>> {
        Ok(means.to_vec())
    }
}

/// Gaussian random-walk drift for real-valued means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradualDrift {
    std: f64,
}

impl GradualDrift {
    /// Construct gradual drift. A zero standard deviation is allowed.
    pub fn new(std: f64) -> Result<Self> {
        if !std.is_finite() || std < 0.0 {
            return Err(PyMabError::configuration(
                "std",
                "must be finite and non-negative",
            ));
        }
        Ok(Self { std })
    }

    fn shift(self, means: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        if self.std == 0.0 {
            return Ok(means.to_vec());
        }
        Ok(means
            .iter()
            .map(|mean| {
                let noise: f64 = StandardNormal.sample(rng);
                mean + self.std * noise
            })
            .collect())
    }
}

impl EnvironmentDynamics for GradualDrift {
    fn supports(&self, domain: RewardDomain) -> bool {
        domain == RewardDomain::Real
    }

    fn apply(&self, means: &[f64], _step: u64, rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.shift(means, rng)
    }
}

/// Periodic Gaussian shifts for real-valued means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbruptShift {
    frequency: u64,
    drift: GradualDrift,
    shift_at_step_zero: bool,
}

impl AbruptShift {
    /// Construct periodic abrupt shifts.
    pub fn new(frequency: u64, std: f64, shift_at_step_zero: bool) -> Result<Self> {
        if frequency == 0 {
            return Err(PyMabError::configuration(
                "frequency",
                "must be greater than zero",
            ));
        }
        Ok(Self {
            frequency,
            drift: GradualDrift::new(std)?,
            shift_at_step_zero,
        })
    }
}

impl EnvironmentDynamics for AbruptShift {
    fn supports(&self, domain: RewardDomain) -> bool {
        domain == RewardDomain::Real
    }

    fn apply(&self, means: &[f64], step: u64, rng: &mut NativeRng) -> Result<Vec<f64>> {
        if (step == 0 && !self.shift_at_step_zero) || step % self.frequency != 0 {
            return Ok(means.to_vec());
        }
        self.drift.shift(means, rng)
    }
}

/// Gaussian random walk in log-odds space for probability means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbabilityDrift {
    logit_std: f64,
    epsilon: f64,
}

impl ProbabilityDrift {
    /// Construct probability drift.
    pub fn new(logit_std: f64, epsilon: f64) -> Result<Self> {
        if !logit_std.is_finite() || logit_std < 0.0 {
            return Err(PyMabError::configuration(
                "logit_std",
                "must be finite and non-negative",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 || epsilon >= 0.5 {
            return Err(PyMabError::configuration("epsilon", "must be in (0, 0.5)"));
        }
        Ok(Self { logit_std, epsilon })
    }
}

impl EnvironmentDynamics for ProbabilityDrift {
    fn supports(&self, domain: RewardDomain) -> bool {
        matches!(domain, RewardDomain::Binary | RewardDomain::UnitInterval)
    }

    fn apply(&self, means: &[f64], _step: u64, rng: &mut NativeRng) -> Result<Vec<f64>> {
        means
            .iter()
            .map(|&mean| {
                probability("means", mean)
                    .map_err(|error| PyMabError::validation("means", error.to_string()))?;
                let clipped = mean.clamp(self.epsilon, 1.0 - self.epsilon);
                let mut logit = (clipped / (1.0 - clipped)).ln();
                if self.logit_std != 0.0 {
                    let noise: f64 = StandardNormal.sample(rng);
                    logit += self.logit_std * noise;
                }
                Ok(1.0 / (1.0 + (-logit).exp()))
            })
            .collect()
    }
}

/// Randomly permute arm identities with a configured probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomArmSwap {
    probability: f64,
}

impl RandomArmSwap {
    /// Construct random arm swapping.
    pub fn new(probability_value: f64) -> Result<Self> {
        probability("probability", probability_value)?;
        Ok(Self {
            probability: probability_value,
        })
    }
}

impl EnvironmentDynamics for RandomArmSwap {
    fn supports(&self, _domain: RewardDomain) -> bool {
        true
    }

    fn apply(&self, means: &[f64], _step: u64, rng: &mut NativeRng) -> Result<Vec<f64>> {
        let mut values = means.to_vec();
        if rng.random::<f64>() < self.probability {
            values.shuffle(rng);
        }
        Ok(values)
    }
}

/// Monomorphic dispatch over built-in environment dynamics.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum BuiltInDynamics {
    /// Stationary means.
    Stationary(StationaryDynamics),
    /// Gradual Gaussian drift.
    Gradual(GradualDrift),
    /// Periodic abrupt shifts.
    Abrupt(AbruptShift),
    /// Probability-space drift.
    Probability(ProbabilityDrift),
    /// Random arm permutations.
    RandomSwap(RandomArmSwap),
}

impl Default for BuiltInDynamics {
    fn default() -> Self {
        Self::Stationary(StationaryDynamics)
    }
}

impl EnvironmentDynamics for BuiltInDynamics {
    fn supports(&self, domain: RewardDomain) -> bool {
        match self {
            Self::Stationary(value) => value.supports(domain),
            Self::Gradual(value) => value.supports(domain),
            Self::Abrupt(value) => value.supports(domain),
            Self::Probability(value) => value.supports(domain),
            Self::RandomSwap(value) => value.supports(domain),
        }
    }

    fn apply(&self, means: &[f64], step: u64, rng: &mut NativeRng) -> Result<Vec<f64>> {
        match self {
            Self::Stationary(value) => value.apply(means, step, rng),
            Self::Gradual(value) => value.apply(means, step, rng),
            Self::Abrupt(value) => value.apply(means, step, rng),
            Self::Probability(value) => value.apply(means, step, rng),
            Self::RandomSwap(value) => value.apply(means, step, rng),
        }
    }
}
