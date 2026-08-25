//! Built-in reward distributions and priors for arm means.

use rand::Rng;
use rand_distr::{Beta, Distribution, Normal, StandardNormal};

use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::RewardDomain;
use crate::validation::{finite, reward, strictly_positive};

/// Distribution of observed rewards conditional on their true means.
pub trait RewardModel {
    /// Mathematical support of observations produced by this model.
    fn domain(&self) -> RewardDomain;

    /// Validate a non-empty collection of true arm means.
    fn validate_means(&self, means: &[f64]) -> Result<()> {
        if means.is_empty() {
            return Err(PyMabError::validation(
                "reward means",
                "must be non-empty and finite",
            ));
        }
        for &mean in means {
            finite("reward means", mean)?;
        }
        Ok(())
    }

    /// Sample one potential reward for every supplied mean.
    fn sample(&self, means: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>>;

    /// Sample one scalar reward.
    fn sample_one(&self, mean: f64, rng: &mut NativeRng) -> Result<f64> {
        self.sample(&[mean], rng)?
            .into_iter()
            .next()
            .ok_or_else(|| PyMabError::internal("reward sampling returned no value"))
    }
}

/// Gaussian observations with known standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianReward {
    std: f64,
}

impl GaussianReward {
    /// Construct a Gaussian reward model.
    pub fn new(std: f64) -> Result<Self> {
        strictly_positive("std", std)?;
        Ok(Self { std })
    }

    /// Return the observation standard deviation.
    #[must_use]
    pub const fn std(self) -> f64 {
        self.std
    }
}

impl RewardModel for GaussianReward {
    fn domain(&self) -> RewardDomain {
        RewardDomain::Real
    }

    fn sample(&self, means: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.validate_means(means)?;
        Ok(means
            .iter()
            .map(|mean| {
                let noise: f64 = StandardNormal.sample(rng);
                mean + self.std * noise
            })
            .collect())
    }
}

/// Bernoulli observations whose means are success probabilities.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BernoulliReward;

impl BernoulliReward {
    /// Construct a Bernoulli reward model.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl RewardModel for BernoulliReward {
    fn domain(&self) -> RewardDomain {
        RewardDomain::Binary
    }

    fn validate_means(&self, means: &[f64]) -> Result<()> {
        if means.is_empty() {
            return Err(PyMabError::validation(
                "reward means",
                "must be non-empty and finite",
            ));
        }
        for &mean in means {
            reward(
                "Bernoulli arm probabilities",
                mean,
                RewardDomain::UnitInterval,
            )?;
        }
        Ok(())
    }

    fn sample(&self, means: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.validate_means(means)?;
        Ok(means
            .iter()
            .map(|&mean| f64::from(rng.random::<f64>() < mean))
            .collect())
    }
}

/// Uniform observations centered on each true arm mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformReward {
    half_width: f64,
}

impl UniformReward {
    /// Construct a centered uniform reward model.
    pub fn new(half_width: f64) -> Result<Self> {
        if !half_width.is_finite() || half_width < 0.0 {
            return Err(PyMabError::configuration(
                "half_width",
                "must be finite and non-negative",
            ));
        }
        Ok(Self { half_width })
    }

    /// Return the half-width of the sampling interval.
    #[must_use]
    pub const fn half_width(self) -> f64 {
        self.half_width
    }
}

impl RewardModel for UniformReward {
    fn domain(&self) -> RewardDomain {
        RewardDomain::Real
    }

    fn sample(&self, means: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.validate_means(means)?;
        if self.half_width == 0.0 {
            return Ok(means.to_vec());
        }
        Ok(means
            .iter()
            .map(|mean| rng.random_range(mean - self.half_width..mean + self.half_width))
            .collect())
    }
}

/// Monomorphic dispatch over built-in reward models.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum BuiltInRewardModel {
    /// Gaussian observations.
    Gaussian(GaussianReward),
    /// Bernoulli observations.
    Bernoulli(BernoulliReward),
    /// Uniform observations.
    Uniform(UniformReward),
}

impl RewardModel for BuiltInRewardModel {
    fn domain(&self) -> RewardDomain {
        match self {
            Self::Gaussian(model) => model.domain(),
            Self::Bernoulli(model) => model.domain(),
            Self::Uniform(model) => model.domain(),
        }
    }

    fn validate_means(&self, means: &[f64]) -> Result<()> {
        match self {
            Self::Gaussian(model) => model.validate_means(means),
            Self::Bernoulli(model) => model.validate_means(means),
            Self::Uniform(model) => model.validate_means(means),
        }
    }

    fn sample(&self, means: &[f64], rng: &mut NativeRng) -> Result<Vec<f64>> {
        match self {
            Self::Gaussian(model) => model.sample(means, rng),
            Self::Bernoulli(model) => model.sample(means, rng),
            Self::Uniform(model) => model.sample(means, rng),
        }
    }
}

/// Distribution used to generate initial true arm means.
pub trait ArmPrior {
    /// Generate one true mean per arm.
    fn sample(&self, n_arms: usize, rng: &mut NativeRng) -> Result<Vec<f64>>;
}

fn validate_n_arms(n_arms: usize) -> Result<()> {
    if n_arms == 0 {
        Err(PyMabError::configuration(
            "n_arms",
            "must be greater than zero",
        ))
    } else {
        Ok(())
    }
}

/// Gaussian prior for initial arm means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianArmPrior {
    mean: f64,
    std: f64,
}

impl GaussianArmPrior {
    /// Construct a Gaussian arm prior. A zero standard deviation is allowed.
    pub fn new(mean: f64, std: f64) -> Result<Self> {
        finite("mean", mean).map_err(|_| PyMabError::configuration("mean", "must be finite"))?;
        if !std.is_finite() || std < 0.0 {
            return Err(PyMabError::configuration(
                "std",
                "must be finite and non-negative",
            ));
        }
        Ok(Self { mean, std })
    }
}

impl ArmPrior for GaussianArmPrior {
    fn sample(&self, n_arms: usize, rng: &mut NativeRng) -> Result<Vec<f64>> {
        validate_n_arms(n_arms)?;
        if self.std == 0.0 {
            return Ok(vec![self.mean; n_arms]);
        }
        let distribution = Normal::new(self.mean, self.std).map_err(|error| {
            PyMabError::configuration("std", format!("invalid Gaussian scale: {error}"))
        })?;
        Ok(distribution.sample_iter(rng).take(n_arms).collect())
    }
}

/// Beta prior for initial arm probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetaArmPrior {
    alpha: f64,
    beta: f64,
}

impl BetaArmPrior {
    /// Construct a Beta arm prior.
    pub fn new(alpha: f64, beta: f64) -> Result<Self> {
        strictly_positive("alpha", alpha)?;
        strictly_positive("beta", beta)?;
        Ok(Self { alpha, beta })
    }
}

impl ArmPrior for BetaArmPrior {
    fn sample(&self, n_arms: usize, rng: &mut NativeRng) -> Result<Vec<f64>> {
        validate_n_arms(n_arms)?;
        let distribution = Beta::new(self.alpha, self.beta)
            .map_err(|error| PyMabError::configuration("alpha and beta", error.to_string()))?;
        Ok(distribution.sample_iter(rng).take(n_arms).collect())
    }
}

/// Uniform prior for initial arm means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniformArmPrior {
    low: f64,
    high: f64,
}

impl UniformArmPrior {
    /// Construct a uniform arm prior over an inclusive configuration interval.
    pub fn new(low: f64, high: f64) -> Result<Self> {
        if !low.is_finite() || !high.is_finite() {
            return Err(PyMabError::configuration("low and high", "must be finite"));
        }
        if low > high {
            return Err(PyMabError::configuration("low", "must be <= high"));
        }
        Ok(Self { low, high })
    }
}

impl ArmPrior for UniformArmPrior {
    fn sample(&self, n_arms: usize, rng: &mut NativeRng) -> Result<Vec<f64>> {
        validate_n_arms(n_arms)?;
        if self.low == self.high {
            return Ok(vec![self.low; n_arms]);
        }
        Ok((0..n_arms)
            .map(|_| rng.random_range(self.low..self.high))
            .collect())
    }
}
