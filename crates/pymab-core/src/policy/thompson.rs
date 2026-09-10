//! Stationary Thompson-sampling policies.

use std::mem::size_of;

use rand_distr::{Beta, Distribution, Normal};

use super::action_value::{deterministic_argmax, ActionValueState};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{
    ActionIndex, PolicyCapabilities, PolicyObjective, RewardDomain, ALL_REWARD_DOMAINS,
};
use crate::validation::{finite, reward, strictly_positive};

const BINARY_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary];
const BINARY_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, BINARY_DOMAINS, PolicyObjective::CumulativeReward);
const GENERAL_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Beta-Bernoulli learned state shared by Thompson sampling and Bayesian UCB.
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliPosteriorState {
    action_values: ActionValueState,
    successes: Vec<u64>,
    failures: Vec<u64>,
}

impl BernoulliPosteriorState {
    pub(crate) fn new(n_arms: usize) -> Result<Self> {
        Ok(Self {
            action_values: ActionValueState::new(n_arms, 0.0)?,
            successes: vec![0; n_arms],
            failures: vec![0; n_arms],
        })
    }

    pub(crate) fn update(&mut self, action: ActionIndex, observed_reward: f64) -> Result<()> {
        let observed_reward = reward("reward", observed_reward, RewardDomain::Binary)?;
        self.action_values.update(action, observed_reward)?;
        if observed_reward == 1.0 {
            self.successes[action.get()] = self.successes[action.get()]
                .checked_add(1)
                .ok_or_else(|| PyMabError::internal("success counter overflowed"))?;
        } else {
            self.failures[action.get()] = self.failures[action.get()]
                .checked_add(1)
                .ok_or_else(|| PyMabError::internal("failure counter overflowed"))?;
        }
        Ok(())
    }

    pub(crate) fn reset(&mut self) {
        self.action_values.reset();
        self.successes.fill(0);
        self.failures.fill(0);
    }

    /// Return common action-value state.
    #[must_use]
    pub const fn action_values(&self) -> &ActionValueState {
        &self.action_values
    }

    /// Return observed successes by arm.
    #[must_use]
    pub fn successes(&self) -> &[u64] {
        &self.successes
    }

    /// Return observed failures by arm.
    #[must_use]
    pub fn failures(&self) -> &[u64] {
        &self.failures
    }

    pub(crate) fn estimated_heap_bytes(&self) -> usize {
        self.action_values.estimated_heap_bytes()
            + (self.successes.capacity() + self.failures.capacity()) * size_of::<u64>()
    }
}

/// Gaussian conjugate-posterior state shared by Thompson sampling and Bayesian UCB.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianPosteriorState {
    prior_mean: f64,
    prior_precision: f64,
    action_values: ActionValueState,
    means: Vec<f64>,
    precisions: Vec<f64>,
}

impl GaussianPosteriorState {
    pub(crate) fn new(n_arms: usize, prior_mean: f64, prior_precision: f64) -> Result<Self> {
        if !prior_mean.is_finite() {
            return Err(PyMabError::configuration("prior_mean", "must be finite"));
        }
        let prior_precision = strictly_positive("prior_precision", prior_precision)?;
        Ok(Self {
            prior_mean,
            prior_precision,
            action_values: ActionValueState::new(n_arms, prior_mean)?,
            means: vec![prior_mean; n_arms],
            precisions: vec![prior_precision; n_arms],
        })
    }

    pub(crate) fn update(
        &mut self,
        action: ActionIndex,
        observed_reward: f64,
        reward_precision: f64,
    ) -> Result<()> {
        finite("reward", observed_reward)?;
        let index = action.get();
        if index >= self.means.len() {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {})", self.means.len()),
            ));
        }
        let precision = self.precisions[index] + reward_precision;
        let mean = (self.precisions[index] * self.means[index]
            + reward_precision * observed_reward)
            / precision;
        if !precision.is_finite() || !mean.is_finite() {
            return Err(PyMabError::numerical(
                "Gaussian posterior update",
                "posterior parameters became non-finite",
            ));
        }
        self.action_values.update(action, observed_reward)?;
        self.precisions[index] = precision;
        self.means[index] = mean;
        self.action_values.set_estimate(action, mean)
    }

    pub(crate) fn reset(&mut self) {
        self.action_values.reset();
        self.means.fill(self.prior_mean);
        self.precisions.fill(self.prior_precision);
    }

    /// Return common action-value state.
    #[must_use]
    pub const fn action_values(&self) -> &ActionValueState {
        &self.action_values
    }

    /// Return posterior means by arm.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Return posterior precisions by arm.
    #[must_use]
    pub fn precisions(&self) -> &[f64] {
        &self.precisions
    }

    pub(crate) fn estimated_heap_bytes(&self) -> usize {
        self.action_values.estimated_heap_bytes()
            + (self.means.capacity() + self.precisions.capacity()) * size_of::<f64>()
    }
}

/// Thompson sampling with Beta-Bernoulli posteriors.
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliThompsonSamplingPolicy {
    alpha_prior: f64,
    beta_prior: f64,
    state: BernoulliPosteriorState,
}

impl BernoulliThompsonSamplingPolicy {
    /// Construct a Beta-Bernoulli Thompson-sampling policy.
    pub fn new(n_arms: usize, alpha_prior: f64, beta_prior: f64) -> Result<Self> {
        Ok(Self {
            alpha_prior: strictly_positive("alpha_prior", alpha_prior)?,
            beta_prior: strictly_positive("beta_prior", beta_prior)?,
            state: BernoulliPosteriorState::new(n_arms)?,
        })
    }
}

impl Policy for BernoulliThompsonSamplingPolicy {
    type State = BernoulliPosteriorState;

    fn capabilities(&self) -> PolicyCapabilities {
        BINARY_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.successes.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let samples: Result<Vec<f64>> = self
            .state
            .successes
            .iter()
            .zip(&self.state.failures)
            .map(|(successes, failures)| {
                let distribution = Beta::new(
                    self.alpha_prior + *successes as f64,
                    self.beta_prior + *failures as f64,
                )
                .map_err(|error| PyMabError::numerical("Beta sampling", error.to_string()))?;
                Ok(distribution.sample(rng))
            })
            .collect();
        deterministic_argmax(&samples?)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.update(action, reward)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.action_values.recommendation()
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

/// Thompson sampling for Gaussian rewards with known observation precision.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianThompsonSamplingPolicy {
    reward_precision: f64,
    state: GaussianPosteriorState,
}

impl GaussianThompsonSamplingPolicy {
    /// Construct a conjugate Gaussian Thompson-sampling policy.
    pub fn new(
        n_arms: usize,
        prior_mean: f64,
        prior_precision: f64,
        reward_precision: f64,
    ) -> Result<Self> {
        Ok(Self {
            reward_precision: strictly_positive("reward_precision", reward_precision)?,
            state: GaussianPosteriorState::new(n_arms, prior_mean, prior_precision)?,
        })
    }
}

impl Policy for GaussianThompsonSamplingPolicy {
    type State = GaussianPosteriorState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.means.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let samples: Result<Vec<f64>> = self
            .state
            .means
            .iter()
            .zip(&self.state.precisions)
            .map(|(mean, precision)| {
                let distribution = Normal::new(*mean, 1.0 / precision.sqrt()).map_err(|error| {
                    PyMabError::numerical("Gaussian sampling", error.to_string())
                })?;
                Ok(distribution.sample(rng))
            })
            .collect();
        deterministic_argmax(&samples?)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.update(action, reward, self.reward_precision)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.action_values.recommendation()
    }

    fn reset(&mut self) {
        self.state.reset();
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}
