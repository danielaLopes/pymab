//! Bayesian upper-confidence-bound policies.

use std::mem::size_of;

use statrs::distribution::{Beta, ContinuousCDF, Normal};

use super::action_value::deterministic_argmax;
use super::thompson::{BernoulliPosteriorState, GaussianPosteriorState};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{
    ActionIndex, PolicyCapabilities, PolicyObjective, RewardDomain, ALL_REWARD_DOMAINS,
};
use crate::validation::strictly_positive;

const BINARY_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary];
const BINARY_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, BINARY_DOMAINS, PolicyObjective::CumulativeReward);
const GENERAL_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

fn open_probability(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err(PyMabError::configuration(
            name,
            "must be finite and strictly between zero and one",
        ))
    }
}

/// Bayesian UCB using Beta posterior quantiles.
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliBayesianUCBPolicy {
    alpha_prior: f64,
    beta_prior: f64,
    quantile: f64,
    state: BernoulliPosteriorState,
}

impl BernoulliBayesianUCBPolicy {
    /// Construct a Beta-Bernoulli Bayesian-UCB policy.
    pub fn new(n_arms: usize, alpha_prior: f64, beta_prior: f64, quantile: f64) -> Result<Self> {
        Ok(Self {
            alpha_prior: strictly_positive("alpha_prior", alpha_prior)?,
            beta_prior: strictly_positive("beta_prior", beta_prior)?,
            quantile: open_probability("quantile", quantile)?,
            state: BernoulliPosteriorState::new(n_arms)?,
        })
    }

    /// Return Beta posterior upper bounds for every arm.
    pub fn upper_bounds(&self) -> Result<Vec<f64>> {
        self.state
            .successes()
            .iter()
            .zip(self.state.failures())
            .map(|(successes, failures)| {
                let distribution = Beta::new(
                    self.alpha_prior + *successes as f64,
                    self.beta_prior + *failures as f64,
                )
                .map_err(|error| PyMabError::numerical("Beta quantile", error.to_string()))?;
                let bound = distribution.inverse_cdf(self.quantile);
                if bound.is_finite() {
                    Ok(bound)
                } else {
                    Err(PyMabError::numerical(
                        "Beta quantile",
                        "quantile became non-finite",
                    ))
                }
            })
            .collect()
    }
}

impl Policy for BernoulliBayesianUCBPolicy {
    type State = BernoulliPosteriorState;

    fn capabilities(&self) -> PolicyCapabilities {
        BINARY_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.successes().len()
    }

    fn select_action(&mut self, _rng: &mut NativeRng) -> Result<ActionIndex> {
        deterministic_argmax(&self.upper_bounds()?)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.update(action, reward)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.action_values().recommendation()
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

/// Bayesian UCB for Gaussian rewards with known observation precision.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianBayesianUCBPolicy {
    reward_precision: f64,
    quantile: f64,
    state: GaussianPosteriorState,
}

impl GaussianBayesianUCBPolicy {
    /// Construct a conjugate Gaussian Bayesian-UCB policy.
    pub fn new(
        n_arms: usize,
        prior_mean: f64,
        prior_precision: f64,
        reward_precision: f64,
        quantile: f64,
    ) -> Result<Self> {
        Ok(Self {
            reward_precision: strictly_positive("reward_precision", reward_precision)?,
            quantile: open_probability("quantile", quantile)?,
            state: GaussianPosteriorState::new(n_arms, prior_mean, prior_precision)?,
        })
    }

    /// Return Gaussian posterior upper bounds for every arm.
    pub fn upper_bounds(&self) -> Result<Vec<f64>> {
        let standard_normal =
            Normal::new(0.0, 1.0).map_err(|error| PyMabError::internal(error.to_string()))?;
        let z = standard_normal.inverse_cdf(self.quantile);
        if !z.is_finite() {
            return Err(PyMabError::numerical(
                "Normal quantile",
                "quantile became non-finite",
            ));
        }
        Ok(self
            .state
            .means()
            .iter()
            .zip(self.state.precisions())
            .map(|(mean, precision)| mean + z / precision.sqrt())
            .collect())
    }
}

impl Policy for GaussianBayesianUCBPolicy {
    type State = GaussianPosteriorState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.means().len()
    }

    fn select_action(&mut self, _rng: &mut NativeRng) -> Result<ActionIndex> {
        deterministic_argmax(&self.upper_bounds()?)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.update(action, reward, self.reward_precision)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.action_values().recommendation()
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
