//! Classic non-contextual bandit environment.

use crate::distribution::{BuiltInRewardModel, RewardModel};
use crate::environment::dynamics::{BuiltInDynamics, EnvironmentDynamics};
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::RewardDomain;

/// Classic K-armed environment with support-aware rewards and dynamics.
#[derive(Clone, Debug, PartialEq)]
pub struct BanditEnvironment {
    means: Vec<f64>,
    reward_model: BuiltInRewardModel,
    dynamics: BuiltInDynamics,
}

impl BanditEnvironment {
    /// Construct a validated classic environment.
    pub fn new(
        means: Vec<f64>,
        reward_model: BuiltInRewardModel,
        dynamics: BuiltInDynamics,
    ) -> Result<Self> {
        reward_model.validate_means(&means)?;
        if !dynamics.supports(reward_model.domain()) {
            return Err(PyMabError::compatibility(
                "dynamics",
                format!("does not support {:?} rewards", reward_model.domain()),
            ));
        }
        Ok(Self {
            means,
            reward_model,
            dynamics,
        })
    }

    /// Return the number of arms.
    #[must_use]
    pub fn n_arms(&self) -> usize {
        self.means.len()
    }

    /// Return the reward domain.
    #[must_use]
    pub fn reward_domain(&self) -> RewardDomain {
        self.reward_model.domain()
    }

    /// Return current true means.
    #[must_use]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Return current expected rewards.
    #[must_use]
    pub fn expected_rewards(&self) -> &[f64] {
        &self.means
    }

    /// Return the built-in reward model.
    #[must_use]
    pub const fn reward_model(&self) -> BuiltInRewardModel {
        self.reward_model
    }

    /// Advance dynamics and validate the resulting state.
    pub fn advance(&mut self, step: u64, rng: &mut NativeRng) -> Result<()> {
        let values = self.dynamics.apply(&self.means, step, rng)?;
        if values.len() != self.means.len() {
            return Err(PyMabError::validation(
                "dynamics",
                "must preserve the arm-mean shape",
            ));
        }
        self.reward_model.validate_means(&values)?;
        self.means = values;
        Ok(())
    }

    /// Sample one potential reward per arm.
    pub fn sample_rewards(&self, rng: &mut NativeRng) -> Result<Vec<f64>> {
        self.reward_model.sample(&self.means, rng)
    }
}
