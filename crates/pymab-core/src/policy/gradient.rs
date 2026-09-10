//! Gradient bandit policy.

use std::mem::size_of;

use rand::Rng;

use super::action_value::{deterministic_argmax, softmax};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{ActionIndex, PolicyCapabilities, PolicyObjective, ALL_REWARD_DOMAINS};
use crate::validation::{finite, strictly_positive};

const CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Learned preference and baseline state for a gradient bandit.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientState {
    step: u64,
    average_reward: f64,
    preferences: Vec<f64>,
    probabilities: Vec<f64>,
}

impl GradientState {
    /// Return the number of completed updates.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Return the running average reward.
    #[must_use]
    pub const fn average_reward(&self) -> f64 {
        self.average_reward
    }

    /// Return action preferences.
    #[must_use]
    pub fn preferences(&self) -> &[f64] {
        &self.preferences
    }

    /// Return probabilities last used for action selection.
    #[must_use]
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    fn estimated_heap_bytes(&self) -> usize {
        (self.preferences.capacity() + self.probabilities.capacity()) * size_of::<f64>()
    }
}

/// Learn action preferences with stochastic gradient ascent.
#[derive(Clone, Debug, PartialEq)]
pub struct GradientBanditPolicy {
    learning_rate: f64,
    use_baseline: bool,
    state: GradientState,
}

impl GradientBanditPolicy {
    /// Construct a gradient bandit policy.
    pub fn new(n_arms: usize, learning_rate: f64, use_baseline: bool) -> Result<Self> {
        if n_arms == 0 {
            return Err(PyMabError::configuration(
                "n_arms",
                "must be greater than zero",
            ));
        }
        let learning_rate = strictly_positive("learning_rate", learning_rate)?;
        Ok(Self {
            learning_rate,
            use_baseline,
            state: GradientState {
                step: 0,
                average_reward: 0.0,
                preferences: vec![0.0; n_arms],
                probabilities: vec![1.0 / n_arms as f64; n_arms],
            },
        })
    }
}

impl Policy for GradientBanditPolicy {
    type State = GradientState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.preferences.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        self.state.probabilities = softmax(&self.state.preferences, 1.0)?;
        sample_probabilities(&self.state.probabilities, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        finite("reward", reward)?;
        let index = action.get();
        if index >= self.n_arms() {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {})", self.n_arms()),
            ));
        }
        let step = self
            .state
            .step
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("policy step counter overflowed"))?;
        let baseline = if self.use_baseline {
            self.state.average_reward
        } else {
            0.0
        };
        let advantage = reward - baseline;
        let mut preferences = self.state.preferences.clone();
        for (arm, preference) in preferences.iter_mut().enumerate() {
            let one_hot = f64::from(arm == index);
            *preference +=
                self.learning_rate * advantage * (one_hot - self.state.probabilities[arm]);
            if !preference.is_finite() {
                return Err(PyMabError::numerical(
                    "gradient update",
                    "action preference became non-finite",
                ));
            }
        }
        let average_reward =
            self.state.average_reward + (reward - self.state.average_reward) / step as f64;
        if !average_reward.is_finite() {
            return Err(PyMabError::numerical(
                "baseline update",
                "average reward became non-finite",
            ));
        }
        self.state.preferences = preferences;
        self.state.step = step;
        self.state.average_reward = average_reward;
        Ok(())
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.state.preferences)
    }

    fn reset(&mut self) {
        let uniform_probability = 1.0 / self.n_arms() as f64;
        self.state.step = 0;
        self.state.average_reward = 0.0;
        self.state.preferences.fill(0.0);
        self.state.probabilities.fill(uniform_probability);
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

fn sample_probabilities(probabilities: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
    let draw = rng.random::<f64>();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if draw < cumulative {
            return ActionIndex::new(index, probabilities.len());
        }
    }
    ActionIndex::new(probabilities.len() - 1, probabilities.len())
}
