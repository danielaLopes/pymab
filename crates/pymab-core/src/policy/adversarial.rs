//! Adversarial bandit policies.

use std::mem::size_of;

use rand::Rng;

use super::action_value::{deterministic_argmax, ActionValueState};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{ActionIndex, PolicyCapabilities, PolicyObjective, RewardDomain};
use crate::validation::{probability, reward, strictly_positive};

const REWARD_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary, RewardDomain::UnitInterval];
const CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Learned EXP3 log weights and action-value diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct EXP3State {
    action_values: ActionValueState,
    log_weights: Vec<f64>,
    last_probabilities: Vec<f64>,
}

impl EXP3State {
    /// Return common action-value diagnostics.
    #[must_use]
    pub const fn action_values(&self) -> &ActionValueState {
        &self.action_values
    }

    /// Return numerically stabilized log weights.
    #[must_use]
    pub fn log_weights(&self) -> &[f64] {
        &self.log_weights
    }

    /// Return probabilities used for the latest selection.
    #[must_use]
    pub fn last_probabilities(&self) -> &[f64] {
        &self.last_probabilities
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.action_values.estimated_heap_bytes()
            + (self.log_weights.capacity() + self.last_probabilities.capacity()) * size_of::<f64>()
    }
}

/// EXP3 with uniform exploration for rewards in `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct EXP3Policy {
    gamma: f64,
    learning_rate: f64,
    state: EXP3State,
}

impl EXP3Policy {
    /// Construct an EXP3 policy. A missing learning rate uses `gamma`.
    pub fn new(n_arms: usize, gamma: f64, learning_rate: Option<f64>) -> Result<Self> {
        let gamma = probability("gamma", gamma)?;
        if gamma == 0.0 {
            return Err(PyMabError::configuration(
                "gamma",
                "must be greater than zero",
            ));
        }
        let learning_rate = match learning_rate {
            Some(value) => {
                let value = strictly_positive("learning_rate", value)?;
                if value > 1.0 {
                    return Err(PyMabError::configuration(
                        "learning_rate",
                        "must be less than or equal to one",
                    ));
                }
                value
            }
            None => gamma,
        };
        let action_values = ActionValueState::new(n_arms, 0.0)?;
        Ok(Self {
            gamma,
            learning_rate,
            state: EXP3State {
                action_values,
                log_weights: vec![0.0; n_arms],
                last_probabilities: vec![1.0 / n_arms as f64; n_arms],
            },
        })
    }

    /// Return normalized EXP3 action probabilities.
    pub fn action_probabilities(&self) -> Result<Vec<f64>> {
        let maximum = self
            .state
            .log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let relative_weights: Vec<f64> = self
            .state
            .log_weights
            .iter()
            .map(|weight| (weight - maximum).exp())
            .collect();
        let total: f64 = relative_weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(PyMabError::numerical(
                "EXP3 probabilities",
                "relative weights do not have a finite positive sum",
            ));
        }
        let exploration = self.gamma / self.n_arms() as f64;
        let mut probabilities: Vec<f64> = relative_weights
            .iter()
            .map(|weight| (1.0 - self.gamma) * weight / total + exploration)
            .collect();
        let normalization: f64 = probabilities.iter().sum();
        for value in &mut probabilities {
            *value /= normalization;
        }
        Ok(probabilities)
    }

    /// Return multiplicative weights normalized to a maximum of one.
    #[must_use]
    pub fn weights(&self) -> Vec<f64> {
        let maximum = self
            .state
            .log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        self.state
            .log_weights
            .iter()
            .map(|weight| (weight - maximum).exp())
            .collect()
    }
}

impl Policy for EXP3Policy {
    type State = EXP3State;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.log_weights.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let probabilities = self.action_probabilities()?;
        let action = sample_probabilities(&probabilities, rng)?;
        self.state.last_probabilities = probabilities;
        Ok(action)
    }

    fn update(&mut self, action: ActionIndex, observed_reward: f64) -> Result<()> {
        let observed_reward = reward("reward", observed_reward, RewardDomain::UnitInterval)?;
        let index = action.get();
        if index >= self.n_arms() {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {})", self.n_arms()),
            ));
        }
        let selected_probability = self.state.last_probabilities[index];
        if !selected_probability.is_finite() || selected_probability <= 0.0 {
            return Err(PyMabError::validation(
                "selected_probability",
                "must be positive and finite",
            ));
        }
        let increment =
            self.learning_rate * (observed_reward / selected_probability) / self.n_arms() as f64;
        if !increment.is_finite() {
            return Err(PyMabError::numerical(
                "EXP3 update",
                "importance-weighted increment became non-finite",
            ));
        }
        let mut log_weights = self.state.log_weights.clone();
        log_weights[index] += increment;
        let maximum = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        for weight in &mut log_weights {
            *weight -= maximum;
        }
        self.state.action_values.update(action, observed_reward)?;
        self.state.log_weights = log_weights;
        Ok(())
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.action_probabilities()?)
    }

    fn reset(&mut self) {
        let uniform = 1.0 / self.n_arms() as f64;
        self.state.action_values.reset();
        self.state.log_weights.fill(0.0);
        self.state.last_probabilities.fill(uniform);
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
