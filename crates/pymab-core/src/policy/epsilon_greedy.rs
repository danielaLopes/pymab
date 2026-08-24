//! Fixed and decaying epsilon-greedy policies.

use std::mem::size_of;

use rand::Rng;

use super::action_value::{choose_argmax, ActionValueState};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{ActionIndex, PolicyCapabilities, PolicyObjective, ALL_REWARD_DOMAINS};
use crate::validation::probability;

const CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Explore uniformly with fixed probability epsilon and otherwise act greedily.
#[derive(Clone, Debug, PartialEq)]
pub struct EpsilonGreedyPolicy {
    epsilon: f64,
    state: ActionValueState,
}

impl EpsilonGreedyPolicy {
    /// Construct a fixed epsilon-greedy policy.
    pub fn new(n_arms: usize, initial_value: f64, epsilon: f64) -> Result<Self> {
        Ok(Self {
            epsilon: probability("epsilon", epsilon)?,
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }

    /// Return the exploration probability.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

impl Policy for EpsilonGreedyPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if rng.random::<f64>() < self.epsilon {
            ActionIndex::new(rng.random_range(0..self.n_arms()), self.n_arms())
        } else {
            choose_argmax(self.state.estimates(), rng)
        }
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.update(action, reward)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.recommendation()
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

/// Epsilon-greedy policy with a hyperbolically decaying exploration schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct DecayingEpsilonGreedyPolicy {
    initial_epsilon: f64,
    min_epsilon: f64,
    decay_rate: f64,
    state: ActionValueState,
}

impl DecayingEpsilonGreedyPolicy {
    /// Construct a decaying epsilon-greedy policy.
    pub fn new(
        n_arms: usize,
        initial_value: f64,
        initial_epsilon: f64,
        min_epsilon: f64,
        decay_rate: f64,
    ) -> Result<Self> {
        let initial_epsilon = probability("initial_epsilon", initial_epsilon)?;
        let min_epsilon = probability("min_epsilon", min_epsilon)?;
        if min_epsilon > initial_epsilon {
            return Err(PyMabError::configuration(
                "min_epsilon",
                "must be less than or equal to initial_epsilon",
            ));
        }
        if !decay_rate.is_finite() || decay_rate < 0.0 {
            return Err(PyMabError::configuration(
                "decay_rate",
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            initial_epsilon,
            min_epsilon,
            decay_rate,
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }

    /// Return the current exploration probability.
    #[must_use]
    pub fn epsilon(&self) -> f64 {
        let decayed = self.min_epsilon
            + (self.initial_epsilon - self.min_epsilon)
                / (1.0 + self.decay_rate * self.state.step() as f64);
        self.min_epsilon.max(decayed)
    }
}

impl Policy for DecayingEpsilonGreedyPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if rng.random::<f64>() < self.epsilon() {
            ActionIndex::new(rng.random_range(0..self.n_arms()), self.n_arms())
        } else {
            choose_argmax(self.state.estimates(), rng)
        }
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.update(action, reward)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.recommendation()
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
