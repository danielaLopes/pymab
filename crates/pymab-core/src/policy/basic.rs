//! Uniform-random and greedy action-value policies.

use std::mem::size_of;

use rand::Rng;

use super::action_value::{choose_argmax, ActionValueState};
use super::Policy;
use crate::error::Result;
use crate::rng::NativeRng;
use crate::types::{ActionIndex, PolicyCapabilities, PolicyObjective, ALL_REWARD_DOMAINS};

const CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Select every arm uniformly at random while tracking observed values.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomPolicy {
    state: ActionValueState,
}

impl RandomPolicy {
    /// Construct a random policy.
    pub fn new(n_arms: usize) -> Result<Self> {
        Ok(Self {
            state: ActionValueState::new(n_arms, 0.0)?,
        })
    }
}

impl Policy for RandomPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        ActionIndex::new(rng.random_range(0..self.n_arms()), self.n_arms())
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

/// Select an arm with the greatest current value estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct GreedyPolicy {
    state: ActionValueState,
}

impl GreedyPolicy {
    /// Construct a greedy policy with an optimistic initial value if desired.
    pub fn new(n_arms: usize, initial_value: f64) -> Result<Self> {
        Ok(Self {
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }
}

impl Policy for GreedyPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        choose_argmax(self.state.estimates(), rng)
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
