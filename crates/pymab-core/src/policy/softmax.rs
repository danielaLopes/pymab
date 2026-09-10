//! Softmax action-selection policy.

use std::mem::size_of;

use rand::Rng;

use super::action_value::{softmax, ActionValueState};
use super::Policy;
use crate::error::Result;
use crate::rng::NativeRng;
use crate::types::{ActionIndex, PolicyCapabilities, PolicyObjective, ALL_REWARD_DOMAINS};
use crate::validation::strictly_positive;

const CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Sample arms according to a softmax over current value estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftmaxPolicy {
    temperature: f64,
    state: ActionValueState,
}

impl SoftmaxPolicy {
    /// Construct a softmax policy.
    pub fn new(n_arms: usize, initial_value: f64, temperature: f64) -> Result<Self> {
        Ok(Self {
            temperature: strictly_positive("temperature", temperature)?,
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }

    /// Return normalized action probabilities for the current estimates.
    pub fn action_probabilities(&self) -> Result<Vec<f64>> {
        softmax(self.state.estimates(), self.temperature)
    }
}

impl Policy for SoftmaxPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let probabilities = self.action_probabilities()?;
        let draw = rng.random::<f64>();
        let mut cumulative = 0.0;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if draw < cumulative {
                return ActionIndex::new(index, self.n_arms());
            }
        }
        ActionIndex::new(self.n_arms() - 1, self.n_arms())
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
