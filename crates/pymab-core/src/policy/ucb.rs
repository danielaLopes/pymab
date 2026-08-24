//! Stationary upper-confidence-bound policies.

use std::mem::size_of;

use super::action_value::{choose_argmax, ActionValueState};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{
    ActionIndex, PolicyCapabilities, PolicyObjective, RewardDomain, ALL_REWARD_DOMAINS,
};
use crate::validation::{reward, strictly_positive};

const GENERAL_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);
const BINARY_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary];
const BINARY_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, BINARY_DOMAINS, PolicyObjective::CumulativeReward);

/// UCB1 for stationary sub-Gaussian rewards.
#[derive(Clone, Debug, PartialEq)]
pub struct UCBPolicy {
    c: f64,
    reward_scale: f64,
    state: ActionValueState,
}

impl UCBPolicy {
    /// Construct a UCB1 policy.
    pub fn new(n_arms: usize, initial_value: f64, c: f64, reward_scale: f64) -> Result<Self> {
        Ok(Self {
            c: strictly_positive("c", c)?,
            reward_scale: strictly_positive("reward_scale", reward_scale)?,
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }

    /// Return confidence bonuses for every arm.
    pub fn confidence_bonus(&self) -> Result<Vec<f64>> {
        let log_term = (self.state.step() as f64 + 1.0).ln();
        Ok(self
            .state
            .counts()
            .iter()
            .map(|count| {
                if *count == 0 {
                    f64::INFINITY
                } else {
                    self.reward_scale * (self.c * log_term / *count as f64).sqrt()
                }
            })
            .collect())
    }
}

impl Policy for UCBPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if let Some(index) = first_unseen(&self.state) {
            return ActionIndex::new(index, self.n_arms());
        }
        let values: Vec<f64> = self
            .state
            .estimates()
            .iter()
            .zip(self.confidence_bonus()?)
            .map(|(estimate, bonus)| estimate + bonus)
            .collect();
        choose_argmax(&values, rng)
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

/// KL-UCB for binary Bernoulli rewards.
#[derive(Clone, Debug, PartialEq)]
pub struct KLUCBPolicy {
    c: f64,
    tolerance: f64,
    max_iterations: usize,
    state: ActionValueState,
}

impl KLUCBPolicy {
    /// Construct a Bernoulli KL-UCB policy.
    pub fn new(
        n_arms: usize,
        initial_value: f64,
        c: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self> {
        if max_iterations == 0 {
            return Err(PyMabError::configuration(
                "max_iterations",
                "must be greater than zero",
            ));
        }
        Ok(Self {
            c: strictly_positive("c", c)?,
            tolerance: strictly_positive("tolerance", tolerance)?,
            max_iterations,
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }

    /// Return current KL-UCB indices for every arm.
    pub fn indices(&self) -> Result<Vec<f64>> {
        let step = self.state.step() as f64;
        let budget = step.max(2.0).ln() + self.c * step.max(3.0).ln().max(1.0).ln();
        self.state
            .estimates()
            .iter()
            .zip(self.state.counts())
            .map(|(mean, count)| self.solve_index(mean.clamp(0.0, 1.0), budget / *count as f64))
            .collect()
    }

    fn solve_index(&self, mean: f64, budget: f64) -> Result<f64> {
        if mean >= 1.0 {
            return Ok(1.0);
        }
        let mut low = mean;
        let mut high = 1.0;
        for _ in 0..self.max_iterations {
            let midpoint = (low + high) / 2.0;
            if bernoulli_kl(mean, midpoint)? <= budget {
                low = midpoint;
            } else {
                high = midpoint;
            }
            if high - low <= self.tolerance {
                break;
            }
        }
        Ok(low)
    }
}

impl Policy for KLUCBPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        BINARY_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if let Some(index) = first_unseen(&self.state) {
            return ActionIndex::new(index, self.n_arms());
        }
        choose_argmax(&self.indices()?, rng)
    }

    fn update(&mut self, action: ActionIndex, observed_reward: f64) -> Result<()> {
        let observed_reward = reward("reward", observed_reward, RewardDomain::Binary)?;
        self.state.update(action, observed_reward)
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

/// Minimax Optimal Strategy in the Stochastic case.
#[derive(Clone, Debug, PartialEq)]
pub struct MOSSPolicy {
    horizon: u64,
    c: f64,
    reward_scale: f64,
    state: ActionValueState,
}

impl MOSSPolicy {
    /// Construct a MOSS policy for a known horizon.
    pub fn new(
        n_arms: usize,
        initial_value: f64,
        horizon: u64,
        c: f64,
        reward_scale: f64,
    ) -> Result<Self> {
        if horizon == 0 {
            return Err(PyMabError::configuration(
                "horizon",
                "must be greater than zero",
            ));
        }
        if horizon < n_arms as u64 {
            return Err(PyMabError::configuration(
                "horizon",
                "must be greater than or equal to n_arms",
            ));
        }
        Ok(Self {
            horizon,
            c: strictly_positive("c", c)?,
            reward_scale: strictly_positive("reward_scale", reward_scale)?,
            state: ActionValueState::new(n_arms, initial_value)?,
        })
    }

    /// Return MOSS confidence bonuses for every arm.
    pub fn confidence_bonus(&self) -> Result<Vec<f64>> {
        let n_arms = self.state.n_arms() as f64;
        Ok(self
            .state
            .counts()
            .iter()
            .map(|count| {
                let count = (*count).max(1) as f64;
                let log_term = (self.horizon as f64 / (n_arms * count)).ln().max(0.0);
                self.reward_scale * (self.c * log_term / count).sqrt()
            })
            .collect())
    }
}

impl Policy for MOSSPolicy {
    type State = ActionValueState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if let Some(index) = first_unseen(&self.state) {
            return ActionIndex::new(index, self.n_arms());
        }
        let values: Vec<f64> = self
            .state
            .estimates()
            .iter()
            .zip(self.confidence_bonus()?)
            .map(|(estimate, bonus)| estimate + bonus)
            .collect();
        choose_argmax(&values, rng)
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

/// Return Bernoulli relative entropy `KL(p || q)` with stable endpoints.
pub fn bernoulli_kl(p: f64, q: f64) -> Result<f64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(PyMabError::validation("p", "must be in [0, 1]"));
    }
    if !q.is_finite() || !(0.0..=1.0).contains(&q) {
        return Err(PyMabError::validation("q", "must be in [0, 1]"));
    }
    let epsilon = 1e-15;
    let clipped_p = p.clamp(epsilon, 1.0 - epsilon);
    let clipped_q = q.clamp(epsilon, 1.0 - epsilon);
    Ok(clipped_p * (clipped_p / clipped_q).ln()
        + (1.0 - clipped_p) * ((1.0 - clipped_p) / (1.0 - clipped_q)).ln())
}

fn first_unseen(state: &ActionValueState) -> Option<usize> {
    state.counts().iter().position(|count| *count == 0)
}
