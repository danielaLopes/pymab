//! Fixed-confidence best-arm identification policies.

use std::mem::size_of;

use rand::Rng;

use super::action_value::ActionValueState;
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{
    ActionIndex, PolicyCapabilities, PolicyObjective, RewardDomain, ALL_REWARD_DOMAINS,
};
use crate::validation::{probability, reward, strictly_positive};

const SUCCESSIVE_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::BestArm);
const BOUNDED_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary, RewardDomain::UnitInterval];
const MEDIAN_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, BOUNDED_DOMAINS, PolicyObjective::BestArm);

/// Learned state for successive elimination.
#[derive(Clone, Debug, PartialEq)]
pub struct SuccessiveEliminationState {
    action_values: ActionValueState,
    active: Vec<bool>,
}

impl SuccessiveEliminationState {
    /// Return common action-value state.
    #[must_use]
    pub const fn action_values(&self) -> &ActionValueState {
        &self.action_values
    }

    /// Return the active-arm mask.
    #[must_use]
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.action_values.estimated_heap_bytes() + self.active.capacity() * size_of::<bool>()
    }
}

/// Successive elimination for fixed-confidence best-arm identification.
#[derive(Clone, Debug, PartialEq)]
pub struct SuccessiveEliminationPolicy {
    delta: f64,
    confidence_scale: f64,
    state: SuccessiveEliminationState,
}

impl SuccessiveEliminationPolicy {
    /// Construct a successive-elimination policy.
    pub fn new(n_arms: usize, delta: f64, confidence_scale: f64) -> Result<Self> {
        let delta = positive_probability("delta", delta)?;
        let confidence_scale = strictly_positive("confidence_scale", confidence_scale)?;
        Ok(Self {
            delta,
            confidence_scale,
            state: SuccessiveEliminationState {
                action_values: ActionValueState::new(n_arms, 0.0)?,
                active: vec![true; n_arms],
            },
        })
    }

    /// Return confidence radii for every arm.
    #[must_use]
    pub fn confidence_radii(&self) -> Vec<f64> {
        let step = self.state.action_values.step().max(1) as f64;
        let log_term = (4.0 * self.n_arms() as f64 * step.powi(2) / self.delta).ln();
        self.state
            .action_values
            .counts()
            .iter()
            .map(|count| self.confidence_scale * (log_term / (2.0 * (*count).max(1) as f64)).sqrt())
            .collect()
    }

    fn eliminate(&mut self) {
        let active = active_indices(&self.state.active);
        if active.len() <= 1
            || active
                .iter()
                .any(|index| self.state.action_values.counts()[*index] == 0)
        {
            return;
        }
        let radii = self.confidence_radii();
        let best_lower = active
            .iter()
            .map(|index| self.state.action_values.estimates()[*index] - radii[*index])
            .fold(f64::NEG_INFINITY, f64::max);
        for index in active {
            self.state.active[index] =
                self.state.action_values.estimates()[index] + radii[index] >= best_lower;
        }
    }
}

impl Policy for SuccessiveEliminationPolicy {
    type State = SuccessiveEliminationState;

    fn capabilities(&self) -> PolicyCapabilities {
        SUCCESSIVE_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.active.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let active = active_indices(&self.state.active);
        if active.len() == 1 {
            return ActionIndex::new(active[0], self.n_arms());
        }
        let minimum = active
            .iter()
            .map(|index| self.state.action_values.counts()[*index])
            .min()
            .ok_or_else(|| PyMabError::internal("no active arms"))?;
        let candidates: Vec<_> = active
            .into_iter()
            .filter(|index| self.state.action_values.counts()[*index] == minimum)
            .collect();
        ActionIndex::new(
            candidates[rng.random_range(0..candidates.len())],
            self.n_arms(),
        )
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        self.state.action_values.update(action, reward)?;
        self.eliminate();
        Ok(())
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        best_active_arm(&self.state.action_values, &self.state.active)
    }

    fn reset(&mut self) {
        self.state.action_values.reset();
        self.state.active.fill(true);
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

/// Learned phase state for median elimination.
#[derive(Clone, Debug, PartialEq)]
pub struct MedianEliminationState {
    action_values: ActionValueState,
    phase_epsilon: f64,
    phase_delta: f64,
    active: Vec<bool>,
    phase_counts: Vec<u64>,
    phase_sums: Vec<f64>,
}

impl MedianEliminationState {
    /// Return common action-value state.
    #[must_use]
    pub const fn action_values(&self) -> &ActionValueState {
        &self.action_values
    }

    /// Return the current phase epsilon.
    #[must_use]
    pub const fn phase_epsilon(&self) -> f64 {
        self.phase_epsilon
    }

    /// Return the current phase delta.
    #[must_use]
    pub const fn phase_delta(&self) -> f64 {
        self.phase_delta
    }

    /// Return the active-arm mask.
    #[must_use]
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    /// Return per-arm samples collected in this phase.
    #[must_use]
    pub fn phase_counts(&self) -> &[u64] {
        &self.phase_counts
    }

    /// Return per-arm reward sums collected in this phase.
    #[must_use]
    pub fn phase_sums(&self) -> &[f64] {
        &self.phase_sums
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.action_values.estimated_heap_bytes()
            + self.active.capacity() * size_of::<bool>()
            + self.phase_counts.capacity() * size_of::<u64>()
            + self.phase_sums.capacity() * size_of::<f64>()
    }
}

/// Median elimination for rewards bounded to `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct MedianEliminationPolicy {
    epsilon: f64,
    delta: f64,
    state: MedianEliminationState,
}

impl MedianEliminationPolicy {
    /// Construct a median-elimination policy.
    pub fn new(n_arms: usize, epsilon: f64, delta: f64) -> Result<Self> {
        let epsilon = positive_probability("epsilon", epsilon)?;
        let delta = positive_probability("delta", delta)?;
        Ok(Self {
            epsilon,
            delta,
            state: MedianEliminationState {
                action_values: ActionValueState::new(n_arms, 0.0)?,
                phase_epsilon: epsilon / 4.0,
                phase_delta: delta / 2.0,
                active: vec![true; n_arms],
                phase_counts: vec![0; n_arms],
                phase_sums: vec![0.0; n_arms],
            },
        })
    }

    /// Return the per-active-arm sample quota for the current phase.
    #[must_use]
    pub fn phase_quota(&self) -> u64 {
        ((4.0 / self.state.phase_epsilon.powi(2)) * (3.0 / self.state.phase_delta).ln())
            .ceil()
            .max(1.0) as u64
    }

    fn complete_phase_if_ready(&mut self) {
        let active = active_indices(&self.state.active);
        if active.len() <= 1 {
            return;
        }
        let quota = self.phase_quota();
        if active
            .iter()
            .any(|index| self.state.phase_counts[*index] < quota)
        {
            return;
        }
        let mut means: Vec<f64> = active
            .iter()
            .map(|index| self.state.phase_sums[*index] / self.state.phase_counts[*index] as f64)
            .collect();
        means.sort_by(f64::total_cmp);
        let middle = means.len() / 2;
        let median = if means.len() % 2 == 0 {
            (means[middle - 1] + means[middle]) / 2.0
        } else {
            means[middle]
        };
        for index in active {
            let mean = self.state.phase_sums[index] / self.state.phase_counts[index] as f64;
            self.state.active[index] = mean >= median;
        }
        self.state.phase_counts.fill(0);
        self.state.phase_sums.fill(0.0);
        self.state.phase_epsilon *= 0.75;
        self.state.phase_delta *= 0.5;
    }
}

impl Policy for MedianEliminationPolicy {
    type State = MedianEliminationState;

    fn capabilities(&self) -> PolicyCapabilities {
        MEDIAN_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.active.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let active = active_indices(&self.state.active);
        if active.len() == 1 {
            return ActionIndex::new(active[0], self.n_arms());
        }
        let quota = self.phase_quota();
        let under_sampled: Vec<_> = active
            .iter()
            .copied()
            .filter(|index| self.state.phase_counts[*index] < quota)
            .collect();
        let candidates = if under_sampled.is_empty() {
            active
        } else {
            under_sampled
        };
        let minimum = candidates
            .iter()
            .map(|index| self.state.phase_counts[*index])
            .min()
            .ok_or_else(|| PyMabError::internal("no active arms"))?;
        let tied: Vec<_> = candidates
            .into_iter()
            .filter(|index| self.state.phase_counts[*index] == minimum)
            .collect();
        ActionIndex::new(tied[rng.random_range(0..tied.len())], self.n_arms())
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
        let phase_count = self.state.phase_counts[index]
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("phase count overflowed"))?;
        let phase_sum = self.state.phase_sums[index] + observed_reward;
        if !phase_sum.is_finite() {
            return Err(PyMabError::numerical(
                "median elimination update",
                "phase reward sum became non-finite",
            ));
        }
        self.state.action_values.update(action, observed_reward)?;
        self.state.phase_counts[index] = phase_count;
        self.state.phase_sums[index] = phase_sum;
        self.complete_phase_if_ready();
        Ok(())
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        best_active_arm(&self.state.action_values, &self.state.active)
    }

    fn reset(&mut self) {
        self.state.action_values.reset();
        self.state.phase_epsilon = self.epsilon / 4.0;
        self.state.phase_delta = self.delta / 2.0;
        self.state.active.fill(true);
        self.state.phase_counts.fill(0);
        self.state.phase_sums.fill(0.0);
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

fn positive_probability(name: &str, value: f64) -> Result<f64> {
    let value = probability(name, value)?;
    if value == 0.0 {
        Err(PyMabError::configuration(name, "must be greater than zero"))
    } else {
        Ok(value)
    }
}

fn active_indices(active: &[bool]) -> Vec<usize> {
    active
        .iter()
        .enumerate()
        .filter_map(|(index, is_active)| (*is_active).then_some(index))
        .collect()
}

fn best_active_arm(state: &ActionValueState, active: &[bool]) -> Result<ActionIndex> {
    let active = active_indices(active);
    let mut candidates = active.into_iter();
    let mut best = candidates
        .next()
        .ok_or_else(|| PyMabError::internal("no active arms"))?;
    for candidate in candidates {
        if state.estimates()[candidate] > state.estimates()[best] {
            best = candidate;
        }
    }
    ActionIndex::new(best, state.n_arms())
}
