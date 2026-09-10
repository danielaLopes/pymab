//! Shared state and numerical helpers for action-value policies.

use std::mem::size_of;

use rand::Rng;

use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::ActionIndex;
use crate::validation::{finite, strictly_positive};

/// Common learned state for policies that estimate one value per arm.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionValueState {
    initial_value: f64,
    step: u64,
    total_reward: f64,
    counts: Vec<u64>,
    estimates: Vec<f64>,
}

impl ActionValueState {
    /// Construct reset state for `n_arms` arms.
    pub fn new(n_arms: usize, initial_value: f64) -> Result<Self> {
        if n_arms == 0 {
            return Err(PyMabError::configuration(
                "n_arms",
                "must be greater than zero",
            ));
        }
        if !initial_value.is_finite() {
            return Err(PyMabError::configuration("initial_value", "must be finite"));
        }
        Ok(Self {
            initial_value,
            step: 0,
            total_reward: 0.0,
            counts: vec![0; n_arms],
            estimates: vec![initial_value; n_arms],
        })
    }

    /// Return the number of arms.
    #[must_use]
    pub fn n_arms(&self) -> usize {
        self.counts.len()
    }

    /// Return the number of completed updates.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    /// Return cumulative observed reward.
    #[must_use]
    pub const fn total_reward(&self) -> f64 {
        self.total_reward
    }

    /// Return pull counts by arm.
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// Return estimated values by arm.
    #[must_use]
    pub fn estimates(&self) -> &[f64] {
        &self.estimates
    }

    /// Update an arm using its incremental sample mean.
    pub fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        finite("reward", reward)?;
        let index = action.get();
        if index >= self.n_arms() {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {})", self.n_arms()),
            ));
        }

        let step = self
            .step
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("policy step counter overflowed"))?;
        let count = self.counts[index]
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("arm pull counter overflowed"))?;
        let total_reward = self.total_reward + reward;
        if !total_reward.is_finite() {
            return Err(PyMabError::numerical(
                "reward accumulation",
                "total reward overflowed",
            ));
        }
        let estimate = self.estimates[index] + (reward - self.estimates[index]) / count as f64;
        if !estimate.is_finite() {
            return Err(PyMabError::numerical(
                "incremental mean",
                "arm estimate became non-finite",
            ));
        }

        self.step = step;
        self.counts[index] = count;
        self.total_reward = total_reward;
        self.estimates[index] = estimate;
        Ok(())
    }

    /// Replace one estimate after a specialized posterior update.
    pub(crate) fn set_estimate(&mut self, action: ActionIndex, estimate: f64) -> Result<()> {
        finite("estimate", estimate)?;
        let index = action.get();
        if index >= self.n_arms() {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {})", self.n_arms()),
            ));
        }
        self.estimates[index] = estimate;
        Ok(())
    }

    /// Restart the learned statistics for one arm after detecting a change.
    ///
    /// The triggering observation becomes the first sample in the new regime.
    /// Global step and reward totals deliberately remain cumulative.
    pub(crate) fn reset_arm(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        finite("reward", reward)?;
        let index = action.get();
        if index >= self.n_arms() {
            return Err(PyMabError::validation(
                "action",
                format!("index {index} is outside [0, {})", self.n_arms()),
            ));
        }
        self.counts[index] = 1;
        self.estimates[index] = reward;
        Ok(())
    }

    /// Reset all learned values and retain allocated buffers.
    pub fn reset(&mut self) {
        self.step = 0;
        self.total_reward = 0.0;
        self.counts.fill(0);
        self.estimates.fill(self.initial_value);
    }

    /// Clone configuration and allocated state into a fresh reset value.
    #[must_use]
    pub fn clone_reset(&self) -> Self {
        let mut cloned = self.clone();
        cloned.reset();
        cloned
    }

    /// Return the first arm with the greatest estimate.
    pub fn recommendation(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.estimates)
    }

    /// Estimate this state and its allocated buffers in bytes.
    #[must_use]
    pub fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.estimated_heap_bytes()
    }

    /// Estimate bytes reserved by owned heap buffers.
    #[must_use]
    pub fn estimated_heap_bytes(&self) -> usize {
        self.counts.capacity() * size_of::<u64>() + self.estimates.capacity() * size_of::<f64>()
    }
}

/// Return a numerically stable softmax probability vector.
pub fn softmax(values: &[f64], temperature: f64) -> Result<Vec<f64>> {
    strictly_positive("temperature", temperature)?;
    validate_values(values)?;

    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities: Vec<f64> = values
        .iter()
        .map(|value| ((value - maximum) / temperature).exp())
        .collect();
    let total: f64 = probabilities.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(PyMabError::numerical(
            "softmax",
            "normalizing constant is not finite and positive",
        ));
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    Ok(probabilities)
}

/// Break an exact maximum tie uniformly at random.
pub fn choose_argmax(values: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
    validate_values(values)?;
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == maximum).then_some(index))
        .collect();
    let selected = candidates[rng.random_range(0..candidates.len())];
    ActionIndex::new(selected, values.len())
}

/// Return the first index containing the maximum finite value.
pub fn deterministic_argmax(values: &[f64]) -> Result<ActionIndex> {
    validate_values(values)?;
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let index = values
        .iter()
        .position(|value| *value == maximum)
        .ok_or_else(|| PyMabError::internal("validated values unexpectedly empty"))?;
    ActionIndex::new(index, values.len())
}

fn validate_values(values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(PyMabError::validation("values", "must be non-empty"));
    }
    for &value in values {
        finite("values", value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{choose_argmax, softmax, ActionValueState};
    use crate::rng::{rng_for, StreamKey, StreamRole};
    use crate::types::ActionIndex;

    #[test]
    fn incremental_means_reset_and_recommendation_match_the_reference_contract() {
        let mut state = ActionValueState::new(3, 0.5).unwrap();
        state.update(ActionIndex::new(1, 3).unwrap(), 1.0).unwrap();
        state.update(ActionIndex::new(1, 3).unwrap(), 0.0).unwrap();
        state.update(ActionIndex::new(2, 3).unwrap(), 0.25).unwrap();

        assert_eq!(state.step(), 3);
        assert_eq!(state.total_reward(), 1.25);
        assert_eq!(state.counts(), &[0, 2, 1]);
        assert_eq!(state.estimates(), &[0.5, 0.5, 0.25]);
        assert_eq!(state.recommendation().unwrap().get(), 0);
        assert!(state.estimated_state_bytes() >= 6 * size_of::<u64>());

        let fresh = state.clone_reset();
        assert_eq!(fresh.step(), 0);
        assert_eq!(fresh.counts(), &[0, 0, 0]);
        assert_eq!(fresh.estimates(), &[0.5, 0.5, 0.5]);

        state.reset();
        assert_eq!(state.step(), 0);
        assert_eq!(state.counts(), &[0, 0, 0]);
        assert_eq!(state.estimates(), &[0.5, 0.5, 0.5]);
    }

    #[test]
    fn softmax_is_stable_for_large_values() {
        let probabilities = softmax(&[10_000.0, 10_001.0], 1.0).unwrap();
        assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(probabilities[1] > probabilities[0]);
    }

    #[test]
    fn tied_argmax_sampling_has_valid_support() {
        let key = StreamKey::new(7, 0, StreamRole::PolicySelection)
            .with_policy_id("tie-test")
            .unwrap();
        let mut rng = rng_for(&key).unwrap();
        for _ in 0..32 {
            assert!(matches!(
                choose_argmax(&[2.0, 1.0, 2.0], &mut rng).unwrap().get(),
                0 | 2
            ));
        }
    }

    #[test]
    fn updates_reject_invalid_actions_and_rewards_without_mutation() {
        let mut state = ActionValueState::new(2, 0.0).unwrap();
        let action_for_larger_policy = ActionIndex::new(2, 3).unwrap();

        assert!(state.update(action_for_larger_policy, 1.0).is_err());
        assert!(state
            .update(ActionIndex::new(0, 2).unwrap(), f64::NAN)
            .is_err());
        assert_eq!(state.step(), 0);
        assert_eq!(state.total_reward(), 0.0);
    }
}
