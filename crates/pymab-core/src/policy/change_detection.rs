//! UCB policies with per-arm abrupt-change detection.

use std::mem::size_of;

use super::action_value::{choose_argmax, ActionValueState};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{ActionIndex, PolicyCapabilities, PolicyObjective, ALL_REWARD_DOMAINS};
use crate::validation::{finite, strictly_positive};

const CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);

/// Per-arm detector used by [`ChangePointUCBPolicy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChangeDetector {
    /// Two-sided cumulative-sum detector.
    Cusum,
    /// One-sided Page-Hinkley detector.
    PageHinkley,
}

/// Learned state shared by the change-detection policies.
#[derive(Clone, Debug, PartialEq)]
pub struct ChangePointState {
    action_values: ActionValueState,
    detector_counts: Vec<u64>,
    detector_means: Vec<f64>,
    positive_cusum: Vec<f64>,
    negative_cusum: Vec<f64>,
    ph_cumulative: Vec<f64>,
    ph_minimum: Vec<f64>,
    change_counts: Vec<u64>,
}

impl ChangePointState {
    fn new(n_arms: usize, initial_value: f64) -> Result<Self> {
        Ok(Self {
            action_values: ActionValueState::new(n_arms, initial_value)?,
            detector_counts: vec![0; n_arms],
            detector_means: vec![0.0; n_arms],
            positive_cusum: vec![0.0; n_arms],
            negative_cusum: vec![0.0; n_arms],
            ph_cumulative: vec![0.0; n_arms],
            ph_minimum: vec![0.0; n_arms],
            change_counts: vec![0; n_arms],
        })
    }

    /// Return the common action-value state.
    #[must_use]
    pub const fn action_values(&self) -> &ActionValueState {
        &self.action_values
    }

    /// Return observations accumulated by each detector since its last reset.
    #[must_use]
    pub fn detector_counts(&self) -> &[u64] {
        &self.detector_counts
    }

    /// Return the running means used by each detector.
    #[must_use]
    pub fn detector_means(&self) -> &[f64] {
        &self.detector_means
    }

    /// Return positive CUSUM statistics by arm.
    #[must_use]
    pub fn positive_cusum(&self) -> &[f64] {
        &self.positive_cusum
    }

    /// Return negative CUSUM statistics by arm.
    #[must_use]
    pub fn negative_cusum(&self) -> &[f64] {
        &self.negative_cusum
    }

    /// Return Page-Hinkley cumulative statistics by arm.
    #[must_use]
    pub fn ph_cumulative(&self) -> &[f64] {
        &self.ph_cumulative
    }

    /// Return Page-Hinkley running minima by arm.
    #[must_use]
    pub fn ph_minimum(&self) -> &[f64] {
        &self.ph_minimum
    }

    /// Return the number of detected changes by arm.
    #[must_use]
    pub fn change_counts(&self) -> &[u64] {
        &self.change_counts
    }

    /// Return whether every floating-point state value is finite.
    #[must_use]
    pub fn all_finite(&self) -> bool {
        self.action_values.total_reward().is_finite()
            && self
                .action_values
                .estimates()
                .iter()
                .all(|value| value.is_finite())
            && self.detector_means.iter().all(|value| value.is_finite())
            && self.positive_cusum.iter().all(|value| value.is_finite())
            && self.negative_cusum.iter().all(|value| value.is_finite())
            && self.ph_cumulative.iter().all(|value| value.is_finite())
            && self.ph_minimum.iter().all(|value| value.is_finite())
    }

    fn reset(&mut self) {
        self.action_values.reset();
        self.detector_counts.fill(0);
        self.detector_means.fill(0.0);
        self.positive_cusum.fill(0.0);
        self.negative_cusum.fill(0.0);
        self.ph_cumulative.fill(0.0);
        self.ph_minimum.fill(0.0);
        self.change_counts.fill(0);
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.action_values.estimated_heap_bytes()
            + self.detector_counts.capacity() * size_of::<u64>()
            + self.detector_means.capacity() * size_of::<f64>()
            + self.positive_cusum.capacity() * size_of::<f64>()
            + self.negative_cusum.capacity() * size_of::<f64>()
            + self.ph_cumulative.capacity() * size_of::<f64>()
            + self.ph_minimum.capacity() * size_of::<f64>()
            + self.change_counts.capacity() * size_of::<u64>()
    }
}

/// UCB with a configurable per-arm change detector and local resets.
#[derive(Clone, Debug, PartialEq)]
pub struct ChangePointUCBPolicy {
    c: f64,
    reward_scale: f64,
    detector: ChangeDetector,
    threshold: f64,
    drift: f64,
    min_observations: u64,
    state: ChangePointState,
}

impl ChangePointUCBPolicy {
    /// Construct a change-detection UCB policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n_arms: usize,
        initial_value: f64,
        c: f64,
        reward_scale: f64,
        detector: ChangeDetector,
        threshold: f64,
        drift: f64,
        min_observations: u64,
    ) -> Result<Self> {
        if min_observations == 0 {
            return Err(PyMabError::configuration(
                "min_observations",
                "must be greater than zero",
            ));
        }
        let drift = finite("drift", drift)?;
        if drift < 0.0 {
            return Err(PyMabError::configuration(
                "drift",
                "must be greater than or equal to zero",
            ));
        }
        Ok(Self {
            c: strictly_positive("c", c)?,
            reward_scale: strictly_positive("reward_scale", reward_scale)?,
            detector,
            threshold: strictly_positive("threshold", threshold)?,
            drift,
            min_observations,
            state: ChangePointState::new(n_arms, initial_value)?,
        })
    }

    fn update_detector(&mut self, index: usize, reward: f64) -> Result<bool> {
        let previous_mean = self.state.detector_means[index];
        let count = self.state.detector_counts[index]
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("detector observation counter overflowed"))?;
        let mean = previous_mean + (reward - previous_mean) / count as f64;
        if !mean.is_finite() {
            return Err(PyMabError::numerical(
                "change detector",
                "running mean became non-finite",
            ));
        }

        let mut positive = self.state.positive_cusum[index];
        let mut negative = self.state.negative_cusum[index];
        let mut cumulative = self.state.ph_cumulative[index];
        let mut minimum = self.state.ph_minimum[index];
        let changed = if count < self.min_observations {
            false
        } else {
            let residual = reward - previous_mean;
            match self.detector {
                ChangeDetector::Cusum => {
                    positive = (positive + residual - self.drift).max(0.0);
                    negative = (negative - residual - self.drift).max(0.0);
                    if !positive.is_finite() || !negative.is_finite() {
                        return Err(PyMabError::numerical(
                            "CUSUM",
                            "detector statistic became non-finite",
                        ));
                    }
                    positive > self.threshold || negative > self.threshold
                }
                ChangeDetector::PageHinkley => {
                    cumulative += residual - self.drift;
                    minimum = minimum.min(cumulative);
                    if !cumulative.is_finite() || !minimum.is_finite() {
                        return Err(PyMabError::numerical(
                            "Page-Hinkley",
                            "detector statistic became non-finite",
                        ));
                    }
                    cumulative - minimum > self.threshold
                }
            }
        };

        self.state.detector_counts[index] = count;
        self.state.detector_means[index] = mean;
        self.state.positive_cusum[index] = positive;
        self.state.negative_cusum[index] = negative;
        self.state.ph_cumulative[index] = cumulative;
        self.state.ph_minimum[index] = minimum;
        Ok(changed)
    }

    fn reset_arm(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        let index = action.get();
        let changes = self.state.change_counts[index]
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("change counter overflowed"))?;
        self.state.action_values.reset_arm(action, reward)?;
        self.state.change_counts[index] = changes;
        self.state.detector_counts[index] = 1;
        self.state.detector_means[index] = reward;
        self.state.positive_cusum[index] = 0.0;
        self.state.negative_cusum[index] = 0.0;
        self.state.ph_cumulative[index] = 0.0;
        self.state.ph_minimum[index] = 0.0;
        Ok(())
    }
}

impl Policy for ChangePointUCBPolicy {
    type State = ChangePointState;

    fn capabilities(&self) -> PolicyCapabilities {
        CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.action_values.n_arms()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if let Some(index) = self
            .state
            .action_values
            .counts()
            .iter()
            .position(|count| *count == 0)
        {
            return ActionIndex::new(index, self.n_arms());
        }
        let log_term = (self.state.action_values.step() as f64 + 1.0).ln();
        let scores: Vec<f64> = self
            .state
            .action_values
            .estimates()
            .iter()
            .zip(self.state.action_values.counts())
            .map(|(estimate, count)| {
                estimate + self.reward_scale * (self.c * log_term / *count as f64).sqrt()
            })
            .collect();
        choose_argmax(&scores, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        finite("reward", reward)?;
        if action.get() >= self.n_arms() {
            return Err(PyMabError::validation(
                "action",
                format!("index {} is outside [0, {})", action.get(), self.n_arms()),
            ));
        }
        self.state.action_values.update(action, reward)?;
        if self.update_detector(action.get(), reward)? {
            self.reset_arm(action, reward)?;
        }
        Ok(())
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        self.state.action_values.recommendation()
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

macro_rules! change_detector_wrapper {
    ($name:ident, $detector:expr, $description:literal) => {
        #[doc = $description]
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name(ChangePointUCBPolicy);

        impl $name {
            /// Construct this change-detection UCB policy.
            #[allow(clippy::too_many_arguments)]
            pub fn new(
                n_arms: usize,
                initial_value: f64,
                c: f64,
                reward_scale: f64,
                threshold: f64,
                drift: f64,
                min_observations: u64,
            ) -> Result<Self> {
                ChangePointUCBPolicy::new(
                    n_arms,
                    initial_value,
                    c,
                    reward_scale,
                    $detector,
                    threshold,
                    drift,
                    min_observations,
                )
                .map(Self)
            }
        }

        impl Policy for $name {
            type State = ChangePointState;

            fn capabilities(&self) -> PolicyCapabilities {
                self.0.capabilities()
            }

            fn n_arms(&self) -> usize {
                self.0.n_arms()
            }

            fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
                self.0.select_action(rng)
            }

            fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
                self.0.update(action, reward)
            }

            fn recommend_action(&self) -> Result<ActionIndex> {
                self.0.recommend_action()
            }

            fn reset(&mut self) {
                self.0.reset();
            }

            fn state(&self) -> &Self::State {
                self.0.state()
            }

            fn estimated_state_bytes(&self) -> usize {
                self.0.estimated_state_bytes()
            }
        }
    };
}

change_detector_wrapper!(
    CUSUMUCBPolicy,
    ChangeDetector::Cusum,
    "CUSUM-triggered resetting UCB."
);
change_detector_wrapper!(
    PageHinkleyUCBPolicy,
    ChangeDetector::PageHinkley,
    "Page-Hinkley-triggered resetting UCB."
);
