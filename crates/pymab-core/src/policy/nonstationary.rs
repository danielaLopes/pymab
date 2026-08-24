//! Policies for non-stationary reward processes.

use std::collections::VecDeque;
use std::mem::size_of;

use rand_distr::{Beta, Distribution};

use super::action_value::{choose_argmax, deterministic_argmax};
use super::Policy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{
    ActionIndex, PolicyCapabilities, PolicyObjective, RewardDomain, ALL_REWARD_DOMAINS,
};
use crate::validation::{finite, reward, strictly_positive};

const GENERAL_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);
const BINARY_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary];
const BINARY_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(false, BINARY_DOMAINS, PolicyObjective::CumulativeReward);

type Observation = (u64, usize, f64);

/// Sliding-window UCB state.
#[derive(Clone, Debug, PartialEq)]
pub struct SlidingWindowUCBState {
    initial_value: f64,
    step: u64,
    total_reward: f64,
    counts: Vec<u64>,
    estimates: Vec<f64>,
    history: VecDeque<Observation>,
}

impl SlidingWindowUCBState {
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }

    #[must_use]
    pub const fn total_reward(&self) -> f64 {
        self.total_reward
    }

    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    #[must_use]
    pub fn estimates(&self) -> &[f64] {
        &self.estimates
    }

    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.counts.capacity() * size_of::<u64>()
            + self.estimates.capacity() * size_of::<f64>()
            + self.history.capacity() * size_of::<Observation>()
    }
}

/// UCB over observations from the most recent global time steps.
#[derive(Clone, Debug, PartialEq)]
pub struct SlidingWindowUCBPolicy {
    c: f64,
    reward_scale: f64,
    window_size: usize,
    state: SlidingWindowUCBState,
}

impl SlidingWindowUCBPolicy {
    pub fn new(
        n_arms: usize,
        initial_value: f64,
        c: f64,
        reward_scale: f64,
        window_size: usize,
    ) -> Result<Self> {
        if n_arms == 0 || window_size == 0 {
            return Err(PyMabError::configuration(
                if n_arms == 0 { "n_arms" } else { "window_size" },
                "must be greater than zero",
            ));
        }
        if !initial_value.is_finite() {
            return Err(PyMabError::configuration("initial_value", "must be finite"));
        }
        Ok(Self {
            c: strictly_positive("c", c)?,
            reward_scale: strictly_positive("reward_scale", reward_scale)?,
            window_size,
            state: SlidingWindowUCBState {
                initial_value,
                step: 0,
                total_reward: 0.0,
                counts: vec![0; n_arms],
                estimates: vec![initial_value; n_arms],
                history: VecDeque::with_capacity(window_size),
            },
        })
    }

    #[must_use]
    pub fn confidence_bonus(&self) -> Vec<f64> {
        let horizon = self.state.step.min(self.window_size as u64).max(2) as f64;
        self.state
            .counts
            .iter()
            .map(|count| {
                self.reward_scale * (self.c * horizon.ln() / (*count).max(1) as f64).sqrt()
            })
            .collect()
    }

    fn prospective_window_state(&self, index: usize, reward: f64) -> Result<(Vec<u64>, Vec<f64>)> {
        let mut counts = vec![0_u64; self.n_arms()];
        let mut sums = vec![0.0; self.n_arms()];
        let skip = usize::from(self.state.history.len() == self.window_size);
        for (_, action, value) in self.state.history.iter().skip(skip) {
            counts[*action] += 1;
            sums[*action] += value;
            if !sums[*action].is_finite() {
                return Err(PyMabError::numerical(
                    "sliding-window reward accumulation",
                    "arm sum overflowed",
                ));
            }
        }
        counts[index] += 1;
        sums[index] += reward;
        if !sums[index].is_finite() {
            return Err(PyMabError::numerical(
                "sliding-window reward accumulation",
                "arm sum overflowed",
            ));
        }
        Ok((counts, sums))
    }

    fn refresh(&mut self, counts: Vec<u64>, sums: Vec<f64>) {
        self.state.counts = counts;
        self.state.estimates.fill(self.state.initial_value);
        for (arm, count) in self.state.counts.iter().enumerate() {
            if *count > 0 {
                self.state.estimates[arm] = sums[arm] / *count as f64;
            }
        }
    }
}

impl Policy for SlidingWindowUCBPolicy {
    type State = SlidingWindowUCBState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn n_arms(&self) -> usize {
        self.state.counts.len()
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if let Some(index) = self.state.counts.iter().position(|count| *count == 0) {
            return ActionIndex::new(index, self.n_arms());
        }
        let scores: Vec<_> = self
            .state
            .estimates
            .iter()
            .zip(self.confidence_bonus())
            .map(|(estimate, bonus)| estimate + bonus)
            .collect();
        choose_argmax(&scores, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        finite("reward", reward)?;
        let index = checked_action(action, self.n_arms())?;
        let step = self
            .state
            .step
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("policy step counter overflowed"))?;
        let total_reward = self.state.total_reward + reward;
        if !total_reward.is_finite() {
            return Err(PyMabError::numerical(
                "reward accumulation",
                "total reward overflowed",
            ));
        }
        let (counts, sums) = self.prospective_window_state(index, reward)?;
        if self.state.history.len() == self.window_size {
            self.state.history.pop_front();
        }
        self.state.history.push_back((step, index, reward));
        self.state.step = step;
        self.state.total_reward = total_reward;
        self.refresh(counts, sums);
        Ok(())
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.state.estimates)
    }

    fn reset(&mut self) {
        self.state.step = 0;
        self.state.total_reward = 0.0;
        self.state.counts.fill(0);
        self.state.estimates.fill(self.state.initial_value);
        self.state.history.clear();
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

/// Exponentially discounted UCB state.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscountedUCBState {
    initial_value: f64,
    step: u64,
    total_reward: f64,
    counts: Vec<u64>,
    estimates: Vec<f64>,
    discounted_counts: Vec<f64>,
    discounted_sums: Vec<f64>,
}

impl DiscountedUCBState {
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }
    #[must_use]
    pub const fn total_reward(&self) -> f64 {
        self.total_reward
    }
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }
    #[must_use]
    pub fn estimates(&self) -> &[f64] {
        &self.estimates
    }
    #[must_use]
    pub fn discounted_counts(&self) -> &[f64] {
        &self.discounted_counts
    }
    #[must_use]
    pub fn discounted_sums(&self) -> &[f64] {
        &self.discounted_sums
    }
    fn estimated_heap_bytes(&self) -> usize {
        self.counts.capacity() * size_of::<u64>()
            + (self.estimates.capacity()
                + self.discounted_counts.capacity()
                + self.discounted_sums.capacity())
                * size_of::<f64>()
    }
}

/// UCB with exponential forgetting.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscountedUCBPolicy {
    c: f64,
    reward_scale: f64,
    discount_factor: f64,
    state: DiscountedUCBState,
}

impl DiscountedUCBPolicy {
    pub fn new(
        n_arms: usize,
        initial_value: f64,
        c: f64,
        reward_scale: f64,
        discount_factor: f64,
    ) -> Result<Self> {
        if n_arms == 0 {
            return Err(PyMabError::configuration(
                "n_arms",
                "must be greater than zero",
            ));
        }
        if !initial_value.is_finite() {
            return Err(PyMabError::configuration("initial_value", "must be finite"));
        }
        let discount_factor = open_unit("discount_factor", discount_factor)?;
        Ok(Self {
            c: strictly_positive("c", c)?,
            reward_scale: strictly_positive("reward_scale", reward_scale)?,
            discount_factor,
            state: DiscountedUCBState {
                initial_value,
                step: 0,
                total_reward: 0.0,
                counts: vec![0; n_arms],
                estimates: vec![initial_value; n_arms],
                discounted_counts: vec![0.0; n_arms],
                discounted_sums: vec![0.0; n_arms],
            },
        })
    }

    #[must_use]
    pub fn confidence_bonus(&self) -> Vec<f64> {
        let horizon = self.state.discounted_counts.iter().sum::<f64>().max(2.0);
        self.state
            .discounted_counts
            .iter()
            .map(|count| self.reward_scale * (self.c * horizon.ln() / count.max(1e-12)).sqrt())
            .collect()
    }
}

impl Policy for DiscountedUCBPolicy {
    type State = DiscountedUCBState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }
    fn n_arms(&self) -> usize {
        self.state.counts.len()
    }
    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        if let Some(index) = self.state.counts.iter().position(|count| *count == 0) {
            return ActionIndex::new(index, self.n_arms());
        }
        let scores: Vec<_> = self
            .state
            .estimates
            .iter()
            .zip(self.confidence_bonus())
            .map(|(estimate, bonus)| estimate + bonus)
            .collect();
        choose_argmax(&scores, rng)
    }
    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        finite("reward", reward)?;
        let index = checked_action(action, self.n_arms())?;
        let step = self
            .state
            .step
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("policy step counter overflowed"))?;
        let total_reward = self.state.total_reward + reward;
        if !total_reward.is_finite() {
            return Err(PyMabError::numerical(
                "reward accumulation",
                "total reward overflowed",
            ));
        }
        let count = self.state.counts[index]
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("arm pull counter overflowed"))?;
        let selected_sum = self.state.discounted_sums[index] * self.discount_factor + reward;
        if !selected_sum.is_finite() {
            return Err(PyMabError::numerical(
                "discounted reward accumulation",
                "discounted sum overflowed",
            ));
        }

        for arm in 0..self.n_arms() {
            self.state.discounted_counts[arm] *= self.discount_factor;
            self.state.discounted_sums[arm] *= self.discount_factor;
        }
        self.state.discounted_counts[index] += 1.0;
        self.state.discounted_sums[index] = selected_sum;
        for arm in 0..self.n_arms() {
            if self.state.discounted_counts[arm] > 0.0 {
                self.state.estimates[arm] =
                    self.state.discounted_sums[arm] / self.state.discounted_counts[arm];
            }
        }
        self.state.step = step;
        self.state.total_reward = total_reward;
        self.state.counts[index] = count;
        Ok(())
    }
    fn recommend_action(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.state.estimates)
    }
    fn reset(&mut self) {
        self.state.step = 0;
        self.state.total_reward = 0.0;
        self.state.counts.fill(0);
        self.state.estimates.fill(self.state.initial_value);
        self.state.discounted_counts.fill(0.0);
        self.state.discounted_sums.fill(0.0);
    }
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

/// Sliding-window Bernoulli posterior state.
#[derive(Clone, Debug, PartialEq)]
pub struct SlidingWindowBernoulliState {
    step: u64,
    total_reward: f64,
    counts: Vec<u64>,
    estimates: Vec<f64>,
    successes: Vec<u64>,
    failures: Vec<u64>,
    history: VecDeque<Observation>,
}

impl SlidingWindowBernoulliState {
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }
    #[must_use]
    pub const fn total_reward(&self) -> f64 {
        self.total_reward
    }
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }
    #[must_use]
    pub fn estimates(&self) -> &[f64] {
        &self.estimates
    }
    #[must_use]
    pub fn successes(&self) -> &[u64] {
        &self.successes
    }
    #[must_use]
    pub fn failures(&self) -> &[u64] {
        &self.failures
    }
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }
    fn estimated_heap_bytes(&self) -> usize {
        (self.counts.capacity() + self.successes.capacity() + self.failures.capacity())
            * size_of::<u64>()
            + self.estimates.capacity() * size_of::<f64>()
            + self.history.capacity() * size_of::<Observation>()
    }
}

/// Beta-Bernoulli sampling over a bounded global-time window.
#[derive(Clone, Debug, PartialEq)]
pub struct SlidingWindowBernoulliThompsonSamplingPolicy {
    alpha_prior: f64,
    beta_prior: f64,
    window_size: usize,
    state: SlidingWindowBernoulliState,
}

impl SlidingWindowBernoulliThompsonSamplingPolicy {
    pub fn new(
        n_arms: usize,
        alpha_prior: f64,
        beta_prior: f64,
        window_size: usize,
    ) -> Result<Self> {
        if n_arms == 0 || window_size == 0 {
            return Err(PyMabError::configuration(
                if n_arms == 0 { "n_arms" } else { "window_size" },
                "must be greater than zero",
            ));
        }
        Ok(Self {
            alpha_prior: strictly_positive("alpha_prior", alpha_prior)?,
            beta_prior: strictly_positive("beta_prior", beta_prior)?,
            window_size,
            state: SlidingWindowBernoulliState {
                step: 0,
                total_reward: 0.0,
                counts: vec![0; n_arms],
                estimates: vec![0.0; n_arms],
                successes: vec![0; n_arms],
                failures: vec![0; n_arms],
                history: VecDeque::with_capacity(window_size),
            },
        })
    }
    fn refresh(&mut self) {
        self.state.counts.fill(0);
        self.state.successes.fill(0);
        self.state.failures.fill(0);
        self.state.estimates.fill(0.0);
        for (_, action, value) in &self.state.history {
            self.state.counts[*action] += 1;
            if *value == 1.0 {
                self.state.successes[*action] += 1;
            } else {
                self.state.failures[*action] += 1;
            }
        }
        for arm in 0..self.n_arms() {
            if self.state.counts[arm] > 0 {
                self.state.estimates[arm] =
                    self.state.successes[arm] as f64 / self.state.counts[arm] as f64;
            }
        }
    }
}

impl Policy for SlidingWindowBernoulliThompsonSamplingPolicy {
    type State = SlidingWindowBernoulliState;
    fn capabilities(&self) -> PolicyCapabilities {
        BINARY_CAPABILITIES
    }
    fn n_arms(&self) -> usize {
        self.state.counts.len()
    }
    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let samples: Result<Vec<_>> = self
            .state
            .successes
            .iter()
            .zip(&self.state.failures)
            .map(|(s, f)| {
                let distribution =
                    Beta::new(self.alpha_prior + *s as f64, self.beta_prior + *f as f64).map_err(
                        |error| PyMabError::numerical("Beta sampling", error.to_string()),
                    )?;
                Ok(distribution.sample(rng))
            })
            .collect();
        deterministic_argmax(&samples?)
    }
    fn update(&mut self, action: ActionIndex, observed_reward: f64) -> Result<()> {
        let observed_reward = reward("reward", observed_reward, RewardDomain::Binary)?;
        let index = checked_action(action, self.n_arms())?;
        let step = self
            .state
            .step
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("policy step counter overflowed"))?;
        if self.state.history.len() == self.window_size {
            self.state.history.pop_front();
        }
        self.state.history.push_back((step, index, observed_reward));
        self.state.step = step;
        self.state.total_reward += observed_reward;
        self.refresh();
        Ok(())
    }
    fn recommend_action(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.state.estimates)
    }
    fn reset(&mut self) {
        self.state.step = 0;
        self.state.total_reward = 0.0;
        self.state.counts.fill(0);
        self.state.estimates.fill(0.0);
        self.state.successes.fill(0);
        self.state.failures.fill(0);
        self.state.history.clear();
    }
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

/// Exponentially discounted Bernoulli posterior state.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscountedBernoulliState {
    step: u64,
    total_reward: f64,
    counts: Vec<f64>,
    estimates: Vec<f64>,
    successes: Vec<f64>,
    failures: Vec<f64>,
}
impl DiscountedBernoulliState {
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }
    #[must_use]
    pub const fn total_reward(&self) -> f64 {
        self.total_reward
    }
    #[must_use]
    pub fn counts(&self) -> &[f64] {
        &self.counts
    }
    #[must_use]
    pub fn estimates(&self) -> &[f64] {
        &self.estimates
    }
    #[must_use]
    pub fn successes(&self) -> &[f64] {
        &self.successes
    }
    #[must_use]
    pub fn failures(&self) -> &[f64] {
        &self.failures
    }
    fn estimated_heap_bytes(&self) -> usize {
        (self.counts.capacity()
            + self.estimates.capacity()
            + self.successes.capacity()
            + self.failures.capacity())
            * size_of::<f64>()
    }
}

/// Beta-Bernoulli Thompson sampling with exponential forgetting.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscountedBernoulliThompsonSamplingPolicy {
    alpha_prior: f64,
    beta_prior: f64,
    discount_factor: f64,
    state: DiscountedBernoulliState,
}
impl DiscountedBernoulliThompsonSamplingPolicy {
    pub fn new(
        n_arms: usize,
        alpha_prior: f64,
        beta_prior: f64,
        discount_factor: f64,
    ) -> Result<Self> {
        if n_arms == 0 {
            return Err(PyMabError::configuration(
                "n_arms",
                "must be greater than zero",
            ));
        }
        Ok(Self {
            alpha_prior: strictly_positive("alpha_prior", alpha_prior)?,
            beta_prior: strictly_positive("beta_prior", beta_prior)?,
            discount_factor: open_unit("discount_factor", discount_factor)?,
            state: DiscountedBernoulliState {
                step: 0,
                total_reward: 0.0,
                counts: vec![0.0; n_arms],
                estimates: vec![0.0; n_arms],
                successes: vec![0.0; n_arms],
                failures: vec![0.0; n_arms],
            },
        })
    }
}
impl Policy for DiscountedBernoulliThompsonSamplingPolicy {
    type State = DiscountedBernoulliState;
    fn capabilities(&self) -> PolicyCapabilities {
        BINARY_CAPABILITIES
    }
    fn n_arms(&self) -> usize {
        self.state.counts.len()
    }
    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        let samples: Result<Vec<_>> = self
            .state
            .successes
            .iter()
            .zip(&self.state.failures)
            .map(|(s, f)| {
                let distribution = Beta::new(self.alpha_prior + s, self.beta_prior + f)
                    .map_err(|error| PyMabError::numerical("Beta sampling", error.to_string()))?;
                Ok(distribution.sample(rng))
            })
            .collect();
        deterministic_argmax(&samples?)
    }
    fn update(&mut self, action: ActionIndex, observed_reward: f64) -> Result<()> {
        let observed_reward = reward("reward", observed_reward, RewardDomain::Binary)?;
        let index = checked_action(action, self.n_arms())?;
        self.state.step = self
            .state
            .step
            .checked_add(1)
            .ok_or_else(|| PyMabError::internal("policy step counter overflowed"))?;
        self.state.total_reward += observed_reward;
        for arm in 0..self.n_arms() {
            self.state.counts[arm] *= self.discount_factor;
            self.state.successes[arm] *= self.discount_factor;
            self.state.failures[arm] *= self.discount_factor;
        }
        self.state.counts[index] += 1.0;
        if observed_reward == 1.0 {
            self.state.successes[index] += 1.0;
        } else {
            self.state.failures[index] += 1.0;
        }
        for arm in 0..self.n_arms() {
            if self.state.counts[arm] > 0.0 {
                self.state.estimates[arm] = self.state.successes[arm] / self.state.counts[arm];
            }
        }
        Ok(())
    }
    fn recommend_action(&self) -> Result<ActionIndex> {
        deterministic_argmax(&self.state.estimates)
    }
    fn reset(&mut self) {
        self.state.step = 0;
        self.state.total_reward = 0.0;
        self.state.counts.fill(0.0);
        self.state.estimates.fill(0.0);
        self.state.successes.fill(0.0);
        self.state.failures.fill(0.0);
    }
    fn state(&self) -> &Self::State {
        &self.state
    }
    fn estimated_state_bytes(&self) -> usize {
        size_of::<Self>() + self.state.estimated_heap_bytes()
    }
}

fn checked_action(action: ActionIndex, n_arms: usize) -> Result<usize> {
    if action.get() >= n_arms {
        Err(PyMabError::validation(
            "action",
            format!("index {} is outside [0, {n_arms})", action.get()),
        ))
    } else {
        Ok(action.get())
    }
}

fn open_unit(name: &str, value: f64) -> Result<f64> {
    if value.is_finite() && value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err(PyMabError::configuration(
            name,
            "must be strictly between zero and one",
        ))
    }
}
