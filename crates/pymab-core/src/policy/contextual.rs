//! Linear contextual bandit policies.

use std::mem::size_of;

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use super::action_value::{choose_argmax, deterministic_argmax};
use super::ContextualPolicy;
use crate::error::{PyMabError, Result};
use crate::rng::NativeRng;
use crate::types::{
    ActionIndex, ContextShape, PolicyCapabilities, PolicyObjective, RewardDomain,
    ALL_REWARD_DOMAINS,
};
use crate::validation::{finite, probability, reward, strictly_positive};

const GENERAL_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(true, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward);
const BINARY_DOMAINS: &[RewardDomain] = &[RewardDomain::Binary];
const BINARY_CAPABILITIES: PolicyCapabilities =
    PolicyCapabilities::new(true, BINARY_DOMAINS, PolicyObjective::CumulativeReward);

/// Learned coefficients for an independent linear model per arm.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearThetaState {
    shape: ContextShape,
    theta: Vec<f64>,
}

impl LinearThetaState {
    fn new(n_arms: usize, n_features: usize) -> Result<Self> {
        let shape = ContextShape::new(n_arms, n_features)?;
        Ok(Self {
            shape,
            theta: vec![0.0; shape.element_count()],
        })
    }

    /// Return the row-major arm-by-feature coefficient matrix.
    #[must_use]
    pub fn theta(&self) -> &[f64] {
        &self.theta
    }

    fn arm_range(&self, arm: usize) -> std::ops::Range<usize> {
        let start = arm * self.shape.n_features();
        start..start + self.shape.n_features()
    }

    fn reset(&mut self) {
        self.theta.fill(0.0);
    }

    fn estimated_heap_bytes(&self) -> usize {
        self.theta.capacity() * size_of::<f64>()
    }
}

/// Linear contextual policy with epsilon-greedy exploration.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearEpsilonGreedyPolicy {
    epsilon: f64,
    learning_rate: f64,
    state: LinearThetaState,
}

impl LinearEpsilonGreedyPolicy {
    /// Construct a linear epsilon-greedy policy.
    pub fn new(n_arms: usize, n_features: usize, epsilon: f64, learning_rate: f64) -> Result<Self> {
        Ok(Self {
            epsilon: probability("epsilon", epsilon)?,
            learning_rate: strictly_positive("learning_rate", learning_rate)?,
            state: LinearThetaState::new(n_arms, n_features)?,
        })
    }

    /// Return predicted linear rewards for every arm.
    pub fn scores(&self, context: &[f64]) -> Result<Vec<f64>> {
        self.state.shape.validate_flat(context)?;
        (0..self.state.shape.n_arms())
            .map(|arm| {
                let range = self.state.arm_range(arm);
                checked_dot(
                    &self.state.theta[range.clone()],
                    &context[range],
                    "linear score",
                )
            })
            .collect()
    }
}

impl ContextualPolicy for LinearEpsilonGreedyPolicy {
    type State = LinearThetaState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn context_shape(&self) -> ContextShape {
        self.state.shape
    }

    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
        self.state.shape.validate_flat(context)?;
        if rng.random::<f64>() < self.epsilon {
            ActionIndex::new(
                rng.random_range(0..self.state.shape.n_arms()),
                self.state.shape.n_arms(),
            )
        } else {
            choose_argmax(&self.scores(context)?, rng)
        }
    }

    fn update(&mut self, action: ActionIndex, reward: f64, context: &[f64]) -> Result<()> {
        let reward = finite("reward", reward)?;
        self.state.shape.validate_flat(context)?;
        let arm = checked_action(action, self.state.shape.n_arms())?;
        let range = self.state.arm_range(arm);
        let prediction = checked_dot(
            &self.state.theta[range.clone()],
            &context[range.clone()],
            "linear prediction",
        )?;
        let error = reward - prediction;
        let scale = self.learning_rate * error;
        if !scale.is_finite() {
            return Err(PyMabError::numerical(
                "linear gradient",
                "update scale became non-finite",
            ));
        }
        let updated: Result<Vec<f64>> = self.state.theta[range.clone()]
            .iter()
            .zip(&context[range.clone()])
            .map(|(coefficient, feature)| {
                let value = coefficient + scale * feature;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(PyMabError::numerical(
                        "linear gradient",
                        "coefficient became non-finite",
                    ))
                }
            })
            .collect();
        self.state.theta[range].copy_from_slice(&updated?);
        Ok(())
    }

    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex> {
        deterministic_argmax(&self.scores(context)?)
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

/// Sufficient statistics for independent Bayesian linear models by arm.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPosteriorState {
    shape: ContextShape,
    l2: f64,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl LinearPosteriorState {
    fn new(n_arms: usize, n_features: usize, l2: f64) -> Result<Self> {
        let shape = ContextShape::new(n_arms, n_features)?;
        let matrix_elements = n_features
            .checked_mul(n_features)
            .and_then(|elements| elements.checked_mul(n_arms))
            .ok_or_else(|| PyMabError::configuration("context_shape", "matrix size overflows"))?;
        let mut state = Self {
            shape,
            l2,
            a: vec![0.0; matrix_elements],
            b: vec![0.0; shape.element_count()],
        };
        state.reset();
        Ok(state)
    }

    /// Return the concatenated column-major precision matrices, one arm at a time.
    #[must_use]
    pub fn a(&self) -> &[f64] {
        &self.a
    }

    /// Return the row-major arm-by-feature reward vectors.
    #[must_use]
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    fn feature_range(&self, arm: usize) -> std::ops::Range<usize> {
        let start = arm * self.shape.n_features();
        start..start + self.shape.n_features()
    }

    fn matrix_range(&self, arm: usize) -> std::ops::Range<usize> {
        let elements = self.shape.n_features() * self.shape.n_features();
        let start = arm * elements;
        start..start + elements
    }

    fn matrix(&self, arm: usize) -> DMatrix<f64> {
        DMatrix::from_column_slice(
            self.shape.n_features(),
            self.shape.n_features(),
            &self.a[self.matrix_range(arm)],
        )
    }

    fn vector(&self, arm: usize) -> DVector<f64> {
        DVector::from_column_slice(&self.b[self.feature_range(arm)])
    }

    fn factor(&self, arm: usize) -> Result<nalgebra::linalg::Cholesky<f64, nalgebra::Dyn>> {
        self.matrix(arm).cholesky().ok_or_else(|| {
            PyMabError::numerical(
                "Cholesky factorization",
                "precision matrix is not positive definite",
            )
        })
    }

    fn update(&mut self, arm: usize, reward: f64, features: &[f64]) -> Result<()> {
        let matrix_range = self.matrix_range(arm);
        let feature_range = self.feature_range(arm);
        let dimension = self.shape.n_features();
        let mut matrix = self.a[matrix_range.clone()].to_vec();
        let mut vector = self.b[feature_range.clone()].to_vec();

        for column in 0..dimension {
            for row in 0..dimension {
                let index = column * dimension + row;
                matrix[index] += features[row] * features[column];
                if !matrix[index].is_finite() {
                    return Err(PyMabError::numerical(
                        "linear posterior update",
                        "precision matrix became non-finite",
                    ));
                }
            }
        }
        for (value, feature) in vector.iter_mut().zip(features) {
            *value += reward * feature;
            if !value.is_finite() {
                return Err(PyMabError::numerical(
                    "linear posterior update",
                    "reward vector became non-finite",
                ));
            }
        }

        self.a[matrix_range].copy_from_slice(&matrix);
        self.b[feature_range].copy_from_slice(&vector);
        Ok(())
    }

    fn reset(&mut self) {
        self.a.fill(0.0);
        self.b.fill(0.0);
        let dimension = self.shape.n_features();
        for arm in 0..self.shape.n_arms() {
            let start = self.matrix_range(arm).start;
            for diagonal in 0..dimension {
                self.a[start + diagonal * dimension + diagonal] = self.l2;
            }
        }
    }

    fn estimated_heap_bytes(&self) -> usize {
        (self.a.capacity() + self.b.capacity()) * size_of::<f64>()
    }
}

/// Disjoint linear upper-confidence-bound policy.
#[derive(Clone, Debug, PartialEq)]
pub struct LinUCBPolicy {
    alpha: f64,
    state: LinearPosteriorState,
}

impl LinUCBPolicy {
    /// Construct a disjoint LinUCB policy.
    pub fn new(n_arms: usize, n_features: usize, alpha: f64, l2: f64) -> Result<Self> {
        Ok(Self {
            alpha: strictly_positive("alpha", alpha)?,
            state: LinearPosteriorState::new(n_arms, n_features, strictly_positive("l2", l2)?)?,
        })
    }

    /// Return posterior linear means for every arm and context row.
    pub fn posterior_means(&self, context: &[f64]) -> Result<Vec<f64>> {
        posterior_means(&self.state, context)
    }

    /// Return the current upper confidence bound for every arm.
    pub fn upper_confidence_bounds(&self, context: &[f64]) -> Result<Vec<f64>> {
        self.state.shape.validate_flat(context)?;
        let mut bounds = Vec::with_capacity(self.state.shape.n_arms());
        for arm in 0..self.state.shape.n_arms() {
            let range = self.state.feature_range(arm);
            let features = DVector::from_column_slice(&context[range]);
            let factor = self.state.factor(arm)?;
            let theta = factor.solve(&self.state.vector(arm));
            let solved_features = factor.solve(&features);
            let mean = checked_vector_dot(&theta, &features, "LinUCB mean")?;
            let variance = checked_vector_dot(&features, &solved_features, "LinUCB variance")?;
            let uncertainty = variance.max(0.0).sqrt();
            let bound = mean + self.alpha * uncertainty;
            if !bound.is_finite() {
                return Err(PyMabError::numerical(
                    "LinUCB score",
                    "upper confidence bound became non-finite",
                ));
            }
            bounds.push(bound);
        }
        Ok(bounds)
    }
}

impl ContextualPolicy for LinUCBPolicy {
    type State = LinearPosteriorState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn context_shape(&self) -> ContextShape {
        self.state.shape
    }

    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
        choose_argmax(&self.upper_confidence_bounds(context)?, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64, context: &[f64]) -> Result<()> {
        let reward = finite("reward", reward)?;
        self.state.shape.validate_flat(context)?;
        let arm = checked_action(action, self.state.shape.n_arms())?;
        let range = self.state.feature_range(arm);
        self.state.update(arm, reward, &context[range])
    }

    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex> {
        deterministic_argmax(&self.posterior_means(context)?)
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

/// Bayesian linear Thompson sampling with independent models by arm.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearThompsonSamplingPolicy {
    exploration_scale: f64,
    state: LinearPosteriorState,
}

impl LinearThompsonSamplingPolicy {
    /// Construct a Bayesian linear Thompson-sampling policy.
    pub fn new(n_arms: usize, n_features: usize, exploration_scale: f64, l2: f64) -> Result<Self> {
        Ok(Self {
            exploration_scale: strictly_positive("exploration_scale", exploration_scale)?,
            state: LinearPosteriorState::new(n_arms, n_features, strictly_positive("l2", l2)?)?,
        })
    }

    /// Return posterior linear means for every arm and context row.
    pub fn posterior_means(&self, context: &[f64]) -> Result<Vec<f64>> {
        posterior_means(&self.state, context)
    }
}

impl ContextualPolicy for LinearThompsonSamplingPolicy {
    type State = LinearPosteriorState;

    fn capabilities(&self) -> PolicyCapabilities {
        GENERAL_CAPABILITIES
    }

    fn context_shape(&self) -> ContextShape {
        self.state.shape
    }

    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
        self.state.shape.validate_flat(context)?;
        let mut scores = Vec::with_capacity(self.state.shape.n_arms());
        for arm in 0..self.state.shape.n_arms() {
            let factor = self.state.factor(arm)?;
            let mean = factor.solve(&self.state.vector(arm));
            let normal: Vec<f64> = (0..self.state.shape.n_features())
                .map(|_| StandardNormal.sample(rng))
                .collect();
            let normal = DVector::from_vec(normal);
            let noise = factor
                .l()
                .transpose()
                .solve_upper_triangular(&normal)
                .ok_or_else(|| {
                    PyMabError::numerical(
                        "linear Thompson sampling",
                        "triangular covariance solve failed",
                    )
                })?;
            let sample = mean + noise * self.exploration_scale;
            let range = self.state.feature_range(arm);
            let features = DVector::from_column_slice(&context[range]);
            scores.push(checked_vector_dot(
                &sample,
                &features,
                "linear Thompson score",
            )?);
        }
        deterministic_argmax(&scores)
    }

    fn update(&mut self, action: ActionIndex, reward: f64, context: &[f64]) -> Result<()> {
        let reward = finite("reward", reward)?;
        self.state.shape.validate_flat(context)?;
        let arm = checked_action(action, self.state.shape.n_arms())?;
        let range = self.state.feature_range(arm);
        self.state.update(arm, reward, &context[range])
    }

    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex> {
        deterministic_argmax(&self.posterior_means(context)?)
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

/// Online logistic contextual bandit for rewards in the unit interval.
#[derive(Clone, Debug, PartialEq)]
pub struct LogisticContextualBanditPolicy {
    epsilon: f64,
    learning_rate: f64,
    l2: f64,
    state: LinearThetaState,
}

impl LogisticContextualBanditPolicy {
    /// Construct an online logistic contextual policy.
    pub fn new(
        n_arms: usize,
        n_features: usize,
        epsilon: f64,
        learning_rate: f64,
        l2: f64,
    ) -> Result<Self> {
        let l2 = finite("l2", l2)?;
        if l2 < 0.0 {
            return Err(PyMabError::configuration(
                "l2",
                "must be greater than or equal to zero",
            ));
        }
        Ok(Self {
            epsilon: probability("epsilon", epsilon)?,
            learning_rate: strictly_positive("learning_rate", learning_rate)?,
            l2,
            state: LinearThetaState::new(n_arms, n_features)?,
        })
    }

    /// Return predicted probabilities for every arm.
    pub fn predicted_probabilities(&self, context: &[f64]) -> Result<Vec<f64>> {
        self.state.shape.validate_flat(context)?;
        (0..self.state.shape.n_arms())
            .map(|arm| {
                let range = self.state.arm_range(arm);
                checked_dot(
                    &self.state.theta[range.clone()],
                    &context[range],
                    "logistic score",
                )
                .map(sigmoid)
            })
            .collect()
    }
}

impl ContextualPolicy for LogisticContextualBanditPolicy {
    type State = LinearThetaState;

    fn capabilities(&self) -> PolicyCapabilities {
        BINARY_CAPABILITIES
    }

    fn context_shape(&self) -> ContextShape {
        self.state.shape
    }

    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
        self.state.shape.validate_flat(context)?;
        if rng.random::<f64>() < self.epsilon {
            ActionIndex::new(
                rng.random_range(0..self.state.shape.n_arms()),
                self.state.shape.n_arms(),
            )
        } else {
            choose_argmax(&self.predicted_probabilities(context)?, rng)
        }
    }

    fn update(&mut self, action: ActionIndex, observed_reward: f64, context: &[f64]) -> Result<()> {
        let observed_reward = reward("reward", observed_reward, RewardDomain::UnitInterval)?;
        self.state.shape.validate_flat(context)?;
        let arm = checked_action(action, self.state.shape.n_arms())?;
        let range = self.state.arm_range(arm);
        let prediction = checked_dot(
            &self.state.theta[range.clone()],
            &context[range.clone()],
            "logistic prediction",
        )?;
        let residual = observed_reward - sigmoid(prediction);
        let updated: Result<Vec<f64>> = self.state.theta[range.clone()]
            .iter()
            .zip(&context[range.clone()])
            .map(|(coefficient, feature)| {
                let gradient = residual * feature - self.l2 * coefficient;
                let value = coefficient + self.learning_rate * gradient;
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(PyMabError::numerical(
                        "logistic gradient",
                        "coefficient became non-finite",
                    ))
                }
            })
            .collect();
        self.state.theta[range].copy_from_slice(&updated?);
        Ok(())
    }

    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex> {
        deterministic_argmax(&self.predicted_probabilities(context)?)
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

fn posterior_means(state: &LinearPosteriorState, context: &[f64]) -> Result<Vec<f64>> {
    state.shape.validate_flat(context)?;
    let mut means = Vec::with_capacity(state.shape.n_arms());
    for arm in 0..state.shape.n_arms() {
        let theta = state.factor(arm)?.solve(&state.vector(arm));
        let range = state.feature_range(arm);
        let features = DVector::from_column_slice(&context[range]);
        means.push(checked_vector_dot(&theta, &features, "posterior mean")?);
    }
    Ok(means)
}

fn checked_dot(left: &[f64], right: &[f64], operation: &str) -> Result<f64> {
    let mut total = 0.0;
    for (left, right) in left.iter().zip(right) {
        let product = left * right;
        total += product;
        if !product.is_finite() || !total.is_finite() {
            return Err(PyMabError::numerical(
                operation,
                "dot product became non-finite",
            ));
        }
    }
    Ok(total)
}

fn checked_vector_dot(left: &DVector<f64>, right: &DVector<f64>, operation: &str) -> Result<f64> {
    checked_dot(left.as_slice(), right.as_slice(), operation)
}

fn checked_action(action: ActionIndex, n_arms: usize) -> Result<usize> {
    if action.get() < n_arms {
        Ok(action.get())
    } else {
        Err(PyMabError::validation(
            "action",
            format!("index {} is outside [0, {n_arms})", action.get()),
        ))
    }
}

fn sigmoid(value: f64) -> f64 {
    let clipped = value.clamp(-35.0, 35.0);
    1.0 / (1.0 + (-clipped).exp())
}
