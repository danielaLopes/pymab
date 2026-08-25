//! Policy traits, shared state, and the built-in policy registry.

use std::fmt::Debug;

use crate::error::Result;
use crate::rng::NativeRng;
use crate::types::{ActionIndex, ContextShape, PolicyCapabilities};

pub mod action_value;
pub mod adversarial;
pub mod basic;
pub mod bayesian_ucb;
pub mod change_detection;
pub mod contextual;
pub mod epsilon_greedy;
pub mod gradient;
pub mod nonstationary;
pub mod pure_exploration;
pub mod registry;
pub mod runtime;
pub mod softmax;
pub mod thompson;
pub mod ucb;

/// Contract implemented by non-contextual policies.
pub trait Policy: Clone {
    /// Learned state exposed for inspection and parity testing.
    type State: Clone + Debug + PartialEq;

    /// Return static compatibility metadata.
    fn capabilities(&self) -> PolicyCapabilities;

    /// Return the number of available arms.
    fn n_arms(&self) -> usize;

    /// Select an action, including any exploratory randomization.
    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex>;

    /// Update learned state from one observed action and reward.
    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()>;

    /// Return a best-arm recommendation without exploration.
    fn recommend_action(&self) -> Result<ActionIndex>;

    /// Reset all learned state while preserving configuration.
    fn reset(&mut self);

    /// Return the current learned state.
    fn state(&self) -> &Self::State;

    /// Estimate owned state bytes, including reserved vector capacity.
    fn estimated_state_bytes(&self) -> usize;

    /// Clone configuration into a fresh, reset policy.
    fn clone_reset(&self) -> Self {
        let mut cloned = self.clone();
        cloned.reset();
        cloned
    }
}

/// Contract implemented by contextual policies.
pub trait ContextualPolicy: Clone {
    /// Learned state exposed for inspection and parity testing.
    type State: Clone + Debug + PartialEq;

    /// Return static compatibility metadata.
    fn capabilities(&self) -> PolicyCapabilities;

    /// Return the required context shape.
    fn context_shape(&self) -> ContextShape;

    /// Select an action for a row-major arm-by-feature context.
    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex>;

    /// Update learned state from one contextual observation.
    fn update(&mut self, action: ActionIndex, reward: f64, context: &[f64]) -> Result<()>;

    /// Return a recommendation for a row-major context without exploration.
    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex>;

    /// Reset all learned state while preserving configuration.
    fn reset(&mut self);

    /// Return the current learned state.
    fn state(&self) -> &Self::State;

    /// Estimate owned state bytes, including reserved vector capacity.
    fn estimated_state_bytes(&self) -> usize;

    /// Clone configuration into a fresh, reset policy.
    fn clone_reset(&self) -> Self {
        let mut cloned = self.clone();
        cloned.reset();
        cloned
    }
}

// These enums gain concrete variants as built-ins are ported. Keeping classic
// and contextual dispatch separate makes invalid runner combinations
// unrepresentable without imposing virtual dispatch on the hot loop.
#[allow(dead_code)]
pub(crate) enum BuiltInPolicy {
    Random(basic::RandomPolicy),
    Greedy(basic::GreedyPolicy),
    EpsilonGreedy(epsilon_greedy::EpsilonGreedyPolicy),
    DecayingEpsilonGreedy(epsilon_greedy::DecayingEpsilonGreedyPolicy),
    Softmax(softmax::SoftmaxPolicy),
    Ucb(ucb::UCBPolicy),
    KlUcb(ucb::KLUCBPolicy),
    Moss(ucb::MOSSPolicy),
    Gradient(gradient::GradientBanditPolicy),
    BernoulliThompson(thompson::BernoulliThompsonSamplingPolicy),
    GaussianThompson(thompson::GaussianThompsonSamplingPolicy),
    BernoulliBayesianUcb(bayesian_ucb::BernoulliBayesianUCBPolicy),
    GaussianBayesianUcb(bayesian_ucb::GaussianBayesianUCBPolicy),
    Exp3(adversarial::EXP3Policy),
    SuccessiveElimination(pure_exploration::SuccessiveEliminationPolicy),
    MedianElimination(pure_exploration::MedianEliminationPolicy),
    SlidingWindowUcb(nonstationary::SlidingWindowUCBPolicy),
    DiscountedUcb(nonstationary::DiscountedUCBPolicy),
    SlidingWindowBernoulliThompson(nonstationary::SlidingWindowBernoulliThompsonSamplingPolicy),
    DiscountedBernoulliThompson(nonstationary::DiscountedBernoulliThompsonSamplingPolicy),
    ChangePointUcb(change_detection::ChangePointUCBPolicy),
    CusumUcb(change_detection::CUSUMUCBPolicy),
    PageHinkleyUcb(change_detection::PageHinkleyUCBPolicy),
}

#[allow(dead_code)]
pub(crate) enum BuiltInContextualPolicy {
    LinearEpsilonGreedy(contextual::LinearEpsilonGreedyPolicy),
    LinUcb(contextual::LinUCBPolicy),
    LinearThompson(contextual::LinearThompsonSamplingPolicy),
    Logistic(contextual::LogisticContextualBanditPolicy),
}
