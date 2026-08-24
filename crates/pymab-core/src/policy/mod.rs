//! Policy traits, shared state, and the built-in policy registry.

use std::fmt::Debug;

use crate::error::Result;
use crate::rng::NativeRng;
use crate::types::{ActionIndex, ContextShape, PolicyCapabilities};

pub mod action_value;
pub mod registry;

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
pub(crate) enum BuiltInPolicy {}

#[allow(dead_code)]
pub(crate) enum BuiltInContextualPolicy {}
