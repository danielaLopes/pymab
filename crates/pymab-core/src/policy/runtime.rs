//! Object-safe policy adapters used by heterogeneous experiment runners.

use crate::error::Result;
use crate::policy::{ContextualPolicy, Policy};
use crate::rng::NativeRng;
use crate::types::{ActionIndex, ContextShape, PolicyCapabilities};

/// Object-safe runtime interface for a classic policy.
pub trait RuntimePolicy: Send + Sync {
    /// Clone configuration into a fresh reset policy.
    fn clone_reset_box(&self) -> Box<dyn RuntimePolicy>;
    /// Return compatibility metadata.
    fn capabilities(&self) -> PolicyCapabilities;
    /// Return the number of arms.
    fn n_arms(&self) -> usize;
    /// Select an action.
    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex>;
    /// Update from one observation.
    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()>;
    /// Return a non-exploratory recommendation.
    fn recommend_action(&self) -> Result<ActionIndex>;
    /// Estimate owned state bytes.
    fn estimated_state_bytes(&self) -> usize;
}

impl<T> RuntimePolicy for T
where
    T: Policy + Send + Sync + 'static,
{
    fn clone_reset_box(&self) -> Box<dyn RuntimePolicy> {
        Box::new(self.clone_reset())
    }

    fn capabilities(&self) -> PolicyCapabilities {
        Policy::capabilities(self)
    }

    fn n_arms(&self) -> usize {
        Policy::n_arms(self)
    }

    fn select_action(&mut self, rng: &mut NativeRng) -> Result<ActionIndex> {
        Policy::select_action(self, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> Result<()> {
        Policy::update(self, action, reward)
    }

    fn recommend_action(&self) -> Result<ActionIndex> {
        Policy::recommend_action(self)
    }

    fn estimated_state_bytes(&self) -> usize {
        Policy::estimated_state_bytes(self)
    }
}

/// Object-safe runtime interface for a contextual policy.
pub trait RuntimeContextualPolicy: Send + Sync {
    /// Clone configuration into a fresh reset policy.
    fn clone_reset_box(&self) -> Box<dyn RuntimeContextualPolicy>;
    /// Return compatibility metadata.
    fn capabilities(&self) -> PolicyCapabilities;
    /// Return the context shape.
    fn context_shape(&self) -> ContextShape;
    /// Select an action.
    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex>;
    /// Update from one contextual observation.
    fn update(&mut self, action: ActionIndex, reward: f64, context: &[f64]) -> Result<()>;
    /// Return a non-exploratory recommendation.
    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex>;
    /// Estimate owned state bytes.
    fn estimated_state_bytes(&self) -> usize;
}

impl<T> RuntimeContextualPolicy for T
where
    T: ContextualPolicy + Send + Sync + 'static,
{
    fn clone_reset_box(&self) -> Box<dyn RuntimeContextualPolicy> {
        Box::new(self.clone_reset())
    }

    fn capabilities(&self) -> PolicyCapabilities {
        ContextualPolicy::capabilities(self)
    }

    fn context_shape(&self) -> ContextShape {
        ContextualPolicy::context_shape(self)
    }

    fn select_action(&mut self, context: &[f64], rng: &mut NativeRng) -> Result<ActionIndex> {
        ContextualPolicy::select_action(self, context, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64, context: &[f64]) -> Result<()> {
        ContextualPolicy::update(self, action, reward, context)
    }

    fn recommend_action(&self, context: &[f64]) -> Result<ActionIndex> {
        ContextualPolicy::recommend_action(self, context)
    }

    fn estimated_state_bytes(&self) -> usize {
        ContextualPolicy::estimated_state_bytes(self)
    }
}
