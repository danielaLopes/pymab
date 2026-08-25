//! Opaque Python handle for every built-in Rust policy.

use std::collections::BTreeSet;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde_json::{json, Map, Value};

use pymab::policy::action_value::ActionValueState;
use pymab::policy::adversarial::{EXP3Policy, EXP3State};
use pymab::policy::basic::{GreedyPolicy, RandomPolicy};
use pymab::policy::bayesian_ucb::{BernoulliBayesianUCBPolicy, GaussianBayesianUCBPolicy};
use pymab::policy::change_detection::{
    CUSUMUCBPolicy, ChangeDetector, ChangePointState, ChangePointUCBPolicy, PageHinkleyUCBPolicy,
};
use pymab::policy::contextual::{
    LinUCBPolicy, LinearEpsilonGreedyPolicy, LinearPosteriorState, LinearThetaState,
    LinearThompsonSamplingPolicy, LogisticContextualBanditPolicy,
};
use pymab::policy::epsilon_greedy::{DecayingEpsilonGreedyPolicy, EpsilonGreedyPolicy};
use pymab::policy::gradient::{GradientBanditPolicy, GradientState};
use pymab::policy::nonstationary::{
    DiscountedBernoulliState, DiscountedBernoulliThompsonSamplingPolicy, DiscountedUCBPolicy,
    DiscountedUCBState, SlidingWindowBernoulliState, SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy, SlidingWindowUCBState,
};
use pymab::policy::pure_exploration::{
    MedianEliminationPolicy, MedianEliminationState, SuccessiveEliminationPolicy,
    SuccessiveEliminationState,
};
use pymab::policy::softmax::SoftmaxPolicy;
use pymab::policy::thompson::{
    BernoulliPosteriorState, BernoulliThompsonSamplingPolicy, GaussianPosteriorState,
    GaussianThompsonSamplingPolicy,
};
use pymab::policy::ucb::{KLUCBPolicy, MOSSPolicy, UCBPolicy};
use pymab::policy::{ContextualPolicy, Policy};
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::ActionIndex;

use crate::error::to_python;

trait StateSnapshot {
    fn snapshot(&self) -> Value;
}

fn action_value_snapshot(state: &ActionValueState) -> Value {
    json!({
        "step": state.step(),
        "total_reward": state.total_reward(),
        "counts": state.counts(),
        "estimates": state.estimates(),
    })
}

fn extend_action_value(state: &ActionValueState, fields: Value) -> Value {
    let mut snapshot = action_value_snapshot(state);
    if let (Some(target), Some(source)) = (snapshot.as_object_mut(), fields.as_object()) {
        target.extend(source.clone());
    }
    snapshot
}

impl StateSnapshot for ActionValueState {
    fn snapshot(&self) -> Value {
        action_value_snapshot(self)
    }
}

impl StateSnapshot for GradientState {
    fn snapshot(&self) -> Value {
        json!({
            "step": self.step(),
            "average_reward": self.average_reward(),
            "preferences": self.preferences(),
            "probabilities": self.probabilities(),
        })
    }
}

impl StateSnapshot for BernoulliPosteriorState {
    fn snapshot(&self) -> Value {
        extend_action_value(
            self.action_values(),
            json!({"successes": self.successes(), "failures": self.failures()}),
        )
    }
}

impl StateSnapshot for GaussianPosteriorState {
    fn snapshot(&self) -> Value {
        extend_action_value(
            self.action_values(),
            json!({"means": self.means(), "precisions": self.precisions()}),
        )
    }
}

impl StateSnapshot for EXP3State {
    fn snapshot(&self) -> Value {
        extend_action_value(
            self.action_values(),
            json!({
                "log_weights": self.log_weights(),
                "last_probabilities": self.last_probabilities(),
            }),
        )
    }
}

impl StateSnapshot for SuccessiveEliminationState {
    fn snapshot(&self) -> Value {
        extend_action_value(self.action_values(), json!({"active": self.active()}))
    }
}

impl StateSnapshot for MedianEliminationState {
    fn snapshot(&self) -> Value {
        extend_action_value(
            self.action_values(),
            json!({
                "active": self.active(),
                "phase_counts": self.phase_counts(),
                "phase_sums": self.phase_sums(),
                "phase_epsilon": self.phase_epsilon(),
                "phase_delta": self.phase_delta(),
            }),
        )
    }
}

impl StateSnapshot for SlidingWindowUCBState {
    fn snapshot(&self) -> Value {
        json!({
            "step": self.step(),
            "total_reward": self.total_reward(),
            "counts": self.counts(),
            "estimates": self.estimates(),
            "history_len": self.history_len(),
        })
    }
}

impl StateSnapshot for DiscountedUCBState {
    fn snapshot(&self) -> Value {
        json!({
            "step": self.step(),
            "total_reward": self.total_reward(),
            "counts": self.counts(),
            "estimates": self.estimates(),
            "discounted_counts": self.discounted_counts(),
            "discounted_sums": self.discounted_sums(),
        })
    }
}

impl StateSnapshot for SlidingWindowBernoulliState {
    fn snapshot(&self) -> Value {
        json!({
            "step": self.step(),
            "total_reward": self.total_reward(),
            "counts": self.counts(),
            "estimates": self.estimates(),
            "successes": self.successes(),
            "failures": self.failures(),
            "history_len": self.history_len(),
        })
    }
}

impl StateSnapshot for DiscountedBernoulliState {
    fn snapshot(&self) -> Value {
        json!({
            "step": self.step(),
            "total_reward": self.total_reward(),
            "counts": self.counts(),
            "estimates": self.estimates(),
            "successes": self.successes(),
            "failures": self.failures(),
        })
    }
}

impl StateSnapshot for ChangePointState {
    fn snapshot(&self) -> Value {
        extend_action_value(
            self.action_values(),
            json!({
                "detector_counts": self.detector_counts(),
                "detector_means": self.detector_means(),
                "positive_cusum": self.positive_cusum(),
                "negative_cusum": self.negative_cusum(),
                "ph_cumulative": self.ph_cumulative(),
                "ph_minimum": self.ph_minimum(),
                "change_counts": self.change_counts(),
            }),
        )
    }
}

impl StateSnapshot for LinearThetaState {
    fn snapshot(&self) -> Value {
        json!({"theta": self.theta()})
    }
}

impl StateSnapshot for LinearPosteriorState {
    fn snapshot(&self) -> Value {
        json!({"a": self.a(), "b": self.b()})
    }
}

trait ErasedPolicy: Send + Sync {
    fn clone_reset_box(&self) -> Box<dyn ErasedPolicy>;
    fn clone_runtime_box(&self) -> Box<dyn pymab::policy::runtime::RuntimePolicy>;
    fn n_arms(&self) -> usize;
    fn select_action(
        &mut self,
        rng: &mut pymab::rng::NativeRng,
    ) -> pymab::error::Result<ActionIndex>;
    fn update(&mut self, action: ActionIndex, reward: f64) -> pymab::error::Result<()>;
    fn recommend_action(&self) -> pymab::error::Result<ActionIndex>;
    fn reset(&mut self);
    fn snapshot(&self) -> Value;
    fn estimated_state_bytes(&self) -> usize;
}

impl<T> ErasedPolicy for T
where
    T: Policy + Send + Sync + 'static,
    T::State: StateSnapshot,
{
    fn clone_reset_box(&self) -> Box<dyn ErasedPolicy> {
        Box::new(self.clone_reset())
    }

    fn clone_runtime_box(&self) -> Box<dyn pymab::policy::runtime::RuntimePolicy> {
        Box::new(self.clone_reset())
    }

    fn n_arms(&self) -> usize {
        Policy::n_arms(self)
    }

    fn select_action(
        &mut self,
        rng: &mut pymab::rng::NativeRng,
    ) -> pymab::error::Result<ActionIndex> {
        Policy::select_action(self, rng)
    }

    fn update(&mut self, action: ActionIndex, reward: f64) -> pymab::error::Result<()> {
        Policy::update(self, action, reward)
    }

    fn recommend_action(&self) -> pymab::error::Result<ActionIndex> {
        Policy::recommend_action(self)
    }

    fn reset(&mut self) {
        Policy::reset(self);
    }

    fn snapshot(&self) -> Value {
        self.state().snapshot()
    }

    fn estimated_state_bytes(&self) -> usize {
        Policy::estimated_state_bytes(self)
    }
}

trait ErasedContextualPolicy: Send + Sync {
    fn clone_reset_box(&self) -> Box<dyn ErasedContextualPolicy>;
    fn clone_runtime_box(&self) -> Box<dyn pymab::policy::runtime::RuntimeContextualPolicy>;
    fn n_arms(&self) -> usize;
    fn select_action(
        &mut self,
        context: &[f64],
        rng: &mut pymab::rng::NativeRng,
    ) -> pymab::error::Result<ActionIndex>;
    fn update(
        &mut self,
        action: ActionIndex,
        reward: f64,
        context: &[f64],
    ) -> pymab::error::Result<()>;
    fn recommend_action(&self, context: &[f64]) -> pymab::error::Result<ActionIndex>;
    fn reset(&mut self);
    fn snapshot(&self) -> Value;
    fn estimated_state_bytes(&self) -> usize;
}

impl<T> ErasedContextualPolicy for T
where
    T: ContextualPolicy + Send + Sync + 'static,
    T::State: StateSnapshot,
{
    fn clone_reset_box(&self) -> Box<dyn ErasedContextualPolicy> {
        Box::new(self.clone_reset())
    }

    fn clone_runtime_box(&self) -> Box<dyn pymab::policy::runtime::RuntimeContextualPolicy> {
        Box::new(self.clone_reset())
    }

    fn n_arms(&self) -> usize {
        self.context_shape().n_arms()
    }

    fn select_action(
        &mut self,
        context: &[f64],
        rng: &mut pymab::rng::NativeRng,
    ) -> pymab::error::Result<ActionIndex> {
        ContextualPolicy::select_action(self, context, rng)
    }

    fn update(
        &mut self,
        action: ActionIndex,
        reward: f64,
        context: &[f64],
    ) -> pymab::error::Result<()> {
        ContextualPolicy::update(self, action, reward, context)
    }

    fn recommend_action(&self, context: &[f64]) -> pymab::error::Result<ActionIndex> {
        ContextualPolicy::recommend_action(self, context)
    }

    fn reset(&mut self) {
        ContextualPolicy::reset(self);
    }

    fn snapshot(&self) -> Value {
        self.state().snapshot()
    }

    fn estimated_state_bytes(&self) -> usize {
        ContextualPolicy::estimated_state_bytes(self)
    }
}

enum PolicyHandle {
    Classic(Box<dyn ErasedPolicy>),
    Contextual(Box<dyn ErasedContextualPolicy>),
}

impl PolicyHandle {
    fn clone_reset(&self) -> Self {
        match self {
            Self::Classic(policy) => Self::Classic(policy.clone_reset_box()),
            Self::Contextual(policy) => Self::Contextual(policy.clone_reset_box()),
        }
    }
}

/// Opaque native policy object used by the public Python wrappers.
#[pyclass(module = "pymab._pymab", name = "_NativePolicy")]
pub(crate) struct NativePolicy {
    kind: String,
    configuration: Value,
    handle: PolicyHandle,
}

pub(crate) enum RuntimePolicyHandle {
    Classic(Box<dyn pymab::policy::runtime::RuntimePolicy>),
    Contextual(Box<dyn pymab::policy::runtime::RuntimeContextualPolicy>),
}

impl NativePolicy {
    pub(crate) fn clone_runtime(&self) -> RuntimePolicyHandle {
        match &self.handle {
            PolicyHandle::Classic(policy) => {
                RuntimePolicyHandle::Classic(policy.clone_runtime_box())
            }
            PolicyHandle::Contextual(policy) => {
                RuntimePolicyHandle::Contextual(policy.clone_runtime_box())
            }
        }
    }
}

#[pymethods]
impl NativePolicy {
    /// Construct one registered policy from its strict JSON configuration.
    #[staticmethod]
    fn create(kind: &str, configuration_json: &str) -> PyResult<Self> {
        let configuration: Value = serde_json::from_str(configuration_json).map_err(|error| {
            PyValueError::new_err(format!("invalid policy configuration: {error}"))
        })?;
        let object = configuration
            .as_object()
            .ok_or_else(|| PyValueError::new_err("policy configuration must be a JSON object"))?;
        let handle = create_handle(kind, object)?;
        Ok(Self {
            kind: kind.to_owned(),
            configuration,
            handle,
        })
    }

    /// Return the stable registered policy kind.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// Return whether this policy consumes contextual observations.
    #[getter]
    fn is_contextual(&self) -> bool {
        matches!(self.handle, PolicyHandle::Contextual(_))
    }

    /// Return the canonical immutable constructor configuration as JSON.
    fn configuration_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.configuration)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Return a copy of learned state as JSON.
    fn state_json(&self) -> PyResult<String> {
        let snapshot = match &self.handle {
            PolicyHandle::Classic(policy) => policy.snapshot(),
            PolicyHandle::Contextual(policy) => policy.snapshot(),
        };
        serde_json::to_string(&snapshot).map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// Select an action from a one-shot deterministic native random stream.
    #[pyo3(signature = (seed, context=None))]
    fn select_action(&mut self, seed: u64, context: Option<Vec<f64>>) -> PyResult<usize> {
        let key = StreamKey::new(seed, 0, StreamRole::PolicySelection)
            .with_policy_id(self.kind.clone())
            .map_err(to_python)?;
        let mut rng = rng_for(&key).map_err(to_python)?;
        let action = match (&mut self.handle, context) {
            (PolicyHandle::Classic(policy), None) => policy.select_action(&mut rng),
            (PolicyHandle::Contextual(policy), Some(context)) => {
                policy.select_action(&context, &mut rng)
            }
            (PolicyHandle::Classic(_), Some(_)) => {
                return Err(PyTypeError::new_err(
                    "classic policy does not accept context",
                ));
            }
            (PolicyHandle::Contextual(_), None) => {
                return Err(PyTypeError::new_err("contextual policy requires context"));
            }
        }
        .map_err(to_python)?;
        Ok(action.get())
    }

    /// Update learned state from one observation.
    #[pyo3(signature = (action, reward, context=None))]
    fn update(&mut self, action: usize, reward: f64, context: Option<Vec<f64>>) -> PyResult<()> {
        match (&mut self.handle, context) {
            (PolicyHandle::Classic(policy), None) => {
                let action = ActionIndex::new(action, policy.n_arms()).map_err(to_python)?;
                policy.update(action, reward).map_err(to_python)
            }
            (PolicyHandle::Contextual(policy), Some(context)) => {
                let action = ActionIndex::new(action, policy.n_arms()).map_err(to_python)?;
                policy.update(action, reward, &context).map_err(to_python)
            }
            (PolicyHandle::Classic(_), Some(_)) => Err(PyTypeError::new_err(
                "classic policy does not accept context",
            )),
            (PolicyHandle::Contextual(_), None) => {
                Err(PyTypeError::new_err("contextual policy requires context"))
            }
        }
    }

    /// Return a non-exploratory recommendation.
    #[pyo3(signature = (context=None))]
    fn recommend_action(&self, context: Option<Vec<f64>>) -> PyResult<usize> {
        let action = match (&self.handle, context) {
            (PolicyHandle::Classic(policy), None) => policy.recommend_action(),
            (PolicyHandle::Contextual(policy), Some(context)) => policy.recommend_action(&context),
            (PolicyHandle::Classic(_), Some(_)) => {
                return Err(PyTypeError::new_err(
                    "classic policy does not accept context",
                ));
            }
            (PolicyHandle::Contextual(_), None) => {
                return Err(PyTypeError::new_err("contextual policy requires context"));
            }
        }
        .map_err(to_python)?;
        Ok(action.get())
    }

    /// Reset learned state without reallocating policy buffers.
    fn reset(&mut self) {
        match &mut self.handle {
            PolicyHandle::Classic(policy) => policy.reset(),
            PolicyHandle::Contextual(policy) => policy.reset(),
        }
    }

    /// Clone immutable configuration into a fresh reset handle.
    fn clone_reset(&self) -> Self {
        Self {
            kind: self.kind.clone(),
            configuration: self.configuration.clone(),
            handle: self.handle.clone_reset(),
        }
    }

    /// Estimate Rust-owned state memory, including reserved vector capacity.
    fn estimated_state_bytes(&self) -> usize {
        match &self.handle {
            PolicyHandle::Classic(policy) => policy.estimated_state_bytes(),
            PolicyHandle::Contextual(policy) => policy.estimated_state_bytes(),
        }
    }
}

fn create_handle(kind: &str, config: &Map<String, Value>) -> PyResult<PolicyHandle> {
    macro_rules! classic {
        ($value:expr) => {
            PolicyHandle::Classic(Box::new($value.map_err(to_python)?))
        };
    }
    macro_rules! contextual {
        ($value:expr) => {
            PolicyHandle::Contextual(Box::new($value.map_err(to_python)?))
        };
    }

    let handle = match kind {
        "random" => {
            exact_fields(config, &["n_arms"])?;
            classic!(RandomPolicy::new(usize_field(config, "n_arms")?))
        }
        "greedy" => {
            exact_fields(config, &["n_arms", "initial_value"])?;
            classic!(GreedyPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
            ))
        }
        "epsilon_greedy" => {
            exact_fields(config, &["n_arms", "initial_value", "epsilon"])?;
            classic!(EpsilonGreedyPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "epsilon")?,
            ))
        }
        "decaying_epsilon_greedy" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "initial_epsilon",
                    "min_epsilon",
                    "decay_rate",
                ],
            )?;
            classic!(DecayingEpsilonGreedyPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "initial_epsilon")?,
                number_field(config, "min_epsilon")?,
                number_field(config, "decay_rate")?,
            ))
        }
        "softmax" => {
            exact_fields(config, &["n_arms", "initial_value", "temperature"])?;
            classic!(SoftmaxPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "temperature")?,
            ))
        }
        "ucb" => {
            exact_fields(config, &["n_arms", "initial_value", "c", "reward_scale"])?;
            classic!(UCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
            ))
        }
        "kl_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "c",
                    "tolerance",
                    "max_iterations",
                ],
            )?;
            classic!(KLUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "tolerance")?,
                usize_field(config, "max_iterations")?,
            ))
        }
        "moss" => {
            exact_fields(
                config,
                &["n_arms", "initial_value", "horizon", "c", "reward_scale"],
            )?;
            classic!(MOSSPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                u64_field(config, "horizon")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
            ))
        }
        "gradient_bandit" => {
            exact_fields(config, &["n_arms", "learning_rate", "use_baseline"])?;
            classic!(GradientBanditPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "learning_rate")?,
                bool_field(config, "use_baseline")?,
            ))
        }
        "bernoulli_thompson_sampling" => {
            exact_fields(config, &["n_arms", "alpha_prior", "beta_prior"])?;
            classic!(BernoulliThompsonSamplingPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "alpha_prior")?,
                number_field(config, "beta_prior")?,
            ))
        }
        "gaussian_thompson_sampling" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "prior_mean",
                    "prior_precision",
                    "reward_precision",
                ],
            )?;
            classic!(GaussianThompsonSamplingPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "prior_mean")?,
                number_field(config, "prior_precision")?,
                number_field(config, "reward_precision")?,
            ))
        }
        "bernoulli_bayesian_ucb" => {
            exact_fields(config, &["n_arms", "alpha_prior", "beta_prior", "quantile"])?;
            classic!(BernoulliBayesianUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "alpha_prior")?,
                number_field(config, "beta_prior")?,
                number_field(config, "quantile")?,
            ))
        }
        "gaussian_bayesian_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "prior_mean",
                    "prior_precision",
                    "reward_precision",
                    "quantile",
                ],
            )?;
            classic!(GaussianBayesianUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "prior_mean")?,
                number_field(config, "prior_precision")?,
                number_field(config, "reward_precision")?,
                number_field(config, "quantile")?,
            ))
        }
        "exp3" => {
            exact_fields(config, &["n_arms", "gamma", "learning_rate"])?;
            classic!(EXP3Policy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "gamma")?,
                Some(number_field(config, "learning_rate")?),
            ))
        }
        "successive_elimination" => {
            exact_fields(config, &["n_arms", "delta", "confidence_scale"])?;
            classic!(SuccessiveEliminationPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "delta")?,
                number_field(config, "confidence_scale")?,
            ))
        }
        "median_elimination" => {
            exact_fields(config, &["n_arms", "epsilon", "delta"])?;
            classic!(MedianEliminationPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "epsilon")?,
                number_field(config, "delta")?,
            ))
        }
        "sliding_window_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "c",
                    "reward_scale",
                    "window_size",
                ],
            )?;
            classic!(SlidingWindowUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
                usize_field(config, "window_size")?,
            ))
        }
        "discounted_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "c",
                    "reward_scale",
                    "discount_factor",
                ],
            )?;
            classic!(DiscountedUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
                number_field(config, "discount_factor")?,
            ))
        }
        "sliding_window_bernoulli_thompson_sampling" => {
            exact_fields(
                config,
                &["n_arms", "alpha_prior", "beta_prior", "window_size"],
            )?;
            classic!(SlidingWindowBernoulliThompsonSamplingPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "alpha_prior")?,
                number_field(config, "beta_prior")?,
                usize_field(config, "window_size")?,
            ))
        }
        "discounted_bernoulli_thompson_sampling" => {
            exact_fields(
                config,
                &["n_arms", "alpha_prior", "beta_prior", "discount_factor"],
            )?;
            classic!(DiscountedBernoulliThompsonSamplingPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "alpha_prior")?,
                number_field(config, "beta_prior")?,
                number_field(config, "discount_factor")?,
            ))
        }
        "change_point_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "c",
                    "reward_scale",
                    "detector",
                    "threshold",
                    "drift",
                    "min_observations",
                ],
            )?;
            let detector = match string_field(config, "detector")? {
                "cusum" => ChangeDetector::Cusum,
                "page_hinkley" => ChangeDetector::PageHinkley,
                _ => {
                    return Err(PyValueError::new_err(
                        "detector must be 'cusum' or 'page_hinkley'",
                    ))
                }
            };
            classic!(ChangePointUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
                detector,
                number_field(config, "threshold")?,
                number_field(config, "drift")?,
                u64_field(config, "min_observations")?,
            ))
        }
        "cusum_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "c",
                    "reward_scale",
                    "threshold",
                    "drift",
                    "min_observations",
                ],
            )?;
            classic!(CUSUMUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
                number_field(config, "threshold")?,
                number_field(config, "drift")?,
                u64_field(config, "min_observations")?,
            ))
        }
        "page_hinkley_ucb" => {
            exact_fields(
                config,
                &[
                    "n_arms",
                    "initial_value",
                    "c",
                    "reward_scale",
                    "threshold",
                    "drift",
                    "min_observations",
                ],
            )?;
            classic!(PageHinkleyUCBPolicy::new(
                usize_field(config, "n_arms")?,
                number_field(config, "initial_value")?,
                number_field(config, "c")?,
                number_field(config, "reward_scale")?,
                number_field(config, "threshold")?,
                number_field(config, "drift")?,
                u64_field(config, "min_observations")?,
            ))
        }
        "linear_epsilon_greedy" => {
            exact_fields(
                config,
                &["n_arms", "n_features", "epsilon", "learning_rate"],
            )?;
            contextual!(LinearEpsilonGreedyPolicy::new(
                usize_field(config, "n_arms")?,
                usize_field(config, "n_features")?,
                number_field(config, "epsilon")?,
                number_field(config, "learning_rate")?,
            ))
        }
        "lin_ucb" => {
            exact_fields(config, &["n_arms", "n_features", "alpha", "l2"])?;
            contextual!(LinUCBPolicy::new(
                usize_field(config, "n_arms")?,
                usize_field(config, "n_features")?,
                number_field(config, "alpha")?,
                number_field(config, "l2")?,
            ))
        }
        "linear_thompson_sampling" => {
            exact_fields(config, &["n_arms", "n_features", "exploration_scale", "l2"])?;
            contextual!(LinearThompsonSamplingPolicy::new(
                usize_field(config, "n_arms")?,
                usize_field(config, "n_features")?,
                number_field(config, "exploration_scale")?,
                number_field(config, "l2")?,
            ))
        }
        "logistic_contextual_bandit" => {
            exact_fields(
                config,
                &["n_arms", "n_features", "epsilon", "learning_rate", "l2"],
            )?;
            contextual!(LogisticContextualBanditPolicy::new(
                usize_field(config, "n_arms")?,
                usize_field(config, "n_features")?,
                number_field(config, "epsilon")?,
                number_field(config, "learning_rate")?,
                number_field(config, "l2")?,
            ))
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown policy kind: {kind}"
            )))
        }
    };
    Ok(handle)
}

fn exact_fields(config: &Map<String, Value>, expected: &[&str]) -> PyResult<()> {
    let expected: BTreeSet<_> = expected.iter().copied().collect();
    let actual: BTreeSet<_> = config.keys().map(String::as_str).collect();
    if actual == expected {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "policy configuration fields differ: missing={:?}, unknown={:?}",
            expected.difference(&actual).collect::<Vec<_>>(),
            actual.difference(&expected).collect::<Vec<_>>()
        )))
    }
}

fn field<'a>(config: &'a Map<String, Value>, name: &str) -> PyResult<&'a Value> {
    config
        .get(name)
        .ok_or_else(|| PyValueError::new_err(format!("missing configuration field: {name}")))
}

fn number_field(config: &Map<String, Value>, name: &str) -> PyResult<f64> {
    field(config, name)?
        .as_f64()
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be numeric")))
}

fn u64_field(config: &Map<String, Value>, name: &str) -> PyResult<u64> {
    field(config, name)?
        .as_u64()
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be a non-negative integer")))
}

fn usize_field(config: &Map<String, Value>, name: &str) -> PyResult<usize> {
    usize::try_from(u64_field(config, name)?)
        .map_err(|_| PyValueError::new_err(format!("{name} is too large")))
}

fn bool_field(config: &Map<String, Value>, name: &str) -> PyResult<bool> {
    field(config, name)?
        .as_bool()
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be boolean")))
}

fn string_field<'a>(config: &'a Map<String, Value>, name: &str) -> PyResult<&'a str> {
    field(config, name)?
        .as_str()
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be a string")))
}
