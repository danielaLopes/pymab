//! Opaque Python handle for built-in native environments.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

use pymab::distribution::{BernoulliReward, BuiltInRewardModel, GaussianReward, UniformReward};
use pymab::environment::contextual::{
    BuiltInContextProvider, FixedContextProvider, GaussianContextProvider,
    LinearContextualEnvironment, LogisticContextualEnvironment,
};
use pymab::environment::dynamics::{
    AbruptShift, BuiltInDynamics, GradualDrift, ProbabilityDrift, RandomArmSwap, StationaryDynamics,
};
use pymab::environment::BanditEnvironment;
use pymab::environment::BuiltInEnvironment;
use pymab::rng::{rng_for, StreamKey, StreamRole};

use crate::error::to_python;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum RewardConfig {
    Gaussian { std: f64 },
    Bernoulli,
    Uniform { half_width: f64 },
}

impl RewardConfig {
    fn build(self) -> pymab::error::Result<BuiltInRewardModel> {
        match self {
            Self::Gaussian { std } => GaussianReward::new(std).map(BuiltInRewardModel::Gaussian),
            Self::Bernoulli => Ok(BuiltInRewardModel::Bernoulli(BernoulliReward::new())),
            Self::Uniform { half_width } => {
                UniformReward::new(half_width).map(BuiltInRewardModel::Uniform)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum DynamicsConfig {
    Stationary,
    Gradual {
        std: f64,
    },
    Abrupt {
        frequency: u64,
        std: f64,
        shift_at_step_zero: bool,
    },
    Probability {
        logit_std: f64,
        epsilon: f64,
    },
    RandomSwap {
        probability: f64,
    },
}

impl DynamicsConfig {
    fn build(self) -> pymab::error::Result<BuiltInDynamics> {
        match self {
            Self::Stationary => Ok(BuiltInDynamics::Stationary(StationaryDynamics)),
            Self::Gradual { std } => GradualDrift::new(std).map(BuiltInDynamics::Gradual),
            Self::Abrupt {
                frequency,
                std,
                shift_at_step_zero,
            } => AbruptShift::new(frequency, std, shift_at_step_zero).map(BuiltInDynamics::Abrupt),
            Self::Probability { logit_std, epsilon } => {
                ProbabilityDrift::new(logit_std, epsilon).map(BuiltInDynamics::Probability)
            }
            Self::RandomSwap { probability } => {
                RandomArmSwap::new(probability).map(BuiltInDynamics::RandomSwap)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ContextProviderConfig {
    Fixed { values: Vec<f64> },
    Gaussian { mean: f64, std: f64 },
}

impl ContextProviderConfig {
    fn build(
        self,
        n_arms: usize,
        n_features: usize,
    ) -> pymab::error::Result<BuiltInContextProvider> {
        match self {
            Self::Fixed { values } => FixedContextProvider::new(n_arms, n_features, values)
                .map(BuiltInContextProvider::Fixed),
            Self::Gaussian { mean, std } => {
                GaussianContextProvider::new(n_arms, n_features, mean, std)
                    .map(BuiltInContextProvider::Gaussian)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum EnvironmentConfig {
    Classic {
        means: Vec<f64>,
        reward: RewardConfig,
        dynamics: DynamicsConfig,
    },
    Linear {
        n_arms: usize,
        n_features: usize,
        theta: Vec<f64>,
        context_provider: ContextProviderConfig,
        reward: RewardConfig,
    },
    Logistic {
        n_arms: usize,
        n_features: usize,
        theta: Vec<f64>,
        context_provider: ContextProviderConfig,
        reward: RewardConfig,
    },
}

#[derive(Clone, Debug)]
enum EnvironmentHandle {
    Classic(BanditEnvironment),
    Linear(LinearContextualEnvironment<BuiltInContextProvider>),
    Logistic(LogisticContextualEnvironment<BuiltInContextProvider>),
}

impl EnvironmentHandle {
    fn create(config: EnvironmentConfig) -> pymab::error::Result<Self> {
        match config {
            EnvironmentConfig::Classic {
                means,
                reward,
                dynamics,
            } => {
                BanditEnvironment::new(means, reward.build()?, dynamics.build()?).map(Self::Classic)
            }
            EnvironmentConfig::Linear {
                n_arms,
                n_features,
                theta,
                context_provider,
                reward,
            } => LinearContextualEnvironment::new(
                n_arms,
                n_features,
                theta,
                context_provider.build(n_arms, n_features)?,
                reward.build()?,
            )
            .map(Self::Linear),
            EnvironmentConfig::Logistic {
                n_arms,
                n_features,
                theta,
                context_provider,
                reward,
            } => LogisticContextualEnvironment::new(
                n_arms,
                n_features,
                theta,
                context_provider.build(n_arms, n_features)?,
                reward.build()?,
            )
            .map(Self::Logistic),
        }
    }

    fn contextual(&self) -> bool {
        !matches!(self, Self::Classic(_))
    }

    fn n_arms(&self) -> usize {
        match self {
            Self::Classic(value) => value.n_arms(),
            Self::Linear(value) => value.shape().n_arms(),
            Self::Logistic(value) => value.shape().n_arms(),
        }
    }

    fn n_features(&self) -> Option<usize> {
        match self {
            Self::Classic(_) => None,
            Self::Linear(value) => Some(value.shape().n_features()),
            Self::Logistic(value) => Some(value.shape().n_features()),
        }
    }

    fn context(&self, seed: u64) -> PyResult<Vec<f64>> {
        let mut rng =
            rng_for(&StreamKey::new(seed, 0, StreamRole::ContextGeneration)).map_err(to_python)?;
        match self {
            Self::Classic(_) => Err(PyTypeError::new_err(
                "classic environment does not produce context",
            )),
            Self::Linear(value) => value.context(&mut rng).map_err(to_python),
            Self::Logistic(value) => value.context(&mut rng).map_err(to_python),
        }
    }

    fn expected_rewards(&self, context: Option<&[f64]>) -> PyResult<Vec<f64>> {
        match (self, context) {
            (Self::Classic(value), None) => Ok(value.expected_rewards().to_vec()),
            (Self::Classic(_), Some(_)) => Err(PyTypeError::new_err(
                "classic environment does not accept context",
            )),
            (Self::Linear(_), None) | (Self::Logistic(_), None) => Err(PyTypeError::new_err(
                "contextual environment requires context",
            )),
            (Self::Linear(value), Some(context)) => {
                value.expected_rewards(context).map_err(to_python)
            }
            (Self::Logistic(value), Some(context)) => {
                value.expected_rewards(context).map_err(to_python)
            }
        }
    }

    fn sample_rewards(&self, seed: u64, context: Option<&[f64]>) -> PyResult<Vec<f64>> {
        let mut rng =
            rng_for(&StreamKey::new(seed, 0, StreamRole::CommonRewards)).map_err(to_python)?;
        match (self, context) {
            (Self::Classic(value), None) => value.sample_rewards(&mut rng).map_err(to_python),
            (Self::Classic(_), Some(_)) => Err(PyTypeError::new_err(
                "classic environment does not accept context",
            )),
            (Self::Linear(_), None) | (Self::Logistic(_), None) => Err(PyTypeError::new_err(
                "contextual environment requires context",
            )),
            (Self::Linear(value), Some(context)) => {
                value.sample_rewards(context, &mut rng).map_err(to_python)
            }
            (Self::Logistic(value), Some(context)) => {
                value.sample_rewards(context, &mut rng).map_err(to_python)
            }
        }
    }
}

/// Opaque native environment used for direct parity tests and experiment setup.
#[pyclass(module = "pymab._pymab", name = "_NativeEnvironment")]
pub(crate) struct NativeEnvironment {
    configuration: Value,
    handle: EnvironmentHandle,
}

impl NativeEnvironment {
    pub(crate) fn runtime_clone(&self) -> BuiltInEnvironment {
        match &self.handle {
            EnvironmentHandle::Classic(value) => BuiltInEnvironment::Classic(value.clone()),
            EnvironmentHandle::Linear(value) => BuiltInEnvironment::Linear(value.clone()),
            EnvironmentHandle::Logistic(value) => BuiltInEnvironment::Logistic(value.clone()),
        }
    }
}

#[pymethods]
impl NativeEnvironment {
    /// Construct a built-in environment from strict tagged JSON configuration.
    #[staticmethod]
    fn create(configuration_json: &str) -> PyResult<Self> {
        let configuration: Value = serde_json::from_str(configuration_json).map_err(|error| {
            PyValueError::new_err(format!("invalid environment configuration: {error}"))
        })?;
        let typed: EnvironmentConfig =
            serde_json::from_value(configuration.clone()).map_err(|error| {
                PyValueError::new_err(format!("invalid environment configuration: {error}"))
            })?;
        let handle = EnvironmentHandle::create(typed).map_err(to_python)?;
        Ok(Self {
            configuration,
            handle,
        })
    }

    /// Return whether this environment requires contexts.
    #[getter]
    fn contextual(&self) -> bool {
        self.handle.contextual()
    }

    /// Return the number of arms.
    #[getter]
    fn n_arms(&self) -> usize {
        self.handle.n_arms()
    }

    /// Return the number of contextual features, if any.
    #[getter]
    fn n_features(&self) -> Option<usize> {
        self.handle.n_features()
    }

    /// Return canonical configuration JSON.
    fn configuration_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.configuration).map_err(|error| {
            PyValueError::new_err(format!(
                "cannot serialize environment configuration: {error}"
            ))
        })
    }

    /// Return serializable current environment state.
    fn state_json(&self) -> PyResult<String> {
        let state = match &self.handle {
            EnvironmentHandle::Classic(value) => json!({"means": value.means()}),
            EnvironmentHandle::Linear(value) => json!({"theta": value.theta()}),
            EnvironmentHandle::Logistic(value) => json!({"theta": value.theta()}),
        };
        serde_json::to_string(&state).map_err(|error| {
            PyValueError::new_err(format!("cannot serialize environment state: {error}"))
        })
    }

    /// Generate one row-major context matrix from a direct-call seed.
    fn context(&self, seed: u64) -> PyResult<Vec<f64>> {
        self.handle.context(seed)
    }

    /// Compute one expected reward per arm.
    #[pyo3(signature = (context=None))]
    fn expected_rewards(&self, context: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.handle.expected_rewards(context.as_deref())
    }

    /// Sample one potential reward per arm from a direct-call seed.
    #[pyo3(signature = (seed, context=None))]
    fn sample_rewards(&self, seed: u64, context: Option<Vec<f64>>) -> PyResult<Vec<f64>> {
        self.handle.sample_rewards(seed, context.as_deref())
    }

    /// Advance classic dynamics by one step.
    fn advance(&mut self, step: u64, seed: u64) -> PyResult<()> {
        let mut rng = rng_for(&StreamKey::new(seed, 0, StreamRole::EnvironmentDynamics))
            .map_err(to_python)?;
        match &mut self.handle {
            EnvironmentHandle::Classic(value) => value.advance(step, &mut rng).map_err(to_python),
            EnvironmentHandle::Linear(_) | EnvironmentHandle::Logistic(_) => Err(
                PyTypeError::new_err("contextual environment has no dynamics to advance"),
            ),
        }
    }

    /// Clone the complete current environment state.
    fn clone_handle(&self) -> Self {
        Self {
            configuration: self.configuration.clone(),
            handle: self.handle.clone(),
        }
    }
}
