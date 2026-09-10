//! Python boundary for complete native experiments and NumPy result buffers.

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use pymab::experiment::{
    Experiment, ExperimentConfig, NamedContextualPolicy, NamedPolicy, RewardCoupling,
};

use crate::environment::NativeEnvironment;
use crate::error::to_python;
use crate::policy::{NativePolicy, RuntimePolicyHandle};

/// NumPy-owned result buffers returned by one native experiment.
#[pyclass(module = "pymab._pymab", name = "_NativeExperimentResult")]
pub(crate) struct NativeExperimentResult {
    rewards: Py<PyAny>,
    actions: Py<PyAny>,
    expected_rewards: Py<PyAny>,
    arm_means: Py<PyAny>,
    optimal_mask: Py<PyAny>,
    recommendations: Py<PyAny>,
    contexts: Option<Py<PyAny>>,
    context_digest: Option<String>,
    policy_state_bytes: Vec<usize>,
}

#[pymethods]
impl NativeExperimentResult {
    #[getter]
    fn rewards(&self, py: Python<'_>) -> Py<PyAny> {
        self.rewards.clone_ref(py)
    }

    #[getter]
    fn actions(&self, py: Python<'_>) -> Py<PyAny> {
        self.actions.clone_ref(py)
    }

    #[getter]
    fn expected_rewards(&self, py: Python<'_>) -> Py<PyAny> {
        self.expected_rewards.clone_ref(py)
    }

    #[getter]
    fn arm_means(&self, py: Python<'_>) -> Py<PyAny> {
        self.arm_means.clone_ref(py)
    }

    #[getter]
    fn optimal_mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.optimal_mask.clone_ref(py)
    }

    #[getter]
    fn recommendations(&self, py: Python<'_>) -> Py<PyAny> {
        self.recommendations.clone_ref(py)
    }

    #[getter]
    fn contexts(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.contexts.as_ref().map(|value| value.clone_ref(py))
    }

    #[getter]
    fn context_digest(&self) -> Option<&str> {
        self.context_digest.as_deref()
    }

    #[getter]
    fn policy_state_bytes(&self) -> Vec<usize> {
        self.policy_state_bytes.clone()
    }
}

/// Stateless entry point for running a complete native experiment.
#[pyclass(module = "pymab._pymab", name = "_NativeExperiment")]
pub(crate) struct NativeExperiment;

#[pymethods]
impl NativeExperiment {
    /// Execute a native-compatible environment and policy collection.
    #[staticmethod]
    #[pyo3(signature = (
        environment,
        policies,
        horizon,
        n_replicates,
        seed,
        reward_coupling,
        record_contexts
    ))]
    #[allow(clippy::too_many_arguments)]
    fn run(
        py: Python<'_>,
        environment: PyRef<'_, NativeEnvironment>,
        policies: Vec<(String, Py<NativePolicy>)>,
        horizon: usize,
        n_replicates: usize,
        seed: u64,
        reward_coupling: &str,
        record_contexts: bool,
    ) -> PyResult<NativeExperimentResult> {
        let coupling = match reward_coupling {
            "common" => RewardCoupling::Common,
            "independent" => RewardCoupling::Independent,
            _ => {
                return Err(PyValueError::new_err(
                    "reward_coupling must be 'common' or 'independent'",
                ))
            }
        };
        let config = ExperimentConfig {
            horizon,
            n_replicates,
            seed,
            reward_coupling: coupling,
            record_contexts,
        };
        let environment_value = environment.runtime_clone();
        let experiment = if environment_value.contextual() {
            let values = policies
                .into_iter()
                .map(|(id, policy)| match policy.borrow(py).clone_runtime() {
                    RuntimePolicyHandle::Contextual(value) => {
                        NamedContextualPolicy::new(id, value).map_err(to_python)
                    }
                    RuntimePolicyHandle::Classic(_) => Err(PyTypeError::new_err(
                        "classic policy cannot run in a contextual environment",
                    )),
                })
                .collect::<PyResult<Vec<_>>>()?;
            Experiment::contextual(environment_value, values, config)
        } else {
            let values = policies
                .into_iter()
                .map(|(id, policy)| match policy.borrow(py).clone_runtime() {
                    RuntimePolicyHandle::Classic(value) => {
                        NamedPolicy::new(id, value).map_err(to_python)
                    }
                    RuntimePolicyHandle::Contextual(_) => Err(PyTypeError::new_err(
                        "contextual policy cannot run in a classic environment",
                    )),
                })
                .collect::<PyResult<Vec<_>>>()?;
            Experiment::classic(environment_value, values, config)
        }
        .map_err(to_python)?;

        let result = py.detach(move || experiment.run()).map_err(to_python)?;
        let shape = result.shape;
        let policy_shape = [shape.n_replicates, shape.horizon, shape.n_policies];
        let environment_shape = [shape.n_replicates, shape.horizon, shape.n_arms];
        let actions = result
            .actions
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| PyOverflowError::new_err("action exceeds int64"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let recommendations = result
            .recommendations
            .into_iter()
            .map(|value| {
                i64::try_from(value)
                    .map_err(|_| PyOverflowError::new_err("recommendation exceeds int64"))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let rewards = result
            .rewards
            .into_pyarray(py)
            .reshape(policy_shape)?
            .into_any()
            .unbind();
        let actions = actions
            .into_pyarray(py)
            .reshape(policy_shape)?
            .into_any()
            .unbind();
        let expected_rewards = result
            .expected_rewards
            .into_pyarray(py)
            .reshape(policy_shape)?
            .into_any()
            .unbind();
        let arm_means = result
            .arm_means
            .into_pyarray(py)
            .reshape(environment_shape)?
            .into_any()
            .unbind();
        let optimal_mask = result
            .optimal_mask
            .into_pyarray(py)
            .reshape(environment_shape)?
            .into_any()
            .unbind();
        let recommendations = recommendations
            .into_pyarray(py)
            .reshape(policy_shape)?
            .into_any()
            .unbind();
        let contexts = match (result.contexts, shape.n_features) {
            (Some(values), Some(n_features)) => Some(
                values
                    .into_pyarray(py)
                    .reshape([shape.n_replicates, shape.horizon, shape.n_arms, n_features])?
                    .into_any()
                    .unbind(),
            ),
            (None, None) | (None, Some(_)) => None,
            (Some(_), None) => {
                return Err(PyValueError::new_err(
                    "native result contains contexts without a feature dimension",
                ))
            }
        };

        Ok(NativeExperimentResult {
            rewards,
            actions,
            expected_rewards,
            arm_means,
            optimal_mask,
            recommendations,
            contexts,
            context_digest: result.context_digest,
            policy_state_bytes: result.policy_state_bytes,
        })
    }
}
