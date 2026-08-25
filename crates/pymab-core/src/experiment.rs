//! Deterministic native experiment runner.

use std::collections::BTreeSet;

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

use crate::environment::BuiltInEnvironment;
use crate::error::{PyMabError, Result};
use crate::policy::runtime::{RuntimeContextualPolicy, RuntimePolicy};
use crate::result::{ExperimentResult, ResultShape};
use crate::rng::{rng_for, NativeRng, StreamKey, StreamRole};

const TIE_RTOL: f64 = 1e-12;
const TIE_ATOL: f64 = 1e-12;
const CONTEXT_DIGEST_DOMAIN: &[u8] = b"pymab-native-context-v1\0";

/// Relationship between potential reward samples observed by compared policies.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RewardCoupling {
    /// Every policy sees the same potential reward vector at a step.
    Common,
    /// Every policy receives an independently sampled potential reward vector.
    Independent,
}

/// Scalar native experiment configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExperimentConfig {
    /// Number of decisions in each replicate.
    pub horizon: usize,
    /// Number of independent replicates.
    pub n_replicates: usize,
    /// Master seed for labeled streams.
    pub seed: u64,
    /// Reward coupling mode.
    pub reward_coupling: RewardCoupling,
    /// Whether complete contexts are retained in the result.
    pub record_contexts: bool,
}

impl ExperimentConfig {
    fn validate(self) -> Result<Self> {
        if self.horizon == 0 {
            return Err(PyMabError::configuration(
                "horizon",
                "must be greater than zero",
            ));
        }
        if self.n_replicates == 0 {
            return Err(PyMabError::configuration(
                "n_replicates",
                "must be greater than zero",
            ));
        }
        Ok(self)
    }
}

/// Named classic policy prototype.
pub struct NamedPolicy {
    id: String,
    policy: Box<dyn RuntimePolicy>,
}

impl NamedPolicy {
    /// Construct a named policy prototype.
    pub fn new(id: impl Into<String>, policy: Box<dyn RuntimePolicy>) -> Result<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(PyMabError::configuration(
                "policy_id",
                "must be a non-empty string",
            ));
        }
        Ok(Self { id, policy })
    }

    /// Return the stable policy identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl Clone for NamedPolicy {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            policy: self.policy.clone_reset_box(),
        }
    }
}

/// Named contextual policy prototype.
pub struct NamedContextualPolicy {
    id: String,
    policy: Box<dyn RuntimeContextualPolicy>,
}

impl NamedContextualPolicy {
    /// Construct a named contextual policy prototype.
    pub fn new(id: impl Into<String>, policy: Box<dyn RuntimeContextualPolicy>) -> Result<Self> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(PyMabError::configuration(
                "policy_id",
                "must be a non-empty string",
            ));
        }
        Ok(Self { id, policy })
    }

    /// Return the stable policy identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }
}

impl Clone for NamedContextualPolicy {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            policy: self.policy.clone_reset_box(),
        }
    }
}

enum PolicySet {
    Classic(Vec<NamedPolicy>),
    Contextual(Vec<NamedContextualPolicy>),
}

/// Validated native experiment ready for deterministic execution.
pub struct Experiment {
    environment: BuiltInEnvironment,
    policies: PolicySet,
    config: ExperimentConfig,
}

impl Experiment {
    /// Construct an experiment with classic policies.
    pub fn classic(
        environment: BuiltInEnvironment,
        policies: Vec<NamedPolicy>,
        config: ExperimentConfig,
    ) -> Result<Self> {
        let config = config.validate()?;
        validate_ids(policies.iter().map(NamedPolicy::id))?;
        if environment.contextual() {
            return Err(PyMabError::compatibility(
                "environment",
                "classic policies require a classic environment",
            ));
        }
        for value in &policies {
            if value.policy.n_arms() != environment.n_arms() {
                return Err(PyMabError::compatibility(
                    format!("policy {}", value.id),
                    "n_arms does not match the environment",
                ));
            }
            if !value
                .policy
                .capabilities()
                .supports(environment.reward_domain())
            {
                return Err(PyMabError::compatibility(
                    format!("policy {}", value.id),
                    "does not support the environment reward domain",
                ));
            }
        }
        Ok(Self {
            environment,
            policies: PolicySet::Classic(policies),
            config,
        })
    }

    /// Construct an experiment with contextual policies.
    pub fn contextual(
        environment: BuiltInEnvironment,
        policies: Vec<NamedContextualPolicy>,
        config: ExperimentConfig,
    ) -> Result<Self> {
        let config = config.validate()?;
        validate_ids(policies.iter().map(NamedContextualPolicy::id))?;
        let environment_shape = environment.context_shape().ok_or_else(|| {
            PyMabError::compatibility(
                "environment",
                "contextual policies require a contextual environment",
            )
        })?;
        for value in &policies {
            if value.policy.context_shape() != environment_shape {
                return Err(PyMabError::compatibility(
                    format!("policy {}", value.id),
                    "context shape does not match the environment",
                ));
            }
            if !value
                .policy
                .capabilities()
                .supports(environment.reward_domain())
            {
                return Err(PyMabError::compatibility(
                    format!("policy {}", value.id),
                    "does not support the environment reward domain",
                ));
            }
        }
        Ok(Self {
            environment,
            policies: PolicySet::Contextual(policies),
            config,
        })
    }

    /// Execute every replicate into preallocated contiguous buffers.
    pub fn run(&self) -> Result<ExperimentResult> {
        match &self.policies {
            PolicySet::Classic(policies) => self.run_classic(policies),
            PolicySet::Contextual(policies) => self.run_contextual(policies),
        }
    }

    fn result(&self, n_policies: usize, n_features: Option<usize>) -> ExperimentResult {
        ExperimentResult::allocate(
            ResultShape {
                n_replicates: self.config.n_replicates,
                horizon: self.config.horizon,
                n_policies,
                n_arms: self.environment.n_arms(),
                n_features,
            },
            self.environment.contextual(),
        )
    }

    fn run_classic(&self, prototypes: &[NamedPolicy]) -> Result<ExperimentResult> {
        let mut result = self.result(prototypes.len(), None);
        result.policy_state_bytes = prototypes
            .iter()
            .map(|value| value.policy.estimated_state_bytes())
            .collect();
        for replicate in 0..self.config.n_replicates {
            let mut environment = self.environment.clone();
            let mut policies = prototypes.to_vec();
            let mut dynamics_rng =
                shared_rng(self.config.seed, replicate, StreamRole::EnvironmentDynamics)?;
            let mut common_rng =
                shared_rng(self.config.seed, replicate, StreamRole::CommonRewards)?;
            let mut action_rngs = policy_rngs(
                self.config.seed,
                replicate,
                StreamRole::PolicySelection,
                prototypes.iter().map(NamedPolicy::id),
            )?;
            let mut reward_rngs = policy_rngs(
                self.config.seed,
                replicate,
                StreamRole::PolicyIndependentRewards,
                prototypes.iter().map(NamedPolicy::id),
            )?;

            for step in 0..self.config.horizon {
                environment.advance(step as u64, &mut dynamics_rng)?;
                let means = environment.expected_rewards(None)?;
                record_environment(&mut result, replicate, step, &means, None);
                let common_rewards = if self.config.reward_coupling == RewardCoupling::Common {
                    Some(environment.sample_rewards(None, &mut common_rng)?)
                } else {
                    None
                };
                for (policy_index, value) in policies.iter_mut().enumerate() {
                    let action = value.policy.select_action(&mut action_rngs[policy_index])?;
                    let reward = match common_rewards.as_ref() {
                        Some(values) => values[action.get()],
                        None => environment.sample_rewards(None, &mut reward_rngs[policy_index])?
                            [action.get()],
                    };
                    value.policy.update(action, reward)?;
                    let recommendation = value.policy.recommend_action()?;
                    record_policy(
                        &mut result,
                        replicate,
                        step,
                        policy_index,
                        action.get(),
                        reward,
                        means[action.get()],
                        recommendation.get(),
                    );
                }
            }
        }
        Ok(result)
    }

    fn run_contextual(&self, prototypes: &[NamedContextualPolicy]) -> Result<ExperimentResult> {
        let context_shape = self
            .environment
            .context_shape()
            .ok_or_else(|| PyMabError::internal("contextual experiment lost its context shape"))?;
        let mut result = self.result(
            prototypes.len(),
            self.config
                .record_contexts
                .then_some(context_shape.n_features()),
        );
        result.policy_state_bytes = prototypes
            .iter()
            .map(|value| value.policy.estimated_state_bytes())
            .collect();
        let mut context_hasher = Blake2b::<U32>::new();
        context_hasher.update(CONTEXT_DIGEST_DOMAIN);

        for replicate in 0..self.config.n_replicates {
            let environment = self.environment.clone();
            let mut policies = prototypes.to_vec();
            let mut context_rng =
                shared_rng(self.config.seed, replicate, StreamRole::ContextGeneration)?;
            let mut common_rng =
                shared_rng(self.config.seed, replicate, StreamRole::CommonRewards)?;
            let mut action_rngs = policy_rngs(
                self.config.seed,
                replicate,
                StreamRole::PolicySelection,
                prototypes.iter().map(NamedContextualPolicy::id),
            )?;
            let mut reward_rngs = policy_rngs(
                self.config.seed,
                replicate,
                StreamRole::PolicyIndependentRewards,
                prototypes.iter().map(NamedContextualPolicy::id),
            )?;

            for step in 0..self.config.horizon {
                let context = environment.context(&mut context_rng)?;
                let means = environment.expected_rewards(Some(&context))?;
                hash_context(&mut context_hasher, context_shape, &context);
                record_environment(&mut result, replicate, step, &means, Some(&context));
                let common_rewards = if self.config.reward_coupling == RewardCoupling::Common {
                    Some(environment.sample_rewards(Some(&context), &mut common_rng)?)
                } else {
                    None
                };
                for (policy_index, value) in policies.iter_mut().enumerate() {
                    let action = value
                        .policy
                        .select_action(&context, &mut action_rngs[policy_index])?;
                    let reward = match common_rewards.as_ref() {
                        Some(values) => values[action.get()],
                        None => environment
                            .sample_rewards(Some(&context), &mut reward_rngs[policy_index])?
                            [action.get()],
                    };
                    value.policy.update(action, reward, &context)?;
                    let recommendation = value.policy.recommend_action(&context)?;
                    record_policy(
                        &mut result,
                        replicate,
                        step,
                        policy_index,
                        action.get(),
                        reward,
                        means[action.get()],
                        recommendation.get(),
                    );
                }
            }
        }
        result.context_digest = Some(to_hex(context_hasher.finalize().as_slice()));
        Ok(result)
    }
}

fn validate_ids<'a>(ids: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = BTreeSet::new();
    let mut count = 0;
    for id in ids {
        count += 1;
        if !seen.insert(id) {
            return Err(PyMabError::configuration(
                "policy_id",
                format!("duplicate identifier: {id}"),
            ));
        }
    }
    if count == 0 {
        return Err(PyMabError::configuration(
            "policies",
            "at least one policy is required",
        ));
    }
    Ok(())
}

fn shared_rng(master_seed: u64, replicate: usize, role: StreamRole) -> Result<NativeRng> {
    rng_for(&StreamKey::new(master_seed, replicate as u64, role))
}

fn policy_rngs<'a>(
    master_seed: u64,
    replicate: usize,
    role: StreamRole,
    ids: impl Iterator<Item = &'a str>,
) -> Result<Vec<NativeRng>> {
    ids.map(|id| {
        let key =
            StreamKey::new(master_seed, replicate as u64, role).with_policy_id(id.to_owned())?;
        rng_for(&key)
    })
    .collect()
}

fn record_environment(
    result: &mut ExperimentResult,
    replicate: usize,
    step: usize,
    means: &[f64],
    context: Option<&[f64]>,
) {
    let best = means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for (arm, &mean) in means.iter().enumerate() {
        let index = result.environment_index(replicate, step, arm);
        result.arm_means[index] = mean;
        result.optimal_mask[index] = (mean - best).abs() <= TIE_ATOL + TIE_RTOL * best.abs();
    }
    if let (Some(output), Some(values)) = (&mut result.contexts, context) {
        let offset = (replicate * result.shape.horizon + step) * values.len();
        output[offset..offset + values.len()].copy_from_slice(values);
    }
}

#[allow(clippy::too_many_arguments)]
fn record_policy(
    result: &mut ExperimentResult,
    replicate: usize,
    step: usize,
    policy: usize,
    action: usize,
    reward: f64,
    expected_reward: f64,
    recommendation: usize,
) {
    let index = result.policy_index(replicate, step, policy);
    result.actions[index] = action;
    result.rewards[index] = reward;
    result.expected_rewards[index] = expected_reward;
    result.recommendations[index] = recommendation;
}

fn hash_context(hasher: &mut Blake2b<U32>, shape: crate::types::ContextShape, context: &[f64]) {
    hasher.update((shape.n_arms() as u64).to_le_bytes());
    hasher.update((shape.n_features() as u64).to_le_bytes());
    for value in context {
        hasher.update(value.to_bits().to_le_bytes());
    }
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[(byte >> 4) as usize]));
        output.push(char::from(HEX[(byte & 0x0f) as usize]));
    }
    output
}
