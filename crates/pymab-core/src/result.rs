//! Contiguous result buffers produced by native experiments.

/// Axis metadata for flat row-major result buffers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultShape {
    /// Independent replicate count.
    pub n_replicates: usize,
    /// Decisions per replicate.
    pub horizon: usize,
    /// Number of compared policies.
    pub n_policies: usize,
    /// Number of environment arms.
    pub n_arms: usize,
    /// Context feature count when contexts are recorded.
    pub n_features: Option<usize>,
}

/// Preallocated, contiguous output of a native experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentResult {
    /// Result dimensions.
    pub shape: ResultShape,
    /// Observed rewards, shaped `(replicate, step, policy)`.
    pub rewards: Vec<f64>,
    /// Selected actions, shaped `(replicate, step, policy)`.
    pub actions: Vec<usize>,
    /// Expected reward of each selected action.
    pub expected_rewards: Vec<f64>,
    /// True arm means, shaped `(replicate, step, arm)`.
    pub arm_means: Vec<f64>,
    /// Tie-aware optimal-arm mask, shaped `(replicate, step, arm)`.
    pub optimal_mask: Vec<bool>,
    /// Per-step recommendations, shaped `(replicate, step, policy)`.
    pub recommendations: Vec<usize>,
    /// Optional contexts, shaped `(replicate, step, arm, feature)`.
    pub contexts: Option<Vec<f64>>,
    /// Native context-stream digest, present for contextual experiments.
    pub context_digest: Option<String>,
    /// Capacity-aware state bytes for each policy prototype.
    pub policy_state_bytes: Vec<usize>,
}

impl ExperimentResult {
    pub(crate) fn allocate(shape: ResultShape, contextual: bool) -> Self {
        let policy_elements = shape.n_replicates * shape.horizon * shape.n_policies;
        let environment_elements = shape.n_replicates * shape.horizon * shape.n_arms;
        let contexts = shape
            .n_features
            .filter(|_| contextual)
            .map(|n_features| vec![0.0; environment_elements * n_features]);
        Self {
            shape,
            rewards: vec![0.0; policy_elements],
            actions: vec![0; policy_elements],
            expected_rewards: vec![0.0; policy_elements],
            arm_means: vec![0.0; environment_elements],
            optimal_mask: vec![false; environment_elements],
            recommendations: vec![0; policy_elements],
            contexts,
            context_digest: None,
            policy_state_bytes: Vec::with_capacity(shape.n_policies),
        }
    }

    pub(crate) const fn policy_index(&self, replicate: usize, step: usize, policy: usize) -> usize {
        (replicate * self.shape.horizon + step) * self.shape.n_policies + policy
    }

    pub(crate) const fn environment_index(
        &self,
        replicate: usize,
        step: usize,
        arm: usize,
    ) -> usize {
        (replicate * self.shape.horizon + step) * self.shape.n_arms + arm
    }
}
