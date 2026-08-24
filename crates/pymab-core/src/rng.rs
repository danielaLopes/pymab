//! Versioned deterministic random-stream derivation.

use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::error::{PyMabError, Result};

const DERIVATION_DOMAIN: &[u8] = b"pymab-rust-stream-v1\0";

/// Seeded generator used by the native backend.
pub type NativeRng = ChaCha12Rng;

/// Logical roles whose streams must remain independent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
#[non_exhaustive]
pub enum StreamRole {
    /// Evolution of environment dynamics.
    EnvironmentDynamics = 1,
    /// Generation of contextual observations.
    ContextGeneration = 2,
    /// Common potential rewards shared by policies.
    CommonRewards = 3,
    /// Per-policy action selection.
    PolicySelection = 4,
    /// Per-policy rewards under independent coupling.
    PolicyIndependentRewards = 5,
}

impl StreamRole {
    const fn requires_policy_id(self) -> bool {
        matches!(self, Self::PolicySelection | Self::PolicyIndependentRewards)
    }
}

/// Complete identity of one deterministic native random stream.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamKey {
    master_seed: u64,
    replicate: u64,
    role: StreamRole,
    policy_id: Option<String>,
}

impl StreamKey {
    /// Construct a stream key without a policy identifier.
    #[must_use]
    pub const fn new(master_seed: u64, replicate: u64, role: StreamRole) -> Self {
        Self {
            master_seed,
            replicate,
            role,
            policy_id: None,
        }
    }

    /// Attach a stable, non-empty policy identifier.
    pub fn with_policy_id(mut self, policy_id: impl Into<String>) -> Result<Self> {
        let policy_id = policy_id.into();
        if policy_id.trim().is_empty() {
            return Err(PyMabError::configuration(
                "policy_id",
                "must be a non-empty string",
            ));
        }
        self.policy_id = Some(policy_id);
        self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<()> {
        match (self.role.requires_policy_id(), self.policy_id.as_ref()) {
            (true, None) => Err(PyMabError::configuration(
                "policy_id",
                "is required for a policy-specific random stream",
            )),
            (false, Some(_)) => Err(PyMabError::configuration(
                "policy_id",
                "is not allowed for a shared random stream",
            )),
            _ => Ok(()),
        }
    }
}

/// Derive a stable 256-bit seed from a labeled stream key.
pub fn derive_seed(key: &StreamKey) -> Result<[u8; 32]> {
    key.validate()?;
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(DERIVATION_DOMAIN);
    hasher.update(key.master_seed.to_le_bytes());
    hasher.update(key.replicate.to_le_bytes());
    hasher.update([key.role as u8]);
    match key.policy_id.as_deref() {
        Some(policy_id) => {
            hasher.update([1]);
            hasher.update((policy_id.len() as u64).to_le_bytes());
            hasher.update(policy_id.as_bytes());
        }
        None => hasher.update([0]),
    }
    Ok(hasher.finalize().into())
}

/// Construct the deterministic ChaCha12 generator for a labeled stream.
pub fn rng_for(key: &StreamKey) -> Result<NativeRng> {
    Ok(NativeRng::from_seed(derive_seed(key)?))
}
