#![doc = include_str!("../README.md")]

pub mod distribution;
pub mod environment;
pub mod error;
pub mod experiment;
pub mod memory;
pub mod policy;
pub mod result;
pub mod rng;
pub mod types;
pub mod validation;

/// Version of the Rust core crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Versioned identifier for the native random-stream scheme.
pub const RNG_SCHEME_VERSION: &str = "pymab-rust-blake2b-chacha12-v1";

/// Return the Rust core version.
#[must_use]
pub const fn version() -> &'static str {
    VERSION
}

/// Return the native random-stream scheme identifier.
#[must_use]
pub const fn rng_scheme_version() -> &'static str {
    RNG_SCHEME_VERSION
}

#[cfg(test)]
mod tests {
    use super::{rng_scheme_version, version};

    #[test]
    fn core_metadata_is_non_empty() {
        assert!(!version().is_empty());
        assert!(rng_scheme_version().starts_with("pymab-rust-"));
    }
}
