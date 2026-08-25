//! Built-in classic and contextual bandit environments.

pub mod classic;
pub mod contextual;
pub mod dynamics;

pub use classic::BanditEnvironment;
pub use contextual::ContextProvider;
pub use dynamics::EnvironmentDynamics;
