# PyMAB for Rust

`pymab` is the Python-independent core of PyMAB: typed multi-armed bandit
policies, reward models, environments, deterministic random streams, and a
complete experiment runner. Public operations validate user input and return
`Result` rather than panicking.

## Minimal policy loop

```rust
use pymab::policy::basic::GreedyPolicy;
use pymab::policy::Policy;
use pymab::rng::{rng_for, StreamKey, StreamRole};

let mut policy = GreedyPolicy::new(3, 0.0)?;
let key = StreamKey::new(42, 0, StreamRole::PolicySelection)
    .with_policy_id("greedy")?;
let mut rng = rng_for(&key)?;
let action = policy.select_action(&mut rng)?;
policy.update(action, 1.0)?;
assert!(policy.recommend_action()?.get() < 3);
# Ok::<(), pymab::error::PyMabError>(())
```

## Complete experiment

```rust
use pymab::distribution::{BuiltInRewardModel, GaussianReward};
use pymab::environment::dynamics::{BuiltInDynamics, StationaryDynamics};
use pymab::environment::{BanditEnvironment, BuiltInEnvironment};
use pymab::experiment::{
    Experiment, ExperimentConfig, NamedPolicy, RewardCoupling,
};
use pymab::policy::ucb::UCBPolicy;

let environment = BuiltInEnvironment::Classic(BanditEnvironment::new(
    vec![0.1, 0.5, 0.9],
    BuiltInRewardModel::Gaussian(GaussianReward::new(1.0)?),
    BuiltInDynamics::Stationary(StationaryDynamics),
)?);
let policies = vec![NamedPolicy::new(
    "ucb",
    Box::new(UCBPolicy::new(3, 0.0, 2.0, 1.0)?),
)?];
let experiment = Experiment::classic(
    environment,
    policies,
    ExperimentConfig {
        horizon: 100,
        n_replicates: 20,
        seed: 42,
        reward_coupling: RewardCoupling::Common,
        record_contexts: false,
    },
)?;
let result = experiment.run()?;
assert_eq!(result.rewards.len(), 2_000);
# Ok::<(), pymab::error::PyMabError>(())
```

## Custom policy trait implementation

```rust
use pymab::error::Result;
use pymab::policy::Policy;
use pymab::rng::NativeRng;
use pymab::types::{
    ActionIndex, PolicyCapabilities, PolicyObjective, ALL_REWARD_DOMAINS,
};

#[derive(Clone, Debug, PartialEq)]
struct FirstArmState;

#[derive(Clone)]
struct FirstArmPolicy {
    n_arms: usize,
    state: FirstArmState,
}

impl Policy for FirstArmPolicy {
    type State = FirstArmState;

    fn capabilities(&self) -> PolicyCapabilities {
        PolicyCapabilities::new(false, ALL_REWARD_DOMAINS, PolicyObjective::CumulativeReward)
    }
    fn n_arms(&self) -> usize { self.n_arms }
    fn select_action(&mut self, _rng: &mut NativeRng) -> Result<ActionIndex> {
        ActionIndex::new(0, self.n_arms)
    }
    fn update(&mut self, _action: ActionIndex, _reward: f64) -> Result<()> { Ok(()) }
    fn recommend_action(&self) -> Result<ActionIndex> { ActionIndex::new(0, self.n_arms) }
    fn reset(&mut self) {}
    fn state(&self) -> &Self::State { &self.state }
    fn estimated_state_bytes(&self) -> usize { std::mem::size_of::<Self>() }
}

let policy = FirstArmPolicy { n_arms: 2, state: FirstArmState };
assert_eq!(policy.n_arms(), 2);
```

## Typed errors and reproducible streams

```rust
use pymab::policy::basic::GreedyPolicy;

let error = GreedyPolicy::new(0, 0.0).expect_err("zero arms are invalid");
assert_eq!(error.code(), pymab::error::ErrorCode::Configuration);
assert_eq!(pymab::rng_scheme_version(), "pymab-rust-blake2b-chacha12-v1");
```

The native RNG contract isolates environment, context, common-reward,
per-policy selection, and independent-reward streams. Adding or reordering a
policy therefore does not alter another policy's native trajectory.
