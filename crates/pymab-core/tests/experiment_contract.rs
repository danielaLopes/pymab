use pymab::distribution::{BuiltInRewardModel, GaussianReward};
use pymab::environment::contextual::{
    BuiltInContextProvider, FixedContextProvider, LinearContextualEnvironment,
};
use pymab::environment::dynamics::{BuiltInDynamics, StationaryDynamics};
use pymab::environment::{BanditEnvironment, BuiltInEnvironment};
use pymab::experiment::{
    Experiment, ExperimentConfig, NamedContextualPolicy, NamedPolicy, RewardCoupling,
};
use pymab::policy::basic::{GreedyPolicy, RandomPolicy};
use pymab::policy::contextual::LinUCBPolicy;

fn config(record_contexts: bool) -> ExperimentConfig {
    ExperimentConfig {
        horizon: 8,
        n_replicates: 3,
        seed: 19,
        reward_coupling: RewardCoupling::Common,
        record_contexts,
    }
}

fn classic_environment() -> BuiltInEnvironment {
    BuiltInEnvironment::Classic(
        BanditEnvironment::new(
            vec![0.0, 1.0, 2.0],
            BuiltInRewardModel::Gaussian(GaussianReward::new(0.5).expect("reward")),
            BuiltInDynamics::Stationary(StationaryDynamics),
        )
        .expect("environment"),
    )
}

fn policy(id: &str) -> NamedPolicy {
    NamedPolicy::new(id, Box::new(RandomPolicy::new(3).expect("policy"))).expect("named")
}

#[test]
fn classic_experiment_preallocates_expected_shapes_and_replays() {
    let experiment = Experiment::classic(
        classic_environment(),
        vec![
            policy("random"),
            NamedPolicy::new(
                "greedy",
                Box::new(GreedyPolicy::new(3, 0.0).expect("policy")),
            )
            .expect("named"),
        ],
        config(false),
    )
    .expect("experiment");
    let left = experiment.run().expect("run");
    let right = experiment.run().expect("replay");
    assert_eq!(left, right);
    assert_eq!(left.shape.n_replicates, 3);
    assert_eq!(left.rewards.len(), 3 * 8 * 2);
    assert_eq!(left.arm_means.len(), 3 * 8 * 3);
    assert_eq!(left.optimal_mask.len(), 3 * 8 * 3);
    assert!(left.contexts.is_none());
    assert!(left.context_digest.is_none());
    assert!(left.actions.iter().all(|action| *action < 3));
    assert!(left.recommendations.iter().all(|action| *action < 3));
}

#[test]
fn policy_streams_are_isolated_from_input_order() {
    let forward = Experiment::classic(
        classic_environment(),
        vec![policy("a"), policy("b")],
        config(false),
    )
    .expect("experiment")
    .run()
    .expect("run");
    let reverse = Experiment::classic(
        classic_environment(),
        vec![policy("b"), policy("a")],
        config(false),
    )
    .expect("experiment")
    .run()
    .expect("run");

    for replicate in 0..forward.shape.n_replicates {
        for step in 0..forward.shape.horizon {
            let forward_a = (replicate * forward.shape.horizon + step) * 2;
            let reverse_a = (replicate * reverse.shape.horizon + step) * 2 + 1;
            assert_eq!(forward.actions[forward_a], reverse.actions[reverse_a]);
            assert_eq!(forward.rewards[forward_a], reverse.rewards[reverse_a]);
        }
    }
}

#[test]
fn contextual_experiment_records_contexts_and_stable_digest() {
    let provider = FixedContextProvider::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("provider");
    let environment = BuiltInEnvironment::Linear(
        LinearContextualEnvironment::new(
            2,
            2,
            vec![1.0, 0.0, 0.0, 2.0],
            BuiltInContextProvider::Fixed(provider),
            BuiltInRewardModel::Gaussian(GaussianReward::new(0.1).expect("reward")),
        )
        .expect("environment"),
    );
    let experiment = Experiment::contextual(
        environment,
        vec![NamedContextualPolicy::new(
            "linucb",
            Box::new(LinUCBPolicy::new(2, 2, 1.0, 1.0).expect("policy")),
        )
        .expect("named")],
        config(true),
    )
    .expect("experiment");
    let left = experiment.run().expect("run");
    let right = experiment.run().expect("replay");
    assert_eq!(left.context_digest, right.context_digest);
    assert_eq!(left.context_digest.as_deref().map(str::len), Some(64));
    assert_eq!(left.contexts.as_ref().map(Vec::len), Some(3 * 8 * 2 * 2));
}

#[test]
fn experiment_rejects_empty_duplicate_and_incompatible_policies() {
    assert!(Experiment::classic(classic_environment(), Vec::new(), config(false)).is_err());
    assert!(Experiment::classic(
        classic_environment(),
        vec![policy("same"), policy("same")],
        config(false),
    )
    .is_err());
    assert!(Experiment::classic(
        classic_environment(),
        vec![NamedPolicy::new(
            "wrong-arms",
            Box::new(RandomPolicy::new(2).expect("policy")),
        )
        .expect("named")],
        config(false),
    )
    .is_err());
}
