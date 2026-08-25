use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pymab::distribution::{BuiltInRewardModel, GaussianReward};
use pymab::environment::contextual::{
    BuiltInContextProvider, GaussianContextProvider, LinearContextualEnvironment,
};
use pymab::environment::dynamics::BuiltInDynamics;
use pymab::environment::{BanditEnvironment, BuiltInEnvironment};
use pymab::experiment::{
    Experiment, ExperimentConfig, NamedContextualPolicy, NamedPolicy, RewardCoupling,
};
use pymab::policy::contextual::{LinUCBPolicy, LinearEpsilonGreedyPolicy};
use pymab::policy::epsilon_greedy::EpsilonGreedyPolicy;
use pymab::policy::ucb::UCBPolicy;

fn config(horizon: usize) -> ExperimentConfig {
    ExperimentConfig {
        horizon,
        n_replicates: 16,
        seed: 41,
        reward_coupling: RewardCoupling::Common,
        record_contexts: false,
    }
}

fn classic_experiment(horizon: usize) -> Experiment {
    let environment = BanditEnvironment::new(
        vec![0.1, 0.3, 0.5, 0.8],
        BuiltInRewardModel::Gaussian(GaussianReward::new(1.0).unwrap()),
        BuiltInDynamics::default(),
    )
    .unwrap();
    Experiment::classic(
        BuiltInEnvironment::Classic(environment),
        vec![
            NamedPolicy::new(
                "epsilon",
                Box::new(EpsilonGreedyPolicy::new(4, 0.0, 0.1).unwrap()),
            )
            .unwrap(),
            NamedPolicy::new("ucb", Box::new(UCBPolicy::new(4, 0.0, 2.0, 1.0).unwrap())).unwrap(),
        ],
        config(horizon),
    )
    .unwrap()
}

fn contextual_experiment(horizon: usize) -> Experiment {
    let provider =
        BuiltInContextProvider::Gaussian(GaussianContextProvider::new(4, 6, 0.0, 1.0).unwrap());
    let environment = LinearContextualEnvironment::new(
        4,
        6,
        vec![0.15; 24],
        provider,
        BuiltInRewardModel::Gaussian(GaussianReward::new(1.0).unwrap()),
    )
    .unwrap();
    Experiment::contextual(
        BuiltInEnvironment::Linear(environment),
        vec![
            NamedContextualPolicy::new(
                "linucb",
                Box::new(LinUCBPolicy::new(4, 6, 1.0, 1.0).unwrap()),
            )
            .unwrap(),
            NamedContextualPolicy::new(
                "linear_epsilon",
                Box::new(LinearEpsilonGreedyPolicy::new(4, 6, 0.1, 0.05).unwrap()),
            )
            .unwrap(),
        ],
        config(horizon),
    )
    .unwrap()
}

fn benchmark_experiments(c: &mut Criterion) {
    let mut group = c.benchmark_group("experiment");
    for horizon in [100, 1_000] {
        let classic = classic_experiment(horizon);
        group.bench_with_input(
            BenchmarkId::new("classic", horizon),
            &horizon,
            |bench, _| {
                bench.iter(|| black_box(classic.run().unwrap()));
            },
        );

        let contextual = contextual_experiment(horizon);
        group.bench_with_input(
            BenchmarkId::new("contextual", horizon),
            &horizon,
            |bench, _| bench.iter(|| black_box(contextual.run().unwrap())),
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_experiments);
criterion_main!(benches);
