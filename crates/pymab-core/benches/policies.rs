use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pymab::policy::contextual::LinUCBPolicy;
use pymab::policy::epsilon_greedy::EpsilonGreedyPolicy;
use pymab::policy::ucb::UCBPolicy;
use pymab::policy::{ContextualPolicy, Policy};
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::ActionIndex;

fn selection_rng(policy_id: &str) -> pymab::rng::NativeRng {
    rng_for(
        &StreamKey::new(41, 0, StreamRole::PolicySelection)
            .with_policy_id(policy_id)
            .unwrap(),
    )
    .unwrap()
}

fn benchmark_classic(c: &mut Criterion) {
    let mut group = c.benchmark_group("classic_policy");
    for n_arms in [4, 32] {
        let epsilon = EpsilonGreedyPolicy::new(n_arms, 0.0, 0.1).unwrap();
        let state_bytes = Policy::estimated_state_bytes(&epsilon);
        group.bench_with_input(
            BenchmarkId::new(
                "epsilon_select",
                format!("arms={n_arms},state={state_bytes}"),
            ),
            &n_arms,
            |bench, _| {
                let mut policy = epsilon.clone();
                let mut rng = selection_rng("epsilon");
                bench.iter(|| black_box(Policy::select_action(&mut policy, &mut rng).unwrap()));
            },
        );

        let ucb = UCBPolicy::new(n_arms, 0.0, 2.0, 1.0).unwrap();
        let state_bytes = Policy::estimated_state_bytes(&ucb);
        group.bench_with_input(
            BenchmarkId::new(
                "ucb_select_update",
                format!("arms={n_arms},state={state_bytes}"),
            ),
            &n_arms,
            |bench, _| {
                let mut policy = ucb.clone();
                let mut rng = selection_rng("ucb");
                bench.iter(|| {
                    let action = Policy::select_action(&mut policy, &mut rng).unwrap();
                    Policy::update(&mut policy, action, black_box(0.75)).unwrap();
                    black_box(action)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "epsilon_update",
                format!("arms={n_arms},state={state_bytes}"),
            ),
            &n_arms,
            |bench, _| {
                let mut policy = EpsilonGreedyPolicy::new(n_arms, 0.0, 0.1).unwrap();
                let action = ActionIndex::new(0, n_arms).unwrap();
                bench.iter(|| Policy::update(&mut policy, action, black_box(0.75)).unwrap());
            },
        );
    }
    group.finish();
}

fn benchmark_contextual(c: &mut Criterion) {
    let mut group = c.benchmark_group("contextual_policy");
    for n_features in [4, 16] {
        let policy = LinUCBPolicy::new(5, n_features, 1.0, 1.0).unwrap();
        let state_bytes = ContextualPolicy::estimated_state_bytes(&policy);
        let context = vec![0.25; 5 * n_features];
        group.bench_with_input(
            BenchmarkId::new(
                "linucb_select_update",
                format!("features={n_features},state={state_bytes}"),
            ),
            &n_features,
            |bench, _| {
                let mut policy = policy.clone();
                let mut rng = selection_rng("linucb");
                bench.iter(|| {
                    let action =
                        ContextualPolicy::select_action(&mut policy, black_box(&context), &mut rng)
                            .unwrap();
                    ContextualPolicy::update(&mut policy, action, 0.75, &context).unwrap();
                    black_box(action)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_classic, benchmark_contextual);
criterion_main!(benches);
