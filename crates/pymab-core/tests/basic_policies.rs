use pymab::policy::basic::{GreedyPolicy, RandomPolicy};
use pymab::policy::epsilon_greedy::{DecayingEpsilonGreedyPolicy, EpsilonGreedyPolicy};
use pymab::policy::softmax::SoftmaxPolicy;
use pymab::policy::Policy;
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::ActionIndex;
use std::collections::BTreeSet;

fn policy_rng(policy_id: &str) -> pymab::rng::NativeRng {
    let key = StreamKey::new(101, 0, StreamRole::PolicySelection)
        .with_policy_id(policy_id)
        .expect("valid policy stream");
    rng_for(&key).expect("valid generator")
}

#[test]
fn basic_constructors_reject_invalid_configuration() {
    assert!(RandomPolicy::new(0).is_err());
    assert!(GreedyPolicy::new(2, f64::NAN).is_err());
    assert!(EpsilonGreedyPolicy::new(2, 0.0, -0.1).is_err());
    assert!(EpsilonGreedyPolicy::new(2, 0.0, f64::INFINITY).is_err());
    assert!(DecayingEpsilonGreedyPolicy::new(2, 0.0, 0.1, 0.2, 0.01).is_err());
    assert!(DecayingEpsilonGreedyPolicy::new(2, 0.0, 1.0, 0.1, -0.01).is_err());
    assert!(DecayingEpsilonGreedyPolicy::new(2, 0.0, 1.0, 0.1, f64::NAN).is_err());
    assert!(SoftmaxPolicy::new(2, 0.0, 0.0).is_err());
    assert!(SoftmaxPolicy::new(2, 0.0, f64::INFINITY).is_err());
}

#[test]
fn basic_greedy_updates_incremental_means_and_clones_reset_state() {
    let mut policy = GreedyPolicy::new(3, 0.5).expect("valid policy");
    policy.update(ActionIndex::new(1, 3).unwrap(), 1.0).unwrap();
    policy.update(ActionIndex::new(1, 3).unwrap(), 0.0).unwrap();

    assert_eq!(policy.state().step(), 2);
    assert_eq!(policy.state().counts(), &[0, 2, 0]);
    assert_eq!(policy.state().estimates(), &[0.5, 0.5, 0.5]);
    assert_eq!(policy.recommend_action().unwrap().get(), 0);
    assert!(policy.estimated_state_bytes() > 0);

    let fresh = policy.clone_reset();
    assert_eq!(fresh.state().step(), 0);
    assert_eq!(fresh.state().counts(), &[0, 0, 0]);
    assert_eq!(fresh.state().estimates(), &[0.5, 0.5, 0.5]);
}

#[test]
fn basic_random_and_epsilon_sampling_are_seed_reproducible_and_valid() {
    let mut random = RandomPolicy::new(3).unwrap();
    let mut left_rng = policy_rng("random");
    let mut right_rng = policy_rng("random");
    let left: Vec<_> = (0..32)
        .map(|_| random.select_action(&mut left_rng).unwrap().get())
        .collect();
    let right: Vec<_> = (0..32)
        .map(|_| random.select_action(&mut right_rng).unwrap().get())
        .collect();
    assert_eq!(left, right);
    assert!(left.iter().all(|action| *action < 3));
    assert_eq!(
        left.iter().copied().collect::<BTreeSet<_>>(),
        [0, 1, 2].into()
    );

    let mut explore = EpsilonGreedyPolicy::new(3, 0.0, 1.0).unwrap();
    let mut rng = policy_rng("epsilon");
    let mut explored = BTreeSet::new();
    for _ in 0..32 {
        explored.insert(explore.select_action(&mut rng).unwrap().get());
    }
    assert_eq!(explored, [0, 1, 2].into());
}

#[test]
fn basic_decaying_epsilon_follows_the_reference_schedule() {
    let mut policy = DecayingEpsilonGreedyPolicy::new(2, 0.0, 1.0, 0.1, 0.5).unwrap();
    assert_eq!(policy.epsilon(), 1.0);

    policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();
    assert!((policy.epsilon() - 0.7).abs() < 1e-15);

    for _ in 0..100 {
        policy.update(ActionIndex::new(0, 2).unwrap(), 0.0).unwrap();
    }
    assert!(policy.epsilon() >= 0.1);
}

#[test]
fn basic_softmax_probabilities_are_stable_and_sampling_has_valid_support() {
    let mut policy = SoftmaxPolicy::new(2, 10_000.0, 0.5).unwrap();
    policy
        .update(ActionIndex::new(1, 2).unwrap(), 10_001.0)
        .unwrap();
    let probabilities = policy.action_probabilities().unwrap();
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    assert!(probabilities[1] > probabilities[0]);

    let mut rng = policy_rng("softmax");
    for _ in 0..32 {
        assert!(policy.select_action(&mut rng).unwrap().get() < 2);
    }
}
