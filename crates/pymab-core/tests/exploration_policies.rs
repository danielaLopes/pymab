use pymab::policy::adversarial::EXP3Policy;
use pymab::policy::pure_exploration::{MedianEliminationPolicy, SuccessiveEliminationPolicy};
use pymab::policy::Policy;
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::ActionIndex;

fn policy_rng(policy_id: &str) -> pymab::rng::NativeRng {
    rng_for(
        &StreamKey::new(404, 0, StreamRole::PolicySelection)
            .with_policy_id(policy_id)
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn adversarial_and_exploration_constructors_validate_boundaries() {
    assert!(EXP3Policy::new(2, 0.0, None).is_err());
    assert!(EXP3Policy::new(2, 0.1, Some(1.1)).is_err());
    assert!(SuccessiveEliminationPolicy::new(2, 0.0, 1.0).is_err());
    assert!(MedianEliminationPolicy::new(2, 0.0, 0.1).is_err());
}

#[test]
fn adversarial_exp3_stays_finite_with_a_positive_exploration_floor() {
    let mut policy = EXP3Policy::new(3, 1e-6, Some(1.0)).unwrap();
    let mut rng = policy_rng("exp3");
    for _ in 0..20_000 {
        let action = policy.select_action(&mut rng).unwrap();
        policy.update(action, 1.0).unwrap();
    }
    let probabilities = policy.action_probabilities().unwrap();
    assert!(policy
        .state()
        .log_weights()
        .iter()
        .all(|value| value.is_finite()));
    assert!(probabilities
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!(policy.update(ActionIndex::new(0, 3).unwrap(), 1.1).is_err());
}

#[test]
fn pure_exploration_successive_elimination_removes_confidently_worse_arm() {
    let mut policy = SuccessiveEliminationPolicy::new(2, 0.5, 1e-6).unwrap();
    policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();
    policy.update(ActionIndex::new(1, 2).unwrap(), 0.0).unwrap();
    assert_eq!(policy.state().active(), &[true, false]);
    assert_eq!(policy.recommend_action().unwrap().get(), 0);
    let mut rng = policy_rng("successive");
    assert_eq!(policy.select_action(&mut rng).unwrap().get(), 0);
}

#[test]
fn pure_exploration_median_elimination_completes_phase_and_terminates() {
    let mut policy = MedianEliminationPolicy::new(2, 1.0, 0.5).unwrap();
    let quota = policy.phase_quota();
    for _ in 0..quota {
        policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();
        policy.update(ActionIndex::new(1, 2).unwrap(), 0.0).unwrap();
    }
    assert_eq!(policy.state().active(), &[true, false]);
    assert_eq!(policy.state().phase_counts(), &[0, 0]);
    assert_eq!(policy.recommend_action().unwrap().get(), 0);
    assert_eq!(policy.phase_quota(), 362);

    policy.reset();
    assert_eq!(policy.state().active(), &[true, true]);
}
