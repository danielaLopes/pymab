use pymab::policy::change_detection::{
    CUSUMUCBPolicy, ChangeDetector, ChangePointUCBPolicy, PageHinkleyUCBPolicy,
};
use pymab::policy::nonstationary::{
    DiscountedBernoulliThompsonSamplingPolicy, DiscountedUCBPolicy,
    SlidingWindowBernoulliThompsonSamplingPolicy, SlidingWindowUCBPolicy,
};
use pymab::policy::Policy;
use pymab::types::ActionIndex;

#[test]
fn nonstationary_constructors_reject_invalid_configuration() {
    assert!(SlidingWindowUCBPolicy::new(2, 0.0, 2.0, 1.0, 0).is_err());
    assert!(DiscountedUCBPolicy::new(2, 0.0, 2.0, 1.0, 1.0).is_err());
    assert!(SlidingWindowBernoulliThompsonSamplingPolicy::new(2, 1.0, 1.0, 0).is_err());
    assert!(DiscountedBernoulliThompsonSamplingPolicy::new(2, 1.0, 1.0, 0.0).is_err());
    assert!(
        ChangePointUCBPolicy::new(2, 0.0, 2.0, 1.0, ChangeDetector::Cusum, 0.0, 0.0, 2,).is_err()
    );
}

#[test]
fn nonstationary_sliding_windows_expire_by_global_time_and_bound_capacity() {
    let mut ucb = SlidingWindowUCBPolicy::new(1, 0.0, 2.0, 1.0, 2).unwrap();
    let action = ActionIndex::new(0, 1).unwrap();
    for reward in [1.0, 3.0, 5.0] {
        ucb.update(action, reward).unwrap();
    }
    assert_eq!(ucb.state().step(), 3);
    assert_eq!(ucb.state().history_len(), 2);
    assert_eq!(ucb.state().estimates(), &[4.0]);

    let mut thompson = SlidingWindowBernoulliThompsonSamplingPolicy::new(1, 1.0, 1.0, 2).unwrap();
    for reward in [1.0, 0.0, 0.0] {
        thompson.update(action, reward).unwrap();
    }
    assert_eq!(thompson.state().history_len(), 2);
    assert_eq!(thompson.state().successes(), &[0]);
    assert_eq!(thompson.state().failures(), &[2]);
}

#[test]
fn nonstationary_discounting_updates_effective_counts_and_sums() {
    let action = ActionIndex::new(0, 1).unwrap();
    let mut ucb = DiscountedUCBPolicy::new(1, 0.0, 2.0, 1.0, 0.5).unwrap();
    ucb.update(action, 1.0).unwrap();
    ucb.update(action, 1.0).unwrap();
    assert_eq!(ucb.state().counts(), &[2]);
    assert_eq!(ucb.state().discounted_counts(), &[1.5]);
    assert_eq!(ucb.state().discounted_sums(), &[1.5]);

    let mut thompson = DiscountedBernoulliThompsonSamplingPolicy::new(1, 1.0, 1.0, 0.5).unwrap();
    thompson.update(action, 1.0).unwrap();
    thompson.update(action, 0.0).unwrap();
    assert_eq!(thompson.state().counts(), &[1.5]);
    assert_eq!(thompson.state().successes(), &[0.5]);
    assert_eq!(thompson.state().failures(), &[1.0]);
    assert!((thompson.state().estimates()[0] - 1.0 / 3.0).abs() < 1e-15);
}

#[test]
fn change_detection_resets_only_the_changed_arm() {
    let action = ActionIndex::new(0, 2).unwrap();
    let mut policy = CUSUMUCBPolicy::new(2, 0.0, 2.0, 1.0, 0.1, 0.0, 2).unwrap();
    for reward in [0.0, 0.0, 5.0] {
        policy.update(action, reward).unwrap();
    }
    assert_eq!(policy.state().change_counts(), &[1, 0]);
    assert_eq!(policy.state().action_values().counts(), &[1, 0]);
    assert_eq!(policy.state().action_values().estimates(), &[5.0, 0.0]);
}

#[test]
fn change_detection_state_remains_finite_on_long_traces() {
    let action = ActionIndex::new(0, 1).unwrap();
    let mut policy = PageHinkleyUCBPolicy::new(1, 0.0, 2.0, 1.0, 0.5, 0.01, 3).unwrap();
    for step in 0..1_000 {
        policy.update(action, f64::from(step % 2)).unwrap();
    }
    assert!(policy.state().all_finite());
}

#[test]
fn sliding_window_rejects_arm_sum_overflow_without_mutating_state() {
    let mut policy = SlidingWindowUCBPolicy::new(2, 0.0, 2.0, 1.0, 3).unwrap();
    let first = ActionIndex::new(0, 2).unwrap();
    let second = ActionIndex::new(1, 2).unwrap();
    policy.update(first, f64::MAX).unwrap();
    policy.update(second, -f64::MAX).unwrap();

    assert!(policy.update(first, f64::MAX).is_err());
    assert_eq!(policy.state().step(), 2);
    assert_eq!(policy.state().history_len(), 2);
    assert!(policy
        .state()
        .estimates()
        .iter()
        .all(|value| value.is_finite()));
}
