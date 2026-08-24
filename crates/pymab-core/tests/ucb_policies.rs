use pymab::policy::ucb::{bernoulli_kl, KLUCBPolicy, MOSSPolicy, UCBPolicy};
use pymab::policy::Policy;
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::ActionIndex;

fn policy_rng(policy_id: &str) -> pymab::rng::NativeRng {
    let key = StreamKey::new(202, 0, StreamRole::PolicySelection)
        .with_policy_id(policy_id)
        .unwrap();
    rng_for(&key).unwrap()
}

#[test]
fn ucb_constructors_reject_invalid_configuration() {
    assert!(UCBPolicy::new(2, 0.0, 0.0, 1.0).is_err());
    assert!(UCBPolicy::new(2, 0.0, 2.0, f64::NAN).is_err());
    assert!(KLUCBPolicy::new(2, 0.0, 3.0, 0.0, 32).is_err());
    assert!(KLUCBPolicy::new(2, 0.0, 3.0, 1e-6, 0).is_err());
    assert!(MOSSPolicy::new(3, 0.0, 2, 1.0, 1.0).is_err());
}

#[test]
fn ucb_selects_unseen_arms_in_index_order() {
    let mut policy = UCBPolicy::new(3, 0.0, 2.0, 1.0).unwrap();
    let mut rng = policy_rng("ucb-order");

    for expected in 0..3 {
        assert_eq!(policy.select_action(&mut rng).unwrap().get(), expected);
        policy
            .update(ActionIndex::new(expected, 3).unwrap(), 0.0)
            .unwrap();
    }
}

#[test]
fn ucb_reward_scale_controls_confidence_width() {
    let mut narrow = UCBPolicy::new(1, 0.0, 2.0, 1.0).unwrap();
    let mut wide = UCBPolicy::new(1, 0.0, 2.0, 2.0).unwrap();
    let action = ActionIndex::new(0, 1).unwrap();
    narrow.update(action, 0.0).unwrap();
    wide.update(action, 0.0).unwrap();

    assert_eq!(
        wide.confidence_bonus().unwrap()[0],
        2.0 * narrow.confidence_bonus().unwrap()[0]
    );
    assert!((narrow.confidence_bonus().unwrap()[0] - (2.0 * 2.0_f64.ln()).sqrt()).abs() < 1e-15);
}

#[test]
fn kl_ucb_enforces_binary_rewards_and_converges_at_boundaries() {
    let mut policy = KLUCBPolicy::new(2, 0.0, 3.0, 1e-9, 64).unwrap();
    assert!(policy.update(ActionIndex::new(0, 2).unwrap(), 0.5).is_err());
    policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();
    policy.update(ActionIndex::new(1, 2).unwrap(), 0.0).unwrap();

    let indices = policy.indices().unwrap();
    assert_eq!(indices[0], 1.0);
    assert!((indices[1] - 0.622_917_265_631_258_5).abs() < 1e-12);
    assert!((bernoulli_kl(0.0, 0.5).unwrap() - std::f64::consts::LN_2).abs() < 1e-12);
    assert!((bernoulli_kl(1.0, 0.5).unwrap() - std::f64::consts::LN_2).abs() < 1e-12);
}

#[test]
fn ucb_moss_clips_negative_log_terms_to_zero() {
    let mut policy = MOSSPolicy::new(2, 0.0, 2, 1.0, 1.0).unwrap();
    policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();
    policy.update(ActionIndex::new(1, 2).unwrap(), 0.0).unwrap();

    assert_eq!(policy.confidence_bonus().unwrap(), vec![0.0, 0.0]);

    let mut wide_horizon = MOSSPolicy::new(2, 0.0, 100, 1.0, 2.0).unwrap();
    wide_horizon
        .update(ActionIndex::new(0, 2).unwrap(), 1.0)
        .unwrap();
    wide_horizon
        .update(ActionIndex::new(1, 2).unwrap(), 0.0)
        .unwrap();
    assert!((wide_horizon.confidence_bonus().unwrap()[0] - 3.955_766_932_177_954).abs() < 1e-12);
}
