use pymab::rng::{derive_seed, rng_for, StreamKey, StreamRole};
use rand::RngCore;

#[test]
fn stream_derivation_has_a_stable_golden_value() {
    let key = StreamKey::new(42, 7, StreamRole::PolicySelection)
        .with_policy_id("epsilon-greedy")
        .expect("valid policy stream");

    assert_eq!(
        derive_seed(&key).expect("valid stream"),
        [
            181, 158, 40, 179, 210, 243, 237, 15, 116, 150, 46, 106, 14, 187, 174, 242, 224, 102,
            145, 198, 174, 226, 60, 225, 4, 176, 25, 146, 219, 115, 234, 44,
        ]
    );
}

#[test]
fn distinct_stream_roles_are_isolated() {
    let dynamics = StreamKey::new(19, 2, StreamRole::EnvironmentDynamics);
    let contexts = StreamKey::new(19, 2, StreamRole::ContextGeneration);
    let rewards = StreamKey::new(19, 2, StreamRole::CommonRewards);

    assert_ne!(
        derive_seed(&dynamics).unwrap(),
        derive_seed(&contexts).unwrap()
    );
    assert_ne!(
        derive_seed(&contexts).unwrap(),
        derive_seed(&rewards).unwrap()
    );
    assert_ne!(
        derive_seed(&dynamics).unwrap(),
        derive_seed(&rewards).unwrap()
    );
}

#[test]
fn policy_ids_are_isolated_and_order_independent() {
    let alpha = StreamKey::new(91, 0, StreamRole::PolicySelection)
        .with_policy_id("alpha")
        .expect("valid policy stream");
    let beta = StreamKey::new(91, 0, StreamRole::PolicySelection)
        .with_policy_id("beta")
        .expect("valid policy stream");

    let alpha_seed = derive_seed(&alpha).unwrap();
    let beta_seed = derive_seed(&beta).unwrap();
    assert_ne!(alpha_seed, beta_seed);

    let reordered = [beta, alpha];
    assert_eq!(derive_seed(&reordered[1]).unwrap(), alpha_seed);
    assert_eq!(derive_seed(&reordered[0]).unwrap(), beta_seed);
}

#[test]
fn seeded_generators_replay_the_same_sequence() {
    let key = StreamKey::new(5, 3, StreamRole::PolicyIndependentRewards)
        .with_policy_id("ucb")
        .expect("valid policy stream");
    let mut left = rng_for(&key).expect("valid stream");
    let mut right = rng_for(&key).expect("valid stream");

    let left_values = [left.next_u64(), left.next_u64(), left.next_u64()];
    let right_values = [right.next_u64(), right.next_u64(), right.next_u64()];
    assert_eq!(left_values, right_values);
}

#[test]
fn policy_stream_roles_require_a_non_empty_policy_id() {
    let missing = StreamKey::new(1, 0, StreamRole::PolicySelection);
    assert!(rng_for(&missing).is_err());

    let empty = StreamKey::new(1, 0, StreamRole::PolicySelection).with_policy_id(" ");
    assert!(empty.is_err());
}
