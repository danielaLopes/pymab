use std::mem::size_of;

use pymab::policy::action_value::ActionValueState;
use pymab::policy::contextual::LinUCBPolicy;
use pymab::policy::nonstationary::SlidingWindowUCBPolicy;
use pymab::policy::{ContextualPolicy, Policy};
use pymab::types::ActionIndex;

#[test]
fn action_value_estimate_includes_owned_buffers_and_stays_stable() {
    let mut state = ActionValueState::new(8, 0.0).unwrap();
    let initial = state.estimated_state_bytes();
    assert!(initial >= size_of::<ActionValueState>() + 8 * size_of::<u64>() + 8 * size_of::<f64>());
    for step in 0..128 {
        state
            .update(ActionIndex::new(step % 8, 8).unwrap(), step as f64)
            .unwrap();
    }
    assert_eq!(state.estimated_state_bytes(), initial);
}

#[test]
fn sliding_window_estimate_accounts_for_reserved_history() {
    let mut policy = SlidingWindowUCBPolicy::new(4, 0.0, 2.0, 1.0, 32).unwrap();
    let reserved = Policy::estimated_state_bytes(&policy);
    for step in 0..96 {
        Policy::update(
            &mut policy,
            ActionIndex::new(step % 4, 4).unwrap(),
            (step % 5) as f64,
        )
        .unwrap();
    }
    assert_eq!(Policy::estimated_state_bytes(&policy), reserved);
    assert_eq!(policy.state().history_len(), 32);
}

#[test]
fn contextual_matrix_estimate_scales_with_shape() {
    let small = LinUCBPolicy::new(3, 2, 1.0, 1.0).unwrap();
    let large = LinUCBPolicy::new(6, 8, 1.0, 1.0).unwrap();
    assert!(
        ContextualPolicy::estimated_state_bytes(&large)
            > ContextualPolicy::estimated_state_bytes(&small)
    );
}
