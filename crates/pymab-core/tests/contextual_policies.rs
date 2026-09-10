use pymab::policy::contextual::{
    LinUCBPolicy, LinearEpsilonGreedyPolicy, LinearThompsonSamplingPolicy,
    LogisticContextualBanditPolicy,
};
use pymab::policy::ContextualPolicy;
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::{ActionIndex, RewardDomain};

fn selection_rng(policy_id: &str) -> pymab::rng::NativeRng {
    rng_for(
        &StreamKey::new(19, 0, StreamRole::PolicySelection)
            .with_policy_id(policy_id)
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn contextual_constructors_reject_invalid_configuration() {
    assert!(LinearEpsilonGreedyPolicy::new(0, 2, 0.1, 0.1).is_err());
    assert!(LinearEpsilonGreedyPolicy::new(2, 0, 0.1, 0.1).is_err());
    assert!(LinearEpsilonGreedyPolicy::new(2, 2, 1.1, 0.1).is_err());
    assert!(LinUCBPolicy::new(2, 2, 0.0, 1.0).is_err());
    assert!(LinearThompsonSamplingPolicy::new(2, 2, 1.0, 0.0).is_err());
    assert!(LogisticContextualBanditPolicy::new(2, 2, 0.1, 0.1, -0.1).is_err());
}

#[test]
fn linear_epsilon_greedy_updates_only_the_selected_arm() {
    let context = [1.0, 0.0, 0.0, 1.0];
    let mut policy = LinearEpsilonGreedyPolicy::new(2, 2, 0.0, 0.5).unwrap();
    policy
        .update(ActionIndex::new(0, 2).unwrap(), 2.0, &context)
        .unwrap();

    assert_eq!(policy.state().theta(), &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(policy.scores(&context).unwrap(), vec![1.0, 0.0]);
    assert_eq!(policy.recommend_action(&context).unwrap().get(), 0);
    assert!(policy.capabilities().contextual());
    assert!(policy.capabilities().supports(RewardDomain::Real));
}

#[test]
fn lin_ucb_uses_spd_solves_for_means_and_uncertainty() {
    let context = [1.0, 2.0, 0.0, 1.0];
    let mut policy = LinUCBPolicy::new(2, 2, 1.0, 1.0).unwrap();
    policy
        .update(ActionIndex::new(0, 2).unwrap(), 2.0, &context)
        .unwrap();

    assert_eq!(
        policy.state().a(),
        &[2.0, 2.0, 2.0, 5.0, 1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(policy.state().b(), &[2.0, 4.0, 0.0, 0.0]);
    let bounds = policy.upper_confidence_bounds(&context).unwrap();
    let expected = 5.0 / 3.0 + (5.0_f64 / 6.0).sqrt();
    assert!((bounds[0] - expected).abs() < 1e-12);
    assert_eq!(bounds[1], 1.0);
    assert_eq!(policy.recommend_action(&context).unwrap().get(), 0);
}

#[test]
fn linear_thompson_updates_posterior_and_replays_seeded_sampling() {
    let context = [1.0, 2.0, 0.0, 1.0];
    let action = ActionIndex::new(0, 2).unwrap();
    let mut policy = LinearThompsonSamplingPolicy::new(2, 2, 0.5, 1.0).unwrap();
    policy.update(action, 2.0, &context).unwrap();
    assert_eq!(
        policy.posterior_means(&context).unwrap(),
        vec![5.0 / 3.0, 0.0]
    );

    let mut first_policy = policy.clone();
    let mut second_policy = policy;
    let mut first_rng = selection_rng("linear-thompson");
    let mut second_rng = selection_rng("linear-thompson");
    assert_eq!(
        first_policy
            .select_action(&context, &mut first_rng)
            .unwrap(),
        second_policy
            .select_action(&context, &mut second_rng)
            .unwrap()
    );
}

#[test]
fn logistic_policy_uses_stable_binary_gradient_updates() {
    let context = [1.0, 0.0, 0.0, 2.0];
    let mut policy = LogisticContextualBanditPolicy::new(2, 2, 0.0, 0.2, 0.1).unwrap();
    policy
        .update(ActionIndex::new(1, 2).unwrap(), 1.0, &context)
        .unwrap();

    assert_eq!(policy.state().theta(), &[0.0, 0.0, 0.0, 0.2]);
    let probabilities = policy.predicted_probabilities(&context).unwrap();
    assert_eq!(probabilities[0], 0.5);
    assert!((probabilities[1] - 0.598_687_660_112_452).abs() < 1e-15);
    assert!(policy.capabilities().supports(RewardDomain::Binary));
    assert!(!policy.capabilities().supports(RewardDomain::Real));
    assert!(policy
        .update(ActionIndex::new(0, 2).unwrap(), 1.1, &context)
        .is_err());
}

#[test]
fn contextual_inputs_are_checked_without_partial_updates() {
    let mut policy = LinearEpsilonGreedyPolicy::new(2, 2, 0.1, 0.1).unwrap();
    let before = policy.state().clone();
    assert!(policy
        .update(ActionIndex::new(0, 2).unwrap(), 1.0, &[0.0; 3])
        .is_err());
    assert_eq!(policy.state(), &before);
    assert!(policy.scores(&[0.0, 0.0, f64::NAN, 0.0]).is_err());

    let mut posterior = LinUCBPolicy::new(1, 2, 1.0, 1.0).unwrap();
    let before = posterior.state().clone();
    assert!(posterior
        .update(ActionIndex::new(0, 1).unwrap(), 1.0, &[f64::MAX, f64::MAX],)
        .is_err());
    assert_eq!(posterior.state(), &before);
}
