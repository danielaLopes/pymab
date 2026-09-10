use pymab::policy::bayesian_ucb::{BernoulliBayesianUCBPolicy, GaussianBayesianUCBPolicy};
use pymab::policy::gradient::GradientBanditPolicy;
use pymab::policy::thompson::{BernoulliThompsonSamplingPolicy, GaussianThompsonSamplingPolicy};
use pymab::policy::Policy;
use pymab::rng::{rng_for, StreamKey, StreamRole};
use pymab::types::ActionIndex;

fn policy_rng(policy_id: &str) -> pymab::rng::NativeRng {
    let key = StreamKey::new(303, 0, StreamRole::PolicySelection)
        .with_policy_id(policy_id)
        .unwrap();
    rng_for(&key).unwrap()
}

#[test]
fn gradient_updates_preferences_and_running_baseline() {
    let mut policy = GradientBanditPolicy::new(2, 0.1, true).unwrap();
    let mut rng = policy_rng("gradient");
    policy.select_action(&mut rng).unwrap();
    policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();

    assert_eq!(policy.state().step(), 1);
    assert_eq!(policy.state().average_reward(), 1.0);
    assert_eq!(policy.state().preferences(), &[0.05, -0.05]);
    assert_eq!(policy.recommend_action().unwrap().get(), 0);
}

#[test]
fn thompson_bernoulli_posteriors_track_successes_failures_and_binary_validation() {
    let mut policy = BernoulliThompsonSamplingPolicy::new(2, 1.0, 1.0).unwrap();
    assert!(policy.update(ActionIndex::new(0, 2).unwrap(), 0.5).is_err());
    policy.update(ActionIndex::new(0, 2).unwrap(), 1.0).unwrap();
    policy.update(ActionIndex::new(1, 2).unwrap(), 0.0).unwrap();
    assert_eq!(policy.state().successes(), &[1, 0]);
    assert_eq!(policy.state().failures(), &[0, 1]);

    let mut left_rng = policy_rng("bernoulli-ts");
    let mut right_rng = policy_rng("bernoulli-ts");
    assert_eq!(
        policy.select_action(&mut left_rng).unwrap(),
        policy.select_action(&mut right_rng).unwrap()
    );
}

#[test]
fn thompson_gaussian_posterior_updates_mean_and_precision_exactly() {
    let mut policy = GaussianThompsonSamplingPolicy::new(1, 0.0, 1.0, 1.0).unwrap();
    policy.update(ActionIndex::new(0, 1).unwrap(), 2.0).unwrap();
    assert_eq!(policy.state().means(), &[1.0]);
    assert_eq!(policy.state().precisions(), &[2.0]);
    assert_eq!(policy.state().action_values().estimates(), &[1.0]);
}

#[test]
fn bayesian_ucb_quantiles_match_high_precision_values() {
    let mut bernoulli = BernoulliBayesianUCBPolicy::new(2, 1.0, 1.0, 0.9).unwrap();
    bernoulli
        .update(ActionIndex::new(0, 2).unwrap(), 1.0)
        .unwrap();
    bernoulli
        .update(ActionIndex::new(1, 2).unwrap(), 0.0)
        .unwrap();
    let bounds = bernoulli.upper_bounds().unwrap();
    assert!((bounds[0] - 0.9_f64.sqrt()).abs() < 1e-12);
    assert!((bounds[1] - (1.0 - 0.1_f64.sqrt())).abs() < 1e-12);

    let gaussian = GaussianBayesianUCBPolicy::new(1, 0.0, 1.0, 1.0, 0.95).unwrap();
    assert!((gaussian.upper_bounds().unwrap()[0] - 1.644_853_626_951_472_2).abs() < 1e-12);
}

#[test]
fn gradient_thompson_bayesian_constructors_reject_invalid_configuration() {
    assert!(GradientBanditPolicy::new(2, 0.0, true).is_err());
    assert!(BernoulliThompsonSamplingPolicy::new(2, 0.0, 1.0).is_err());
    assert!(GaussianThompsonSamplingPolicy::new(2, 0.0, 0.0, 1.0).is_err());
    assert!(BernoulliBayesianUCBPolicy::new(2, 1.0, 1.0, 1.0).is_err());
    assert!(GaussianBayesianUCBPolicy::new(2, 0.0, 1.0, 1.0, 0.0).is_err());
}
