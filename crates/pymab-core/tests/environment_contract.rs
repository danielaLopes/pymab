use pymab::distribution::{
    ArmPrior, BernoulliReward, BetaArmPrior, BuiltInRewardModel, GaussianArmPrior, GaussianReward,
    RewardModel, UniformArmPrior, UniformReward,
};
use pymab::environment::contextual::{
    FixedContextProvider, GaussianContextProvider, LinearContextualEnvironment,
    LogisticContextualEnvironment,
};
use pymab::environment::dynamics::{
    AbruptShift, BuiltInDynamics, GradualDrift, ProbabilityDrift, RandomArmSwap, StationaryDynamics,
};
use pymab::environment::{BanditEnvironment, ContextProvider, EnvironmentDynamics};
use pymab::types::RewardDomain;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

fn seeded(seed: u64) -> ChaCha12Rng {
    ChaCha12Rng::seed_from_u64(seed)
}

#[test]
fn reward_models_validate_support_and_replay_seeded_sampling() {
    let gaussian = GaussianReward::new(0.2).expect("valid Gaussian reward");
    let uniform = UniformReward::new(0.5).expect("valid uniform reward");
    let bernoulli = BernoulliReward::new();

    for model in [
        BuiltInRewardModel::Gaussian(gaussian),
        BuiltInRewardModel::Uniform(uniform),
    ] {
        let means = [-1.0, 2.0];
        assert_eq!(
            model.sample(&means, &mut seeded(3)).expect("sample"),
            model.sample(&means, &mut seeded(3)).expect("replay")
        );
    }
    let means = [0.1, 0.9];
    assert_eq!(
        bernoulli.sample(&means, &mut seeded(4)).expect("sample"),
        bernoulli.sample(&means, &mut seeded(4)).expect("replay")
    );
    assert!(bernoulli.validate_means(&[1.1]).is_err());
    assert!(gaussian.validate_means(&[]).is_err());
    assert!(GaussianReward::new(0.0).is_err());
    assert!(UniformReward::new(-1.0).is_err());
}

#[test]
fn arm_priors_validate_configuration_and_generate_one_mean_per_arm() {
    let gaussian = GaussianArmPrior::new(1.0, 0.0).expect("valid prior");
    let beta = BetaArmPrior::new(2.0, 3.0).expect("valid prior");
    let uniform = UniformArmPrior::new(-1.0, 2.0).expect("valid prior");

    assert_eq!(
        gaussian.sample(3, &mut seeded(1)).expect("sample"),
        vec![1.0; 3]
    );
    assert_eq!(beta.sample(4, &mut seeded(1)).expect("sample").len(), 4);
    assert_eq!(uniform.sample(4, &mut seeded(1)).expect("sample").len(), 4);
    assert!(gaussian.sample(0, &mut seeded(1)).is_err());
    assert!(BetaArmPrior::new(0.0, 1.0).is_err());
    assert!(UniformArmPrior::new(2.0, 1.0).is_err());
}

#[test]
fn dynamics_preserve_domains_and_expected_step_semantics() {
    let means = [0.0, 1.0];
    assert_eq!(
        StationaryDynamics
            .apply(&means, 10, &mut seeded(1))
            .expect("apply"),
        means
    );
    let gradual = GradualDrift::new(0.1).expect("valid drift");
    assert_ne!(
        gradual.apply(&means, 1, &mut seeded(1)).expect("apply"),
        means
    );
    let abrupt = AbruptShift::new(2, 1.0, false).expect("valid shift");
    assert_eq!(
        abrupt.apply(&means, 0, &mut seeded(1)).expect("apply"),
        means
    );
    assert_ne!(
        abrupt.apply(&means, 2, &mut seeded(1)).expect("apply"),
        means
    );
    let swapped = RandomArmSwap::new(1.0)
        .expect("valid swap")
        .apply(&[0.0, 1.0, 2.0], 1, &mut seeded(2))
        .expect("apply");
    let mut sorted = swapped;
    sorted.sort_by(f64::total_cmp);
    assert_eq!(sorted, vec![0.0, 1.0, 2.0]);

    let probability = ProbabilityDrift::new(5.0, 1e-9).expect("valid drift");
    let drifted = probability
        .apply(&[0.0, 0.5, 1.0], 1, &mut seeded(3))
        .expect("apply");
    assert!(drifted.iter().all(|value| *value > 0.0 && *value < 1.0));
    assert!(BuiltInDynamics::Gradual(gradual).supports(RewardDomain::Real));
    assert!(!BuiltInDynamics::Gradual(gradual).supports(RewardDomain::Binary));
}

#[test]
fn classic_environment_validates_compatibility_clones_and_advances() {
    let mut environment = BanditEnvironment::new(
        vec![0.2, 0.8],
        BuiltInRewardModel::Bernoulli(BernoulliReward::new()),
        BuiltInDynamics::Probability(ProbabilityDrift::new(0.1, 1e-9).expect("valid drift")),
    )
    .expect("valid environment");
    let original = environment.clone();
    environment.advance(1, &mut seeded(5)).expect("advance");
    assert_ne!(environment.means(), original.means());
    assert_eq!(environment.n_arms(), 2);
    assert_eq!(environment.reward_domain(), RewardDomain::Binary);
    assert_eq!(original.expected_rewards(), &[0.2, 0.8]);
    assert_eq!(
        environment
            .sample_rewards(&mut seeded(6))
            .expect("sample")
            .len(),
        2
    );

    assert!(BanditEnvironment::new(
        vec![0.2, 0.8],
        BuiltInRewardModel::Bernoulli(BernoulliReward::new()),
        BuiltInDynamics::Gradual(GradualDrift::new(0.1).expect("valid drift")),
    )
    .is_err());
}

#[test]
fn context_providers_and_linear_environment_validate_shapes() {
    let fixed =
        FixedContextProvider::new(2, 2, vec![1.0, 0.0]).expect("shared feature vector is valid");
    assert_eq!(
        fixed.sample(&mut seeded(1)).expect("sample"),
        vec![1.0, 0.0, 1.0, 0.0]
    );
    let gaussian = GaussianContextProvider::new(2, 2, 0.0, 1.0).expect("valid provider");
    assert_eq!(gaussian.sample(&mut seeded(1)).expect("sample").len(), 4);

    let environment = LinearContextualEnvironment::new(
        2,
        2,
        vec![1.0, 0.0, 0.0, 2.0],
        fixed,
        BuiltInRewardModel::Gaussian(GaussianReward::new(1.0).expect("valid reward")),
    )
    .expect("valid environment");
    let context = environment.context(&mut seeded(1)).expect("context");
    assert_eq!(
        environment
            .expected_rewards(&context)
            .expect("expected rewards"),
        vec![1.0, 0.0]
    );
    assert!(environment.expected_rewards(&[1.0]).is_err());
}

#[test]
fn logistic_environment_clips_logits_and_requires_binary_rewards() {
    let provider =
        FixedContextProvider::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).expect("valid provider");
    let environment = LogisticContextualEnvironment::new(
        2,
        2,
        vec![100.0, 0.0, 0.0, -100.0],
        provider,
        BuiltInRewardModel::Bernoulli(BernoulliReward::new()),
    )
    .expect("valid environment");
    let context = environment.context(&mut seeded(1)).expect("context");
    let expected = environment
        .expected_rewards(&context)
        .expect("expected rewards");
    assert!(expected.iter().all(|value| (0.0..=1.0).contains(value)));
    assert!(expected[0] > 0.999);
    assert!(expected[1] < 0.001);
}
