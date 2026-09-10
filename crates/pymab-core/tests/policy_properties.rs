use proptest::prelude::*;
use pymab::environment::dynamics::{EnvironmentDynamics, ProbabilityDrift};
use pymab::policy::basic::{GreedyPolicy, RandomPolicy};
use pymab::policy::contextual::{LinUCBPolicy, LinearEpsilonGreedyPolicy};
use pymab::policy::{ContextualPolicy, Policy};
use pymab::types::ActionIndex;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

proptest! {
    #[test]
    fn classic_policies_return_valid_actions_and_finite_state(
        n_arms in 1_usize..16,
        rewards in prop::collection::vec(-1000.0_f64..1000.0, 1..100),
        seed in any::<u64>(),
    ) {
        let mut random = RandomPolicy::new(n_arms).expect("valid policy");
        let mut greedy = GreedyPolicy::new(n_arms, 0.0).expect("valid policy");
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        for reward in rewards {
            let action = Policy::select_action(&mut random, &mut rng).expect("selection");
            prop_assert!(action.get() < n_arms);
            Policy::update(&mut random, action, reward).expect("update");
            Policy::update(&mut greedy, action, reward).expect("update");
            prop_assert!(greedy.state().estimates().iter().all(|value| value.is_finite()));
        }
        Policy::reset(&mut greedy);
        prop_assert_eq!(greedy.state().step(), 0);
        prop_assert!(greedy.state().counts().iter().all(|value| *value == 0));
    }

    #[test]
    fn contextual_policies_stay_finite_on_well_conditioned_inputs(
        rewards in prop::collection::vec(-10.0_f64..10.0, 1..40),
        seed in any::<u64>(),
    ) {
        let mut linear = LinearEpsilonGreedyPolicy::new(3, 2, 0.2, 0.1)
            .expect("valid policy");
        let mut lin_ucb = LinUCBPolicy::new(3, 2, 1.0, 1.0).expect("valid policy");
        let context = [1.0, 0.0, 0.0, 1.0, 0.5, -0.5];
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        for reward in rewards {
            let action = ContextualPolicy::select_action(&mut linear, &context, &mut rng)
                .expect("selection");
            prop_assert!(action.get() < 3);
            ContextualPolicy::update(&mut linear, action, reward, &context).expect("update");
            ContextualPolicy::update(&mut lin_ucb, action, reward, &context).expect("update");
            prop_assert!(linear.state().theta().iter().all(|value| value.is_finite()));
            prop_assert!(lin_ucb.state().a().iter().all(|value| value.is_finite()));
            prop_assert!(lin_ucb.state().b().iter().all(|value| value.is_finite()));
        }
    }

    #[test]
    fn probability_drift_never_leaves_open_unit_interval(
        means in prop::collection::vec(0.0_f64..=1.0, 1..20),
        seed in any::<u64>(),
    ) {
        let drift = ProbabilityDrift::new(5.0, 1e-9).expect("valid drift");
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let values = drift.apply(&means, 1, &mut rng).expect("drift");
        prop_assert!(values.iter().all(|value| *value > 0.0 && *value < 1.0));
    }

    #[test]
    fn checked_action_indices_never_accept_out_of_range_values(
        n_arms in 1_usize..1000,
        offset in 0_usize..1000,
    ) {
        prop_assert!(ActionIndex::new(n_arms + offset, n_arms).is_err());
    }
}
