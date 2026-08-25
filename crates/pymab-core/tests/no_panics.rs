use std::panic::{catch_unwind, AssertUnwindSafe};

use proptest::prelude::*;
use pymab::distribution::{GaussianReward, RewardModel, UniformReward};
use pymab::policy::basic::GreedyPolicy;
use pymab::policy::Policy;
use pymab::types::{ActionIndex, ContextShape};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

proptest! {
    #[test]
    fn malformed_public_inputs_return_errors_without_panicking(
        n_arms in 0_usize..32,
        action in 0_usize..64,
        reward in any::<f64>(),
        context in prop::collection::vec(any::<f64>(), 0..64),
    ) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let _ = ActionIndex::new(action, n_arms);
            if let Ok(shape) = ContextShape::new(n_arms, 3) {
                let _ = shape.validate_flat(&context);
            }
            if let Ok(mut policy) = GreedyPolicy::new(n_arms, 0.0) {
                if let Ok(index) = ActionIndex::new(action, n_arms) {
                    let _ = Policy::update(&mut policy, index, reward);
                }
            }
            let mut rng = ChaCha12Rng::seed_from_u64(1);
            if let Ok(model) = GaussianReward::new(reward) {
                let _ = model.sample(&context, &mut rng);
            }
            if let Ok(model) = UniformReward::new(reward) {
                let _ = model.sample(&context, &mut rng);
            }
        }));
        prop_assert!(outcome.is_ok());
    }
}
