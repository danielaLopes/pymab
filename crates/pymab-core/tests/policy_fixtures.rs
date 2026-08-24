use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use pymab::policy::action_value::ActionValueState;
use pymab::policy::adversarial::EXP3Policy;
use pymab::policy::basic::{GreedyPolicy, RandomPolicy};
use pymab::policy::bayesian_ucb::{BernoulliBayesianUCBPolicy, GaussianBayesianUCBPolicy};
use pymab::policy::change_detection::{
    CUSUMUCBPolicy, ChangeDetector, ChangePointState, ChangePointUCBPolicy, PageHinkleyUCBPolicy,
};
use pymab::policy::epsilon_greedy::{DecayingEpsilonGreedyPolicy, EpsilonGreedyPolicy};
use pymab::policy::gradient::GradientBanditPolicy;
use pymab::policy::nonstationary::{
    DiscountedBernoulliState, DiscountedBernoulliThompsonSamplingPolicy, DiscountedUCBPolicy,
    DiscountedUCBState, SlidingWindowBernoulliState, SlidingWindowBernoulliThompsonSamplingPolicy,
    SlidingWindowUCBPolicy, SlidingWindowUCBState,
};
use pymab::policy::pure_exploration::{MedianEliminationPolicy, SuccessiveEliminationPolicy};
use pymab::policy::registry::PolicyKind;
use pymab::policy::softmax::SoftmaxPolicy;
use pymab::policy::thompson::{
    BernoulliPosteriorState, BernoulliThompsonSamplingPolicy, GaussianPosteriorState,
    GaussianThompsonSamplingPolicy,
};
use pymab::policy::ucb::{KLUCBPolicy, MOSSPolicy, UCBPolicy};
use pymab::policy::Policy;
use pymab::types::ActionIndex;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    schema_version: u64,
    policies: Vec<RegistryEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryEntry {
    python_name: String,
    rust_kind: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyFixture {
    schema_version: u64,
    policy_kind: String,
    config: Value,
    updates: Vec<Value>,
    checkpoints: Vec<Value>,
    recommendation: Value,
    reset_state: Value,
    expected_error: Option<String>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/policies")
        .join(name)
}

fn read_policy_fixture(name: &str) -> PolicyFixture {
    let payload = fs::read_to_string(fixture_path(name)).expect("fixture exists");
    serde_json::from_str(&payload).expect("strict policy fixture")
}

fn integer(value: &Value, field: &str) -> u64 {
    value[field]
        .as_u64()
        .unwrap_or_else(|| panic!("{field} is an unsigned integer"))
}

fn number(value: &Value, field: &str) -> f64 {
    value[field]
        .as_f64()
        .unwrap_or_else(|| panic!("{field} is numeric"))
}

fn boolean(value: &Value, field: &str) -> bool {
    value[field]
        .as_bool()
        .unwrap_or_else(|| panic!("{field} is boolean"))
}

fn assert_action_value_state(state: &ActionValueState, expected: &Value) {
    assert_eq!(state.step(), integer(expected, "step"));
    assert_eq!(state.total_reward(), number(expected, "total_reward"));
    let counts: Vec<_> = expected["counts"]
        .as_array()
        .expect("counts array")
        .iter()
        .map(|value| value.as_u64().expect("integer count"))
        .collect();
    let estimates: Vec<_> = expected["estimates"]
        .as_array()
        .expect("estimates array")
        .iter()
        .map(|value| value.as_f64().expect("numeric estimate"))
        .collect();
    assert_eq!(state.counts(), counts);
    assert_eq!(state.estimates(), estimates);
}

fn assert_bernoulli_state(state: &BernoulliPosteriorState, expected: &Value) {
    assert_action_value_state(state.action_values(), expected);
    let successes: Vec<_> = expected["successes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect();
    let failures: Vec<_> = expected["failures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap())
        .collect();
    assert_eq!(state.successes(), successes);
    assert_eq!(state.failures(), failures);
}

fn assert_gaussian_state(state: &GaussianPosteriorState, expected: &Value) {
    assert_action_value_state(state.action_values(), expected);
    let means: Vec<_> = expected["means"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let precisions: Vec<_> = expected["precisions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    assert_eq!(state.means(), means);
    assert_eq!(state.precisions(), precisions);
}

fn apply_updates<P: Policy>(policy: &mut P, fixture: &PolicyFixture) {
    for update in &fixture.updates {
        let repeat = update
            .get("repeat")
            .map_or(1, |value| value.as_u64().unwrap());
        for _ in 0..repeat {
            policy
                .update(
                    ActionIndex::new(integer(update, "action") as usize, policy.n_arms()).unwrap(),
                    number(update, "reward"),
                )
                .unwrap();
        }
    }
}

fn assert_basic_fixture<P>(mut policy: P, fixture: &PolicyFixture)
where
    P: Policy<State = ActionValueState>,
{
    assert_eq!(fixture.schema_version, 1);
    assert!(fixture.expected_error.is_none());
    for update in &fixture.updates {
        let action = integer(update, "action") as usize;
        policy
            .update(
                ActionIndex::new(action, policy.n_arms()).expect("valid fixture action"),
                number(update, "reward"),
            )
            .expect("valid fixture update");
    }
    let checkpoint = fixture.checkpoints.last().expect("final checkpoint");
    assert_eq!(
        integer(checkpoint, "after_update") as usize,
        fixture.updates.len()
    );
    assert_action_value_state(policy.state(), &checkpoint["state"]);
    assert_eq!(
        policy.recommend_action().unwrap().get() as u64,
        fixture.recommendation.as_u64().expect("recommendation")
    );

    policy.reset();
    assert_action_value_state(policy.state(), &fixture.reset_state);
}

#[test]
fn fixture_registry_covers_every_rust_policy_kind() {
    let payload = fs::read_to_string(fixture_path("registry.json")).expect("registry exists");
    let registry: Registry = serde_json::from_str(&payload).expect("strict registry schema");
    assert_eq!(registry.schema_version, 1);

    let registered: BTreeSet<_> = registry
        .policies
        .iter()
        .map(|entry| (entry.rust_kind.as_str(), entry.python_name.as_str()))
        .collect();
    let rust_kinds: BTreeSet<_> = PolicyKind::ALL
        .iter()
        .map(|kind| (kind.as_str(), kind.python_name()))
        .collect();

    assert_eq!(registered, rust_kinds);
    assert_eq!(registry.policies.len(), PolicyKind::ALL.len());
}

#[test]
fn policy_fixture_loader_rejects_unknown_and_incomplete_fields() {
    let complete = r#"{
        "schema_version": 1,
        "policy_kind": "greedy",
        "config": {},
        "updates": [],
        "checkpoints": [],
        "recommendation": 0,
        "reset_state": {},
        "expected_error": null
    }"#;
    assert!(serde_json::from_str::<PolicyFixture>(complete).is_ok());

    let unknown = complete.replace("\"config\": {}", "\"unknown\": 1, \"config\": {}");
    assert!(serde_json::from_str::<PolicyFixture>(&unknown).is_err());

    let incomplete = complete.replace("\"updates\": [],", "");
    assert!(serde_json::from_str::<PolicyFixture>(&incomplete).is_err());
}

#[test]
fn basic_policy_fixtures_match_rust_state() {
    for name in [
        "random.json",
        "greedy.json",
        "epsilon_greedy.json",
        "decaying_epsilon_greedy.json",
        "softmax.json",
    ] {
        let fixture = read_policy_fixture(name);
        let config = &fixture.config;
        let n_arms = integer(config, "n_arms") as usize;
        match fixture.policy_kind.as_str() {
            "random" => assert_basic_fixture(RandomPolicy::new(n_arms).unwrap(), &fixture),
            "greedy" => assert_basic_fixture(
                GreedyPolicy::new(n_arms, number(config, "initial_value")).unwrap(),
                &fixture,
            ),
            "epsilon_greedy" => assert_basic_fixture(
                EpsilonGreedyPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "epsilon"),
                )
                .unwrap(),
                &fixture,
            ),
            "decaying_epsilon_greedy" => assert_basic_fixture(
                DecayingEpsilonGreedyPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "initial_epsilon"),
                    number(config, "min_epsilon"),
                    number(config, "decay_rate"),
                )
                .unwrap(),
                &fixture,
            ),
            "softmax" => assert_basic_fixture(
                SoftmaxPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "temperature"),
                )
                .unwrap(),
                &fixture,
            ),
            other => panic!("unexpected basic policy fixture {other}"),
        }
    }
}

#[test]
fn ucb_policy_fixtures_match_rust_state() {
    for name in ["ucb.json", "kl_ucb.json", "moss.json"] {
        let fixture = read_policy_fixture(name);
        let config = &fixture.config;
        let n_arms = integer(config, "n_arms") as usize;
        let initial_value = number(config, "initial_value");
        let c = number(config, "c");
        match fixture.policy_kind.as_str() {
            "ucb" => assert_basic_fixture(
                UCBPolicy::new(n_arms, initial_value, c, number(config, "reward_scale")).unwrap(),
                &fixture,
            ),
            "kl_ucb" => assert_basic_fixture(
                KLUCBPolicy::new(
                    n_arms,
                    initial_value,
                    c,
                    number(config, "tolerance"),
                    integer(config, "max_iterations") as usize,
                )
                .unwrap(),
                &fixture,
            ),
            "moss" => assert_basic_fixture(
                MOSSPolicy::new(
                    n_arms,
                    initial_value,
                    integer(config, "horizon"),
                    c,
                    number(config, "reward_scale"),
                )
                .unwrap(),
                &fixture,
            ),
            other => panic!("unexpected UCB policy fixture {other}"),
        }
    }
}

#[test]
fn posterior_policy_fixtures_match_rust_state() {
    for name in [
        "gradient.json",
        "bernoulli_thompson.json",
        "gaussian_thompson.json",
        "bernoulli_bayesian_ucb.json",
        "gaussian_bayesian_ucb.json",
    ] {
        let fixture = read_policy_fixture(name);
        let config = &fixture.config;
        let n_arms = integer(config, "n_arms") as usize;
        let expected = &fixture.checkpoints.last().unwrap()["state"];
        match fixture.policy_kind.as_str() {
            "gradient_bandit" => {
                let mut policy = GradientBanditPolicy::new(
                    n_arms,
                    number(config, "learning_rate"),
                    boolean(config, "use_baseline"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_eq!(policy.state().step(), integer(expected, "step"));
                assert_eq!(
                    policy.state().average_reward(),
                    number(expected, "average_reward")
                );
                assert_eq!(
                    policy.state().preferences(),
                    expected["preferences"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_f64().unwrap())
                        .collect::<Vec<_>>()
                );
                policy.reset();
                assert_eq!(policy.state().step(), 0);
            }
            "bernoulli_thompson_sampling" => {
                let mut policy = BernoulliThompsonSamplingPolicy::new(
                    n_arms,
                    number(config, "alpha_prior"),
                    number(config, "beta_prior"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_bernoulli_state(policy.state(), expected);
                policy.reset();
                assert_bernoulli_state(policy.state(), &fixture.reset_state);
            }
            "gaussian_thompson_sampling" => {
                let mut policy = GaussianThompsonSamplingPolicy::new(
                    n_arms,
                    number(config, "prior_mean"),
                    number(config, "prior_precision"),
                    number(config, "reward_precision"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_gaussian_state(policy.state(), expected);
                policy.reset();
                assert_gaussian_state(policy.state(), &fixture.reset_state);
            }
            "bernoulli_bayesian_ucb" => {
                let mut policy = BernoulliBayesianUCBPolicy::new(
                    n_arms,
                    number(config, "alpha_prior"),
                    number(config, "beta_prior"),
                    number(config, "quantile"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_bernoulli_state(policy.state(), expected);
                let expected_bounds =
                    &fixture.checkpoints.last().unwrap()["scores"]["upper_bounds"];
                for (actual, expected) in policy
                    .upper_bounds()
                    .unwrap()
                    .iter()
                    .zip(expected_bounds.as_array().unwrap())
                {
                    assert!((actual - expected.as_f64().unwrap()).abs() < 1e-12);
                }
                policy.reset();
                assert_bernoulli_state(policy.state(), &fixture.reset_state);
            }
            "gaussian_bayesian_ucb" => {
                let mut policy = GaussianBayesianUCBPolicy::new(
                    n_arms,
                    number(config, "prior_mean"),
                    number(config, "prior_precision"),
                    number(config, "reward_precision"),
                    number(config, "quantile"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_gaussian_state(policy.state(), expected);
                let expected_bounds =
                    &fixture.checkpoints.last().unwrap()["scores"]["upper_bounds"];
                for (actual, expected) in policy
                    .upper_bounds()
                    .unwrap()
                    .iter()
                    .zip(expected_bounds.as_array().unwrap())
                {
                    assert!((actual - expected.as_f64().unwrap()).abs() < 1e-12);
                }
                policy.reset();
                assert_gaussian_state(policy.state(), &fixture.reset_state);
            }
            other => panic!("unexpected posterior fixture {other}"),
        }
    }
}

fn bool_values(value: &Value, field: &str) -> Vec<bool> {
    value[field]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_bool().unwrap())
        .collect()
}

fn float_values(value: &Value, field: &str) -> Vec<f64> {
    value[field]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap())
        .collect()
}

fn integer_values(value: &Value, field: &str) -> Vec<u64> {
    value[field]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_u64().unwrap())
        .collect()
}

fn assert_float_slices_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "{actual} does not match {expected}"
        );
    }
}

#[test]
fn exploration_policy_fixtures_match_rust_state() {
    for name in [
        "exp3.json",
        "successive_elimination.json",
        "median_elimination.json",
    ] {
        let fixture = read_policy_fixture(name);
        let config = &fixture.config;
        let n_arms = integer(config, "n_arms") as usize;
        let expected = &fixture.checkpoints.last().unwrap()["state"];
        match fixture.policy_kind.as_str() {
            "exp3" => {
                let mut policy = EXP3Policy::new(
                    n_arms,
                    number(config, "gamma"),
                    Some(number(config, "learning_rate")),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_action_value_state(policy.state().action_values(), expected);
                assert_eq!(
                    policy.state().log_weights(),
                    float_values(expected, "log_weights")
                );
                assert_eq!(
                    policy.state().last_probabilities(),
                    float_values(expected, "last_probabilities")
                );
                policy.reset();
                assert_action_value_state(policy.state().action_values(), &fixture.reset_state);
            }
            "successive_elimination" => {
                let mut policy = SuccessiveEliminationPolicy::new(
                    n_arms,
                    number(config, "delta"),
                    number(config, "confidence_scale"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_action_value_state(policy.state().action_values(), expected);
                assert_eq!(policy.state().active(), bool_values(expected, "active"));
                policy.reset();
                assert_eq!(
                    policy.state().active(),
                    bool_values(&fixture.reset_state, "active")
                );
            }
            "median_elimination" => {
                let mut policy = MedianEliminationPolicy::new(
                    n_arms,
                    number(config, "epsilon"),
                    number(config, "delta"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_action_value_state(policy.state().action_values(), expected);
                assert_eq!(policy.state().active(), bool_values(expected, "active"));
                assert_eq!(
                    policy.state().phase_counts(),
                    expected["phase_counts"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| value.as_u64().unwrap())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    policy.phase_quota(),
                    integer(&fixture.checkpoints[0]["scores"], "phase_quota")
                );
                policy.reset();
                assert_eq!(
                    policy.state().active(),
                    bool_values(&fixture.reset_state, "active")
                );
            }
            other => panic!("unexpected exploration fixture {other}"),
        }
    }
}

fn assert_sliding_ucb_state(state: &SlidingWindowUCBState, expected: &Value) {
    assert_eq!(state.step(), integer(expected, "step"));
    assert_eq!(state.total_reward(), number(expected, "total_reward"));
    assert_eq!(state.counts(), integer_values(expected, "counts"));
    assert_eq!(state.estimates(), float_values(expected, "estimates"));
    assert_eq!(state.history_len() as u64, integer(expected, "history_len"));
}

fn assert_discounted_ucb_state(state: &DiscountedUCBState, expected: &Value) {
    assert_eq!(state.step(), integer(expected, "step"));
    assert_eq!(state.total_reward(), number(expected, "total_reward"));
    assert_eq!(state.counts(), integer_values(expected, "counts"));
    assert_eq!(state.estimates(), float_values(expected, "estimates"));
    assert_eq!(
        state.discounted_counts(),
        float_values(expected, "discounted_counts")
    );
    assert_eq!(
        state.discounted_sums(),
        float_values(expected, "discounted_sums")
    );
}

fn assert_sliding_bernoulli_state(state: &SlidingWindowBernoulliState, expected: &Value) {
    assert_eq!(state.step(), integer(expected, "step"));
    assert_eq!(state.total_reward(), number(expected, "total_reward"));
    assert_eq!(state.counts(), integer_values(expected, "counts"));
    assert_eq!(state.estimates(), float_values(expected, "estimates"));
    assert_eq!(state.successes(), integer_values(expected, "successes"));
    assert_eq!(state.failures(), integer_values(expected, "failures"));
    assert_eq!(state.history_len() as u64, integer(expected, "history_len"));
}

fn assert_discounted_bernoulli_state(state: &DiscountedBernoulliState, expected: &Value) {
    assert_eq!(state.step(), integer(expected, "step"));
    assert_eq!(state.total_reward(), number(expected, "total_reward"));
    assert_eq!(state.counts(), float_values(expected, "counts"));
    assert_eq!(state.estimates(), float_values(expected, "estimates"));
    assert_eq!(state.successes(), float_values(expected, "successes"));
    assert_eq!(state.failures(), float_values(expected, "failures"));
}

fn assert_change_state(state: &ChangePointState, expected: &Value) {
    assert_action_value_state(state.action_values(), expected);
    assert_eq!(
        state.detector_counts(),
        integer_values(expected, "detector_counts")
    );
    assert_eq!(
        state.detector_means(),
        float_values(expected, "detector_means")
    );
    assert_eq!(
        state.positive_cusum(),
        float_values(expected, "positive_cusum")
    );
    assert_eq!(
        state.negative_cusum(),
        float_values(expected, "negative_cusum")
    );
    assert_eq!(
        state.ph_cumulative(),
        float_values(expected, "ph_cumulative")
    );
    assert_eq!(state.ph_minimum(), float_values(expected, "ph_minimum"));
    assert_eq!(
        state.change_counts(),
        integer_values(expected, "change_counts")
    );
}

fn assert_recommendation<P: Policy>(policy: &P, fixture: &PolicyFixture) {
    assert_eq!(
        policy.recommend_action().unwrap().get() as u64,
        fixture.recommendation.as_u64().unwrap()
    );
}

#[test]
fn adaptive_policy_fixtures_match_rust_state() {
    for name in [
        "sliding_window_ucb.json",
        "discounted_ucb.json",
        "sliding_window_bernoulli_thompson.json",
        "discounted_bernoulli_thompson.json",
        "change_point_ucb.json",
        "cusum_ucb.json",
        "page_hinkley_ucb.json",
    ] {
        let fixture = read_policy_fixture(name);
        let config = &fixture.config;
        let n_arms = integer(config, "n_arms") as usize;
        let expected = &fixture.checkpoints.last().unwrap()["state"];
        match fixture.policy_kind.as_str() {
            "sliding_window_ucb" => {
                let mut policy = SlidingWindowUCBPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "c"),
                    number(config, "reward_scale"),
                    integer(config, "window_size") as usize,
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_sliding_ucb_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                let bonuses = &fixture.checkpoints.last().unwrap()["scores"]["bonuses"];
                let bonuses = bonuses
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .collect::<Vec<_>>();
                assert_float_slices_close(&policy.confidence_bonus(), &bonuses);
                policy.reset();
                assert_sliding_ucb_state(policy.state(), &fixture.reset_state);
            }
            "discounted_ucb" => {
                let mut policy = DiscountedUCBPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "c"),
                    number(config, "reward_scale"),
                    number(config, "discount_factor"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_discounted_ucb_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                let bonuses = fixture.checkpoints.last().unwrap()["scores"]["bonuses"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_f64().unwrap())
                    .collect::<Vec<_>>();
                assert_float_slices_close(&policy.confidence_bonus(), &bonuses);
                policy.reset();
                assert_discounted_ucb_state(policy.state(), &fixture.reset_state);
            }
            "sliding_window_bernoulli_thompson" => {
                let mut policy = SlidingWindowBernoulliThompsonSamplingPolicy::new(
                    n_arms,
                    number(config, "alpha_prior"),
                    number(config, "beta_prior"),
                    integer(config, "window_size") as usize,
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_sliding_bernoulli_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                policy.reset();
                assert_sliding_bernoulli_state(policy.state(), &fixture.reset_state);
            }
            "discounted_bernoulli_thompson" => {
                let mut policy = DiscountedBernoulliThompsonSamplingPolicy::new(
                    n_arms,
                    number(config, "alpha_prior"),
                    number(config, "beta_prior"),
                    number(config, "discount_factor"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_discounted_bernoulli_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                policy.reset();
                assert_discounted_bernoulli_state(policy.state(), &fixture.reset_state);
            }
            "change_point_ucb" => {
                let detector = match config["detector"].as_str().unwrap() {
                    "cusum" => ChangeDetector::Cusum,
                    "page_hinkley" => ChangeDetector::PageHinkley,
                    other => panic!("unexpected detector {other}"),
                };
                let mut policy = ChangePointUCBPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "c"),
                    number(config, "reward_scale"),
                    detector,
                    number(config, "threshold"),
                    number(config, "drift"),
                    integer(config, "min_observations"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_change_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                policy.reset();
                assert_change_state(policy.state(), &fixture.reset_state);
            }
            "cusum_ucb" => {
                let mut policy = CUSUMUCBPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "c"),
                    number(config, "reward_scale"),
                    number(config, "threshold"),
                    number(config, "drift"),
                    integer(config, "min_observations"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_change_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                policy.reset();
                assert_change_state(policy.state(), &fixture.reset_state);
            }
            "page_hinkley_ucb" => {
                let mut policy = PageHinkleyUCBPolicy::new(
                    n_arms,
                    number(config, "initial_value"),
                    number(config, "c"),
                    number(config, "reward_scale"),
                    number(config, "threshold"),
                    number(config, "drift"),
                    integer(config, "min_observations"),
                )
                .unwrap();
                apply_updates(&mut policy, &fixture);
                assert_change_state(policy.state(), expected);
                assert_recommendation(&policy, &fixture);
                policy.reset();
                assert_change_state(policy.state(), &fixture.reset_state);
            }
            other => panic!("unexpected adaptive policy fixture {other}"),
        }
    }
}
