use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use pymab::policy::action_value::ActionValueState;
use pymab::policy::basic::{GreedyPolicy, RandomPolicy};
use pymab::policy::bayesian_ucb::{BernoulliBayesianUCBPolicy, GaussianBayesianUCBPolicy};
use pymab::policy::epsilon_greedy::{DecayingEpsilonGreedyPolicy, EpsilonGreedyPolicy};
use pymab::policy::gradient::GradientBanditPolicy;
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
        policy
            .update(
                ActionIndex::new(integer(update, "action") as usize, policy.n_arms()).unwrap(),
                number(update, "reward"),
            )
            .unwrap();
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
