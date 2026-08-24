use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use pymab::policy::registry::PolicyKind;
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

#[allow(dead_code)]
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
