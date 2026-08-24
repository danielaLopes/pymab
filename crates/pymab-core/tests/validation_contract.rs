use pymab::error::{ErrorCode, PyMabError};
use pymab::types::{ActionIndex, ContextShape, PolicyCapabilities, PolicyObjective, RewardDomain};
use pymab::validation::{finite, probability, reward, strictly_positive};

#[test]
fn finite_numbers_and_positive_configuration_are_checked() {
    assert_eq!(finite("temperature", 0.25).expect("finite"), 0.25);
    assert_eq!(
        strictly_positive("temperature", 0.25).expect("positive"),
        0.25
    );

    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = finite("temperature", value).expect_err("non-finite rejected");
        assert_eq!(error.code(), ErrorCode::Validation);
    }
    let error = strictly_positive("temperature", 0.0).expect_err("zero rejected");
    assert_eq!(error.code(), ErrorCode::Configuration);
}

#[test]
fn reward_domains_enforce_their_mathematical_support() {
    assert_eq!(
        reward("reward", 4.5, RewardDomain::Real).expect("real"),
        4.5
    );
    assert_eq!(
        reward("reward", 0.25, RewardDomain::UnitInterval).expect("unit interval"),
        0.25
    );
    assert_eq!(
        reward("reward", 1.0, RewardDomain::Binary).expect("binary"),
        1.0
    );

    assert!(reward("reward", -0.1, RewardDomain::UnitInterval).is_err());
    assert!(reward("reward", 0.5, RewardDomain::Binary).is_err());
}

#[test]
fn probabilities_include_both_boundaries() {
    assert_eq!(probability("epsilon", 0.0).expect("zero"), 0.0);
    assert_eq!(probability("epsilon", 1.0).expect("one"), 1.0);
    assert!(probability("epsilon", -f64::EPSILON).is_err());
    assert!(probability("epsilon", 1.0 + f64::EPSILON).is_err());
}

#[test]
fn action_indices_are_checked_without_panicking() {
    assert_eq!(ActionIndex::new(2, 3).expect("valid").get(), 2);
    let error = ActionIndex::new(3, 3).expect_err("out of range");
    assert_eq!(error.code(), ErrorCode::Validation);
    assert!(matches!(error, PyMabError::Validation { .. }));

    let error = ActionIndex::new(0, 0).expect_err("zero arms invalid");
    assert_eq!(error.code(), ErrorCode::Configuration);
}

#[test]
fn context_shapes_reject_empty_and_mismatched_inputs() {
    let shape = ContextShape::new(3, 2).expect("valid shape");
    assert_eq!(shape.element_count(), 6);
    shape
        .validate_flat(&[1.0, 0.0, 0.5, 0.5, -1.0, 2.0])
        .expect("valid context");

    assert!(ContextShape::new(0, 2).is_err());
    assert!(ContextShape::new(3, 0).is_err());
    assert!(shape.validate_flat(&[0.0; 5]).is_err());
    assert!(shape
        .validate_flat(&[0.0, 0.0, 0.0, f64::NAN, 0.0, 0.0])
        .is_err());
}

#[test]
fn capability_metadata_reports_compatibility() {
    const DOMAINS: &[RewardDomain] = &[RewardDomain::Binary, RewardDomain::UnitInterval];
    let capabilities = PolicyCapabilities::new(false, DOMAINS, PolicyObjective::BestArm);

    assert!(capabilities.supports(RewardDomain::Binary));
    assert!(capabilities.supports(RewardDomain::UnitInterval));
    assert!(!capabilities.supports(RewardDomain::Real));
    assert!(!capabilities.contextual());
    assert_eq!(capabilities.objective(), PolicyObjective::BestArm);
}

#[test]
fn public_error_categories_are_stable() {
    let errors = [
        (
            PyMabError::configuration("epsilon", "must be a probability"),
            ErrorCode::Configuration,
        ),
        (
            PyMabError::validation("reward", "must be finite"),
            ErrorCode::Validation,
        ),
        (
            PyMabError::compatibility("policy", "requires binary rewards"),
            ErrorCode::Compatibility,
        ),
        (
            PyMabError::numerical("cholesky", "matrix is not positive definite"),
            ErrorCode::Numerical,
        ),
        (
            PyMabError::internal("unreachable policy state"),
            ErrorCode::Internal,
        ),
    ];

    for (error, code) in errors {
        assert_eq!(error.code(), code);
        assert!(!error.to_string().is_empty());
    }
}
