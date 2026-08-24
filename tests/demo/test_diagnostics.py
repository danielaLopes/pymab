from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pymab_demo.diagnostics import epsilon_decision, linucb_decision

from pymab.policies import EpsilonGreedyPolicy, LinUCBPolicy


def test_epsilon_peek_matches_policy_without_mutating_original_stream() -> None:
    policy = EpsilonGreedyPolicy(n_arms=3, epsilon=0.2)
    rng = np.random.default_rng(91)
    expected_rng = deepcopy(rng)
    action, diagnostic = epsilon_decision(policy, rng)
    expected = policy.select_action(rng=expected_rng)
    assert action == expected
    assert diagnostic["selectionBranch"] in {"explore", "exploit"}
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_linucb_decomposition_matches_public_scores() -> None:
    policy = LinUCBPolicy(n_arms=3, n_features=4, alpha=1.25, l2=1.0)
    context = np.array([[1.0, -1, 1, -1]] * 3)
    _, diagnostic = linucb_decision(policy, context, np.random.default_rng(2))
    np.testing.assert_allclose(
        diagnostic["predictedMeans"] + diagnostic["bonuses"],
        policy.upper_confidence_bounds(context),
    )


def test_diagnostic_invariants_reject_divergent_policy_outputs() -> None:
    epsilon = EpsilonGreedyPolicy(n_arms=3, epsilon=0.2)
    epsilon.select_action = lambda *, rng: 99  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="epsilon diagnostic"):
        epsilon_decision(epsilon, np.random.default_rng(1))

    linucb = LinUCBPolicy(n_arms=3, n_features=4)
    linucb.upper_confidence_bounds = lambda context: np.zeros(3)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="score decomposition"):
        linucb_decision(linucb, np.ones((3, 4)), np.random.default_rng(1))
