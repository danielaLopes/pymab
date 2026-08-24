"""Exact diagnostics captured around real PyMAB decisions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from pymab.policies import EpsilonGreedyPolicy, LinUCBPolicy


def epsilon_decision(
    policy: EpsilonGreedyPolicy, rng: np.random.Generator
) -> tuple[int, dict[str, Any]]:
    """Peek at the branch without mutating the stream, then call the real policy."""

    probe = deepcopy(rng)
    sampled = float(probe.random())
    branch = "explore" if sampled < policy.epsilon else "exploit"
    greedy = np.flatnonzero(policy.estimates == np.max(policy.estimates)).tolist()
    predicted = (
        int(probe.integers(policy.n_arms))
        if branch == "explore"
        else int(probe.choice(np.asarray(greedy, dtype=int)))
    )
    action = policy.select_action(rng=rng)
    if action != predicted:
        raise RuntimeError("epsilon diagnostic diverged from policy selection")
    return action, {
        "kind": "epsilon",
        "epsilon": policy.epsilon,
        "sampledRandom": sampled,
        "selectionBranch": branch,
        "greedyArms": greedy,
        "countsBefore": policy.counts.copy(),
        "estimatesBefore": policy.estimates.copy(),
    }


def linucb_decision(
    policy: LinUCBPolicy, context: np.ndarray, rng: np.random.Generator
) -> tuple[int, dict[str, Any]]:
    """Decompose public LinUCB scores before selecting with the real policy."""

    theta = np.stack(
        [np.linalg.solve(policy.a[i], policy.b[i]) for i in range(policy.n_arms)]
    )
    means = np.einsum("ij,ij->i", theta, context)
    raw_uncertainty = np.asarray(
        [
            np.sqrt(
                max(float(context[i] @ np.linalg.solve(policy.a[i], context[i])), 0.0)
            )
            for i in range(policy.n_arms)
        ]
    )
    bonuses = policy.alpha * raw_uncertainty
    scores = policy.upper_confidence_bounds(context)
    if not np.allclose(scores, means + bonuses):
        raise RuntimeError("LinUCB score decomposition diverged from public output")
    action = policy.select_action(context=context, rng=rng)
    return action, {
        "kind": "linucb",
        "contextMatrix": context.copy(),
        "thetaBefore": theta,
        "predictedMeans": means,
        "rawUncertainty": raw_uncertainty,
        "bonuses": bonuses,
        "ucbScores": scores,
        "aBefore": policy.a.copy(),
        "bBefore": policy.b.copy(),
    }
