"""Generate complete public-PyMAB examples for the browser Lab."""

from __future__ import annotations

from typing import Any

from pymab._random import stable_seed


def epsilon_example(
    *,
    seed: int,
    epsilon: float,
    horizon: int,
    probabilities: tuple[float, ...],
    initial_value: float = 0.0,
) -> str:
    """Return a standalone epsilon-greedy reproduction."""

    action_seed = stable_seed(seed, "epsilon-greedy", "lesson", "action")
    reward_seed = stable_seed(seed, "epsilon-greedy", "lesson", "reward")
    means = ", ".join(repr(value) for value in probabilities)
    return f"""import numpy as np
from pymab.policies import EpsilonGreedyPolicy

means = np.array([{means}])
policy = EpsilonGreedyPolicy(n_arms=3, epsilon={epsilon!r}, initial_value={initial_value!r})
action_rng = np.random.default_rng(np.random.SeedSequence({action_seed}))
reward_rng = np.random.default_rng(np.random.SeedSequence({reward_seed}))
total_reward = 0
cumulative_regret = 0.0
for _ in range({horizon}):
    potential_rewards = (reward_rng.random(3) < means).astype(int)
    action = policy.select_action(rng=action_rng)
    reward = int(potential_rewards[action])
    policy.update(action=action, reward=float(reward))
    total_reward += reward
    cumulative_regret += float(means.max() - means[action])
print({{"totalReward": total_reward, "cumulativeExpectedRegret": cumulative_regret}})
"""


def linucb_example(
    *, seed: int, alpha: float, l2: float, horizon: int, theta: object | None = None
) -> str:
    """Return a standalone LinUCB reproduction."""

    context_seed = stable_seed(seed, "arcade", 1, "context")
    action_seed = stable_seed(seed, "arcade", 1, "action")
    reward_seed = stable_seed(seed, "arcade", 1, "reward")
    theta_value = (
        [[0.1, -1.2, 0.2, -0.8], [0.0, 1.0, 0.3, 1.0], [0.2, 0.0, -1.1, 0.2]]
        if theta is None
        else getattr(theta, "tolist", lambda: theta)()
    )
    return f"""import numpy as np
from pymab.policies import LinUCBPolicy

theta = np.array({theta_value!r})
policy = LinUCBPolicy(n_arms=3, n_features=4, alpha={alpha!r}, l2={l2!r})
context_rng = np.random.default_rng(np.random.SeedSequence({context_seed}))
action_rng = np.random.default_rng(np.random.SeedSequence({action_seed}))
reward_rng = np.random.default_rng(np.random.SeedSequence({reward_seed}))
total_reward = 0
cumulative_regret = 0.0
for _ in range({horizon}):
    feature = np.concatenate((np.ones(1), context_rng.choice(np.array([-1.0, 1.0]), size=3)))
    context = np.repeat(feature[np.newaxis, :], 3, axis=0)
    probabilities = 1.0 / (1.0 + np.exp(-(theta @ feature)))
    potential_rewards = (reward_rng.random(3) < probabilities).astype(int)
    action = policy.select_action(context=context, rng=action_rng)
    reward = int(potential_rewards[action])
    policy.update(action=action, reward=float(reward), context=context)
    total_reward += reward
    cumulative_regret += float(probabilities.max() - probabilities[action])
print({{"totalReward": total_reward, "cumulativeExpectedRegret": cumulative_regret}})
"""


def catalog_example(
    *,
    policy_id: str,
    class_name: str,
    constructor: dict[str, object],
    environment_kind: str,
    objective: str,
    seed: int,
    horizon: int,
    environment: dict[str, Any],
) -> str:
    """Return a standalone replay for a catalog-driven session."""

    action_seed = stable_seed(seed, policy_id, "arcade", "action")
    reward_seed = stable_seed(seed, policy_id, "arcade", "reward")
    context_seed = stable_seed(seed, policy_id, "arcade", "context")
    arguments = ", ".join(f"{name}={value!r}" for name, value in constructor.items())
    setup: list[str] = []
    round_setup: list[str]
    if environment_kind in {"stationary-bernoulli", "best-arm"}:
        probabilities = environment.get("probabilities", (0.25, 0.5, 0.75))
        setup.append(f"probabilities = np.array({list(probabilities)!r}, dtype=float)")
        round_setup = [
            "    expected = probabilities",
            "    potential_rewards = (reward_rng.random(3) < probabilities).astype(float)",
        ]
    elif environment_kind == "stationary-gaussian":
        means = environment.get("means", (-0.25, 0.3, 0.85))
        standard_deviation = environment.get("standardDeviation", 0.5)
        setup.extend(
            [
                f"means = np.array({list(means)!r}, dtype=float)",
                f"standard_deviation = {standard_deviation!r}",
            ]
        )
        round_setup = [
            "    expected = means",
            "    potential_rewards = reward_rng.normal(means, standard_deviation)",
        ]
    elif environment_kind == "changing-bernoulli":
        phases = environment.get(
            "phases",
            (
                {"start": 0, "probabilities": (0.75, 0.5, 0.25)},
                {"start": max(1, horizon // 3), "probabilities": (0.2, 0.75, 0.45)},
                {
                    "start": max(2, (2 * horizon) // 3),
                    "probabilities": (0.45, 0.2, 0.8),
                },
            ),
        )
        serializable = [
            {
                "start": int(phase["start"]),
                "probabilities": list(phase["probabilities"]),
            }
            for phase in phases
        ]
        setup.append(f"phases = {serializable!r}")
        round_setup = [
            "    phase = max((item for item in phases if item['start'] <= step), key=lambda item: item['start'])",
            "    expected = np.array(phase['probabilities'], dtype=float)",
            "    potential_rewards = (reward_rng.random(3) < expected).astype(float)",
        ]
    elif environment_kind == "adversarial":
        configured_rewards = environment.get("rewards")
        setup.append(f"configured_rewards = {configured_rewards!r}")
        round_setup = [
            "    if configured_rewards is None:",
            f"        leader = (step // 3 + {seed!r} % 3) % 3",
            "        potential_rewards = np.full(3, 0.1, dtype=float)",
            "        potential_rewards[leader] = 1.0",
            "        potential_rewards[(leader + 1) % 3] = 0.4",
            "    else:",
            "        potential_rewards = np.array(configured_rewards[step], dtype=float)",
            "    expected = potential_rewards.copy()",
        ]
    else:
        theta = environment.get(
            "theta",
            (
                (0.1, -1.2, 0.2, -0.8),
                (0.0, 1.0, 0.3, 1.0),
                (0.2, 0.0, -1.1, 0.2),
            ),
        )
        setup.append(f"theta = np.array({[list(row) for row in theta]!r}, dtype=float)")
        if environment_kind == "contextual-linear":
            setup.append(
                f"standard_deviation = {environment.get('standardDeviation', 0.2)!r}"
            )
            reward_line = "    potential_rewards = reward_rng.normal(expected, standard_deviation)"
        else:
            reward_line = "    potential_rewards = (reward_rng.random(3) < expected).astype(float)"
        round_setup = [
            "    feature = np.concatenate((np.ones(1), context_rng.choice(np.array([-1.0, 1.0]), size=3)))",
            "    context = np.repeat(feature[np.newaxis, :], 3, axis=0)",
            "    expected = theta @ feature",
        ]
        if environment_kind == "contextual-logistic":
            round_setup.append("    expected = 1.0 / (1.0 + np.exp(-expected))")
        round_setup.append(reward_line)

    contextual = environment_kind.startswith("contextual-")
    select = (
        "    action = policy.select_action(context=context, rng=action_rng)"
        if contextual
        else "    action = policy.select_action(rng=action_rng)"
    )
    update = (
        "    policy.update(action=action, reward=reward, context=context)"
        if contextual
        else "    policy.update(action=action, reward=reward)"
    )
    recommendation = (
        "result['recommendation'] = int(policy.recommend_action())"
        if objective == "best-arm"
        else ""
    )
    lines = [
        "import numpy as np",
        f"from pymab.policies import {class_name}",
        "",
        f"policy = {class_name}({arguments})",
        f"action_rng = np.random.default_rng(np.random.SeedSequence({action_seed}))",
        f"reward_rng = np.random.default_rng(np.random.SeedSequence({reward_seed}))",
        f"context_rng = np.random.default_rng(np.random.SeedSequence({context_seed}))",
        *setup,
        "total_reward = 0.0",
        "cumulative_regret = 0.0",
        f"for step in range({horizon}):",
        *round_setup,
        select,
        "    reward = float(potential_rewards[action])",
        update,
        "    total_reward += reward",
        "    cumulative_regret += float(expected.max() - expected[action])",
        "result = {'totalReward': total_reward, 'cumulativeExpectedRegret': cumulative_regret}",
    ]
    if recommendation:
        lines.append(recommendation)
    lines.append("print(result)")
    return "\n".join(lines) + "\n"
