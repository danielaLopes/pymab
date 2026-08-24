"""Generate complete public-PyMAB examples for the browser Lab."""

from __future__ import annotations

from pymab._random import stable_seed


def epsilon_example(*, seed: int, epsilon: float, horizon: int) -> str:
    """Return a standalone epsilon-greedy reproduction."""

    action_seed = stable_seed(seed, "epsilon-greedy", "lesson", "action")
    reward_seed = stable_seed(seed, "epsilon-greedy", "lesson", "reward")
    return f"""import numpy as np
from pymab.policies import EpsilonGreedyPolicy

means = np.array([0.25, 0.50, 0.75])
policy = EpsilonGreedyPolicy(n_arms=3, epsilon={epsilon!r})
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


def linucb_example(*, seed: int, alpha: float, l2: float, horizon: int) -> str:
    """Return a standalone LinUCB reproduction."""

    context_seed = stable_seed(seed, "arcade", 1, "context")
    action_seed = stable_seed(seed, "arcade", 1, "action")
    reward_seed = stable_seed(seed, "arcade", 1, "reward")
    return f"""import numpy as np
from pymab.policies import LinUCBPolicy

theta = np.array([[0.1, -1.2, 0.2, -0.8], [0.0, 1.0, 0.3, 1.0], [0.2, 0.0, -1.1, 0.2]])
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
