import numpy as np
import pytest
from pymab.policies.greedy import GreedyPolicy
from pymab.game import Game


def test_game_initialization():
    policy = GreedyPolicy(np.array([0.1, 0.2, 0.3]), optimistic_initilization=1)
    game = Game(n_episodes=10, n_steps=100, policy=policy, n_bandits=3)
    assert game.n_episodes == 10
    assert game.n_steps == 100
    assert game.n_bandits == 3
    assert isinstance(game.policy, GreedyPolicy)


def test_game_loop():
    policy = GreedyPolicy(np.array([0.1, 0.2, 0.3]), optimistic_initilization=1)
    game = Game(n_episodes=1, n_steps=5, policy=policy, n_bandits=3)
    game.game_loop()
    assert game.rewards_by_policy.shape == (1, 5)  # Check the shape of the rewards matrix for consistency
