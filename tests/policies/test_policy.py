import numpy as np
import pytest
from pymab.policies.greedy import GreedyPolicy


def test_greedy_initialization():
    Q_values = np.array([0.1, 0.2, 0.3])
    policy = GreedyPolicy(Q_values, optimistic_initilization=1)
    assert policy.optimistic_initilization == 1
    assert np.array_equal(policy.Q_values, Q_values)
    assert (
        policy.select_action() == 2
    )  # Should initially select the last one due to optimistic initialization


def test_greedy_selection():
    Q_values = np.array([0.1, 0.2, 0.3])
    policy = GreedyPolicy(Q_values, optimistic_initilization=1)
    # Assuming the select_action function updates the estimated rewards based on some logic
    for _ in range(10):
        policy.select_action()  # this will keep selecting the best estimated action
    assert policy.select_action() == 2  # Still should select the highest Q value
