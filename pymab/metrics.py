"""Metrics computed from simulation results."""

from __future__ import annotations

import numpy as np

from pymab.simulation import SimulationResult


def average_reward_by_step(result: SimulationResult) -> np.ndarray:
    """Mean realized reward per step and policy."""

    return result.average_reward_by_step


def cumulative_reward_by_step(result: SimulationResult) -> np.ndarray:
    """Cumulative mean realized reward per step and policy."""

    return result.cumulative_reward_by_step


def expected_regret_by_step(result: SimulationResult) -> np.ndarray:
    """Expected regret per episode, step, and policy."""

    return result.regret


def cumulative_regret_by_step(result: SimulationResult) -> np.ndarray:
    """Cumulative expected regret averaged across episodes."""

    return result.cumulative_regret


def optimal_action_rate_by_step(result: SimulationResult) -> np.ndarray:
    """Share of episodes selecting the optimal action at each step."""

    return result.optimal_action_rate_by_step


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute a 1D moving average."""

    if window <= 0:
        raise ValueError("window must be positive")
    if data.ndim != 1:
        raise ValueError("data must be 1D")
    if window > data.size:
        raise ValueError("window cannot be larger than data")
    return np.convolve(data, np.ones(window) / window, mode="valid")
