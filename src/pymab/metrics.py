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
    """Expected regret per replicate, step, and policy."""

    return result.regret


def cumulative_regret_by_step(result: SimulationResult) -> np.ndarray:
    """Cumulative expected regret averaged across replicates."""

    return result.cumulative_regret


def optimal_action_rate_by_step(result: SimulationResult) -> np.ndarray:
    """Share of replicates selecting an optimal action at each step."""

    return result.optimal_action_rate_by_step


def simple_regret_by_step(result: SimulationResult) -> np.ndarray:
    """Mean recommendation regret per step and policy."""

    return np.asarray(np.mean(result.simple_regret, axis=0), dtype=float)


def best_arm_identification_rate_by_step(result: SimulationResult) -> np.ndarray:
    """Fraction of replicates recommending any optimal arm."""

    return np.asarray(np.mean(result.recommendation_is_optimal, axis=0), dtype=float)


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute a 1D moving average."""

    if window <= 0:
        raise ValueError("window must be positive")
    if data.ndim != 1:
        raise ValueError("data must be 1D")
    if window > data.size:
        raise ValueError("window cannot be larger than data")
    return np.convolve(data, np.ones(window) / window, mode="valid")


__all__ = [
    "average_reward_by_step",
    "best_arm_identification_rate_by_step",
    "cumulative_regret_by_step",
    "cumulative_reward_by_step",
    "expected_regret_by_step",
    "moving_average",
    "optimal_action_rate_by_step",
    "simple_regret_by_step",
]
