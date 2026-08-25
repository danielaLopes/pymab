from __future__ import annotations

import numpy as np
import pytest

from pymab import (
    BanditEnvironment,
    CompatibilityError,
    Experiment,
    ExperimentConfig,
    UniformReward,
    ValidationError,
)
from pymab.environments import FixedContextProvider, LinearContextualEnvironment
from pymab.policies import GreedyPolicy, LinUCBPolicy, RandomPolicy, UCBPolicy
from pymab.policies.policy import Policy
from pymab.results import SimulationResult


def _classic(*, backend: str, policies: dict[str, Policy] | None = None) -> Experiment:
    return Experiment(
        environment=BanditEnvironment(
            means=np.array([0.0, 0.5, 1.0]),
            reward_model=UniformReward(half_width=0.25),
        ),
        policies=(
            {"random": RandomPolicy(n_arms=3), "ucb": UCBPolicy(n_arms=3)}
            if policies is None
            else policies
        ),
        config=ExperimentConfig(
            horizon=12,
            n_replicates=3,
            seed=7,
            backend=backend,  # type: ignore[arg-type]
        ),
    )


def test_explicit_native_experiment_preserves_result_contract() -> None:
    result = _classic(backend="rust").run()
    assert result.rewards.shape == (3, 12, 2)
    assert result.actions.shape == (3, 12, 2)
    assert result.expected_rewards.shape == (3, 12, 2)
    assert result.arm_means.shape == (3, 12, 3)
    assert result.optimal_mask.shape == (3, 12, 3)
    assert result.recommendations.shape == (3, 12, 2)
    assert result.actions.dtype == np.int64
    assert result.optimal_mask.dtype == np.bool_
    assert result.provenance.backend == "rust"
    assert result.provenance.rng_scheme.startswith("pymab-rust-")


def test_backend_independent_deterministic_single_arm_trace_matches_exactly() -> None:
    def run(backend: str) -> SimulationResult:
        return Experiment(
            environment=BanditEnvironment(
                means=np.array([0.5]),
                reward_model=UniformReward(half_width=0.0),
            ),
            policies={"greedy": GreedyPolicy(n_arms=1)},
            config=ExperimentConfig(
                horizon=5,
                n_replicates=2,
                seed=3,
                backend=backend,  # type: ignore[arg-type]
            ),
        ).run()

    python = run("python")
    rust = run("rust")
    for left, right in (
        (python.rewards, rust.rewards),
        (python.actions, rust.actions),
        (python.expected_rewards, rust.expected_rewards),
        (python.arm_means, rust.arm_means),
        (python.optimal_mask, rust.optimal_mask),
        (python.recommendations, rust.recommendations),
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_each_backend_is_independently_reproducible(backend: str) -> None:
    left = _classic(backend=backend).run()
    right = _classic(backend=backend).run()
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.rewards, right.rewards)


def test_native_policy_streams_are_order_and_added_policy_independent() -> None:
    alone = _classic(
        backend="rust",
        policies={"random": RandomPolicy(n_arms=3)},
    ).run()
    together = _classic(
        backend="rust",
        policies={"ucb": UCBPolicy(n_arms=3), "random": RandomPolicy(n_arms=3)},
    ).run()
    np.testing.assert_array_equal(alone.actions[:, :, 0], together.actions[:, :, 1])
    np.testing.assert_array_equal(alone.rewards[:, :, 0], together.rewards[:, :, 1])


def test_native_context_recording_and_digest_are_stable() -> None:
    experiment = Experiment(
        environment=LinearContextualEnvironment(
            theta=np.eye(2),
            context_provider=FixedContextProvider(np.eye(2)),
            reward_model=UniformReward(half_width=0.0),
        ),
        policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2)},
        config=ExperimentConfig(
            horizon=4,
            n_replicates=2,
            seed=5,
            record_contexts=True,
            backend="rust",
        ),
    )
    left = experiment.run()
    right = experiment.run()
    assert left.contexts is not None
    assert left.contexts.shape == (2, 4, 2, 2)
    assert left.context_digest == right.context_digest
    assert left.context_digest is not None and len(left.context_digest) == 64


def test_auto_falls_back_for_callable_context_and_rust_reports_every_issue() -> None:
    def context(_: np.random.Generator) -> np.ndarray:
        return np.eye(2)

    environment = LinearContextualEnvironment(
        theta=np.eye(2),
        context_provider=context,
        reward_model=UniformReward(half_width=0.0),
    )
    auto = Experiment(
        environment=environment,
        policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2)},
        config=ExperimentConfig(horizon=2, n_replicates=1, seed=1),
    )
    report = auto.backend_compatibility()
    assert not report.compatible
    assert "requires Python callbacks" in report.message()
    assert auto.run().provenance.backend == "python"

    required = Experiment(
        environment=environment,
        policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2)},
        config=ExperimentConfig(
            horizon=2,
            n_replicates=1,
            seed=1,
            backend="rust",
        ),
    )
    with pytest.raises(CompatibilityError, match="requires Python callbacks"):
        required.run()


def test_custom_policy_forces_reference_runner() -> None:
    class FirstArmPolicy(Policy):
        def select_action(self, *, rng: np.random.Generator) -> int:
            del rng
            return 0

        def update(self, *, action: int, reward: float) -> None:
            self._validate_action(action)
            del reward

        def reset(self) -> None:
            pass

        def recommend_action(self) -> int:
            return 0

    experiment = Experiment(
        environment=BanditEnvironment(means=np.array([0.0, 1.0])),
        policies={"custom": FirstArmPolicy(n_arms=2)},
        config=ExperimentConfig(horizon=2, n_replicates=1, seed=1),
    )
    assert "not a native built-in" in experiment.backend_compatibility().message()
    assert experiment.run().provenance.backend == "python"


def test_backend_configuration_is_validated() -> None:
    with pytest.raises(ValidationError, match="backend"):
        ExperimentConfig(
            horizon=1,
            n_replicates=1,
            seed=1,
            backend="gpu",  # type: ignore[arg-type]
        )
