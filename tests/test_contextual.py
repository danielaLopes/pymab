from __future__ import annotations

import numpy as np
import pytest

from pymab.distributions import BernoulliReward, GaussianReward
from pymab.environments import (
    BanditEnvironment,
    ContextProvider,
    LinearContextualEnvironment,
    LogisticContextualEnvironment,
)
from pymab.policies import (
    LinearEpsilonGreedyPolicy,
    LinearThompsonSamplingPolicy,
    LinUCBPolicy,
    LogisticContextualBanditPolicy,
    UCBPolicy,
)
from pymab.simulation import Experiment, ExperimentConfig


def fixed_context(rng: np.random.Generator) -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, 1.0]])


def test_context_shape_forms_and_validation() -> None:
    shared = LinearContextualEnvironment(
        theta=np.ones((2, 2)),
        context_provider=lambda rng: np.array([1.0, 2.0]),
    )
    assert shared.context(np.random.default_rng(1)).shape == (2, 2)
    bad = LinearContextualEnvironment(
        theta=np.ones((2, 2)), context_provider=lambda rng: np.ones(3)
    )
    with pytest.raises(ValueError, match="context"):
        bad.context(np.random.default_rng(1))
    nonfinite = LinearContextualEnvironment(
        theta=np.ones((2, 2)), context_provider=lambda rng: np.array([1.0, np.nan])
    )
    with pytest.raises(ValueError, match="finite"):
        nonfinite.context(np.random.default_rng(1))


def test_linear_environment_rejects_binary_model() -> None:
    with pytest.raises(ValueError, match="Logistic"):
        LinearContextualEnvironment(
            theta=np.ones((2, 2)),
            context_provider=fixed_context,
            reward_model=BernoulliReward(),
        )


def test_logistic_environment_outputs_valid_probabilities() -> None:
    environment = LogisticContextualEnvironment(
        theta=np.array([[20.0, 0.0], [0.0, -20.0]]),
        context_provider=fixed_context,
        reward_model=BernoulliReward(),
    )
    context = environment.context(np.random.default_rng(1))
    means = environment.expected_rewards(context)
    assert np.all((means >= 0) & (means <= 1))
    assert set(
        environment.sample_rewards(context=context, rng=np.random.default_rng(1))
    ) <= {
        0.0,
        1.0,
    }
    clone = environment.clone()
    clone.theta[0, 0] = 0
    assert environment.theta[0, 0] == 20


def test_logistic_environment_requires_binary_model() -> None:
    with pytest.raises(ValueError, match="binary"):
        LogisticContextualEnvironment(
            theta=np.ones((2, 2)),
            context_provider=fixed_context,
            reward_model=GaussianReward(),
        )


def test_contextual_policies_update_and_recommend() -> None:
    context = fixed_context(np.random.default_rng(1))
    epsilon = LinearEpsilonGreedyPolicy(
        n_arms=2, n_features=2, epsilon=0.0, learning_rate=0.5
    )
    epsilon.update(action=1, reward=2.0, context=context)
    assert epsilon.recommend_action(context=context) == 1
    linucb = LinUCBPolicy(n_arms=2, n_features=2)
    linucb.update(action=0, reward=1.0, context=context)
    assert linucb.upper_confidence_bounds(context).shape == (2,)
    assert linucb.recommend_action(context=context) == 0
    thompson = LinearThompsonSamplingPolicy(n_arms=2, n_features=2)
    thompson.update(action=0, reward=1.0, context=context)
    assert thompson.select_action(context=context, rng=np.random.default_rng(1)) in {
        0,
        1,
    }
    assert thompson.recommend_action(context=context) == 0


def test_logistic_policy_requires_binary_reward() -> None:
    policy = LogisticContextualBanditPolicy(n_arms=2, n_features=2)
    with pytest.raises(ValueError, match="reward"):
        policy.update(
            action=0, reward=2.0, context=fixed_context(np.random.default_rng(1))
        )


def test_contextual_experiment_runs_and_is_order_invariant() -> None:
    environment = LogisticContextualEnvironment(
        theta=np.array([[2.0, 0.0], [0.0, 2.0]]),
        context_provider=fixed_context,
        reward_model=BernoulliReward(),
    )
    config = ExperimentConfig(horizon=8, n_replicates=3, seed=3)
    alone = Experiment(
        environment=environment,
        policies={"logistic": LogisticContextualBanditPolicy(n_arms=2, n_features=2)},
        config=config,
    ).run()
    together = Experiment(
        environment=environment,
        policies={
            "linucb": LinUCBPolicy(n_arms=2, n_features=2),
            "logistic": LogisticContextualBanditPolicy(n_arms=2, n_features=2),
        },
        config=config,
    ).run()
    np.testing.assert_array_equal(alone.actions[:, :, 0], together.actions[:, :, 1])
    np.testing.assert_array_equal(alone.arm_means, together.arm_means)


def test_stateful_context_provider_is_cloned_per_replicate() -> None:
    class StatefulProvider(ContextProvider):
        clones: list[StatefulProvider] = []

        def __init__(self) -> None:
            self.calls = 0

        def sample(self, rng):
            self.calls += 1
            return np.eye(2)

        def clone(self):
            clone = type(self)()
            type(self).clones.append(clone)
            return clone

    provider = StatefulProvider()
    Experiment(
        environment=LinearContextualEnvironment(
            theta=np.eye(2),
            context_provider=provider,
        ),
        policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2)},
        config=ExperimentConfig(horizon=3, n_replicates=2, seed=2),
    ).run()
    assert provider.calls == 0
    assert [clone.calls for clone in StatefulProvider.clones] == [3, 3]


def test_context_recording_and_provenance_are_explicit() -> None:
    result = Experiment(
        environment=LinearContextualEnvironment(
            theta=np.eye(2),
            context_provider=fixed_context,
        ),
        policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2, alpha=0.5)},
        config=ExperimentConfig(
            horizon=2,
            n_replicates=2,
            seed=9,
            record_contexts=True,
        ),
    ).run()
    assert result.contexts is not None
    assert result.contexts.shape == (2, 2, 2, 2)
    assert result.context_digest is not None
    provenance = result.provenance.to_dict()
    assert provenance["python_version"] != "unknown"
    assert provenance["numpy_version"] == np.__version__
    assert "LinearContextualEnvironment" in provenance["environment"]["class"]
    assert provenance["policies"]["linucb"]["parameters"]["alpha"] == 0.5


def test_experiment_rejects_context_and_feature_mismatches() -> None:
    config = ExperimentConfig(horizon=1, n_replicates=1, seed=1)
    contextual = LinearContextualEnvironment(
        theta=np.ones((2, 2)), context_provider=fixed_context
    )
    with pytest.raises(TypeError, match="contextual"):
        Experiment(
            environment=contextual,
            policies={"ucb": UCBPolicy(n_arms=2)},
            config=config,
        )
    with pytest.raises(TypeError, match="n_features"):
        Experiment(
            environment=contextual,
            policies={"linucb": LinUCBPolicy(n_arms=2, n_features=3)},
            config=config,
        )
    with pytest.raises(TypeError, match="contextual"):
        Experiment(
            environment=BanditEnvironment(means=np.array([0.0, 1.0])),
            policies={"linucb": LinUCBPolicy(n_arms=2, n_features=2)},
            config=config,
        )
