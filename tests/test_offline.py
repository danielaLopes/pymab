from __future__ import annotations

import numpy as np
import pytest

from pymab.errors import OverlapError
from pymab.offline import (
    EstimateMethod,
    EstimatorConfig,
    LoggedBanditDataset,
    LoggingScheme,
    OverlapStatus,
    PolicyValueEstimator,
    ResamplingUnit,
    estimate_policy_value,
    sequential_replay,
)
from pymab.policies import LogisticContextualBanditPolicy, RandomPolicy
from pymab.policies.policy import Policy
from pymab.statistics import BootstrapConfig, ConfidenceMethod


class FixedTarget:
    def __init__(self, probabilities) -> None:
        self._probabilities = np.asarray(probabilities, dtype=float)

    def probabilities(self, context):
        return self._probabilities


class PerfectCrossFittedModel:
    def predict_event(self, event_index, context):
        return np.array([0.0, 1.0])


class BatchFixedTarget:
    def __init__(self, probabilities) -> None:
        self._probabilities = np.asarray(probabilities, dtype=float)

    def probabilities_batch(self, contexts, *, n_events):
        return np.repeat(self._probabilities[np.newaxis, :], n_events, axis=0)


class FixedActionPolicy(Policy):
    def __init__(self, n_arms: int, action: int) -> None:
        self.action = action
        self.updates = 0
        super().__init__(n_arms=n_arms)

    def select_action(self, *, rng: np.random.Generator) -> int:
        return self.action

    def update(self, *, action: int, reward: float) -> None:
        self.updates += 1

    def reset(self) -> None:
        self.updates = 0

    def recommend_action(self) -> int:
        return self.action


def _dataset() -> LoggedBanditDataset:
    return LoggedBanditDataset(
        actions=np.array([0, 1, 0, 1]),
        rewards=np.array([0.0, 1.0, 0.0, 1.0]),
        propensities=np.full(4, 0.5),
        n_arms=2,
    )


def _config(
    method: EstimateMethod | str = EstimateMethod.IPS,
    *,
    weight_clip: float | None = None,
    n_resamples: int = 100,
    seed: int = 0,
    max_chunk_elements: int = 1_000_000,
    confidence_level: float = 0.95,
) -> EstimatorConfig:
    return EstimatorConfig(
        method=method,
        weight_clip=weight_clip,
        bootstrap=BootstrapConfig(
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed,
            max_chunk_elements=max_chunk_elements,
        ),
    )


@pytest.mark.parametrize("method", ["ips", "snips", "dr"])
def test_offline_estimators_match_known_value(method) -> None:
    kwargs = {"reward_model": PerfectCrossFittedModel()} if method == "dr" else {}
    estimate = estimate_policy_value(
        _dataset(),
        FixedTarget([0.25, 0.75]),
        config=_config(method, n_resamples=200, seed=1),
        **kwargs,
    )
    assert estimate.estimate == pytest.approx(0.75)
    assert estimate.effective_sample_size == pytest.approx(3.2)
    assert estimate.ci_lower is not None
    assert estimate.max_weight == 1.5
    assert estimate.confidence_method is ConfidenceMethod.PERCENTILE_BOOTSTRAP


def test_weight_clipping_is_reported() -> None:
    estimate = estimate_policy_value(
        _dataset(),
        FixedTarget([0.25, 0.75]),
        config=_config(weight_clip=1.0, n_resamples=20),
    )
    assert estimate.clipped_fraction == 0.5
    assert estimate.max_weight == 1.0
    assert estimate.weights.raw_max_weight == 1.5
    assert estimate.weights.raw_effective_sample_size == pytest.approx(3.2)


def test_zero_overlap_is_explicit() -> None:
    dataset = LoggedBanditDataset(
        actions=np.zeros(4, dtype=int),
        rewards=np.zeros(4),
        propensities=np.ones(4),
        n_arms=2,
    )
    with pytest.raises(OverlapError, match="zero support"):
        estimate_policy_value(
            dataset,
            FixedTarget([0.0, 1.0]),
            config=_config("ips", n_resamples=10),
        )
    estimate = estimate_policy_value(
        dataset,
        FixedTarget([0.0, 1.0]),
        config=_config("dr", n_resamples=10),
        reward_model=PerfectCrossFittedModel(),
    )
    assert estimate.estimate == 1.0
    assert estimate.overlap_status is OverlapStatus.MODEL_ONLY
    assert estimate.weights.raw_effective_sample_size == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actions": [], "rewards": [], "propensities": [], "n_arms": 2},
        {"actions": [2], "rewards": [1], "propensities": [1], "n_arms": 2},
        {"actions": [0], "rewards": [np.nan], "propensities": [1], "n_arms": 2},
        {"actions": [0], "rewards": [1], "propensities": [0], "n_arms": 2},
        {"actions": [0], "rewards": [1], "propensities": [2], "n_arms": 2},
        {"actions": [0], "rewards": [1], "propensities": [1], "n_arms": 0},
        {
            "actions": [0],
            "rewards": [1],
            "propensities": [1],
            "n_arms": 2,
            "contexts": [[np.nan]],
        },
    ],
)
def test_logged_dataset_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        LoggedBanditDataset(**kwargs)


@pytest.mark.parametrize("actions", [[0.0], ["0"], [True]])
def test_logged_actions_reject_lossy_integer_coercion(actions) -> None:
    with pytest.raises(ValueError, match="integers"):
        LoggedBanditDataset(
            actions=actions,
            rewards=[1.0],
            propensities=[0.5],
            n_arms=2,
        )


def test_logged_dataset_is_immutable_and_contextual() -> None:
    dataset = LoggedBanditDataset(
        actions=np.array([0]),
        rewards=np.array([1.0]),
        propensities=np.array([0.5]),
        n_arms=2,
        contexts=np.array([[1.0, 2.0]]),
    )
    assert dataset.n_events == 1
    np.testing.assert_array_equal(dataset.context_at(0), [1.0, 2.0])
    with pytest.raises(ValueError):
        dataset.actions[0] = 1


@pytest.mark.parametrize(
    "target",
    [FixedTarget([1.0]), FixedTarget([-1.0, 2.0]), FixedTarget([0.2, 0.2])],
)
def test_target_probability_validation(target) -> None:
    with pytest.raises(ValueError):
        estimate_policy_value(_dataset(), target, config=_config(n_resamples=10))


def test_estimator_argument_validation() -> None:
    with pytest.raises(ValueError, match="method"):
        _config("x")
    with pytest.raises(ValueError, match="cross-fitted"):
        estimate_policy_value(_dataset(), FixedTarget([0.5, 0.5]), config=_config("dr"))
    with pytest.raises(ValueError, match="weight_clip"):
        _config(weight_clip=0)
    with pytest.raises(ValueError, match="confidence"):
        _config(confidence_level=1)
    with pytest.raises(ValueError, match="n_resamples"):
        _config(n_resamples=0)
    with pytest.raises(ValueError, match="max_chunk"):
        _config(max_chunk_elements=0)
    with pytest.raises(TypeError, match="BootstrapConfig"):
        EstimatorConfig(bootstrap=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="EstimatorConfig"):
        PolicyValueEstimator(object())  # type: ignore[arg-type]


def test_reward_model_prediction_validation() -> None:
    class BadModel:
        def predict_event(self, event_index, context):
            return np.array([np.nan])

    with pytest.raises(ValueError, match="prediction"):
        estimate_policy_value(
            _dataset(),
            FixedTarget([0.5, 0.5]),
            config=_config("dr", n_resamples=10),
            reward_model=BadModel(),
        )

    class WrongShapeModel:
        def predict_event(self, event_index, context):
            return np.array([0.0])

    with pytest.raises(ValueError, match="one prediction per arm"):
        estimate_policy_value(
            _dataset(),
            FixedTarget([0.5, 0.5]),
            config=_config("dr", n_resamples=10),
            reward_model=WrongShapeModel(),
        )


def test_estimators_recover_known_logged_policy_value() -> None:
    rng = np.random.default_rng(42)
    actions = (rng.random(20_000) >= 0.8).astype(np.int64)
    propensities = np.where(actions == 0, 0.8, 0.2)
    dataset = LoggedBanditDataset(
        actions=actions,
        rewards=actions.astype(float),
        propensities=propensities,
        n_arms=2,
    )
    for method in ("ips", "snips"):
        estimate = estimate_policy_value(
            dataset,
            FixedTarget([0.3, 0.7]),
            config=_config(method, n_resamples=200, seed=3),
        )
        assert estimate.estimate == pytest.approx(0.7, abs=0.025)
        assert estimate.resampling_unit is ResamplingUnit.EVENT

    dr = estimate_policy_value(
        dataset,
        FixedTarget([0.3, 0.7]),
        config=_config("dr", n_resamples=50),
        reward_model=PerfectCrossFittedModel(),
    )
    assert dr.estimate == pytest.approx(0.7)


@pytest.mark.parametrize(
    "method",
    [EstimateMethod.IPS, EstimateMethod.SNIPS, EstimateMethod.DOUBLY_ROBUST],
)
def test_offline_bootstrap_has_reference_monte_carlo_coverage(
    method: EstimateMethod,
) -> None:
    rng = np.random.default_rng(2026)
    estimates: list[float] = []
    covered = 0
    for trial in range(30):
        actions = (rng.random(400) >= 0.8).astype(np.int64)
        propensities = np.where(actions == 0, 0.8, 0.2)
        estimate = estimate_policy_value(
            LoggedBanditDataset(
                actions=actions,
                rewards=actions.astype(float),
                propensities=propensities,
                n_arms=2,
            ),
            FixedTarget([0.3, 0.7]),
            config=_config(method, n_resamples=200, seed=trial),
            reward_model=(
                PerfectCrossFittedModel()
                if method is EstimateMethod.DOUBLY_ROBUST
                else None
            ),
        )
        estimates.append(estimate.estimate)
        assert estimate.ci_lower is not None
        assert estimate.ci_upper is not None
        covered += int(estimate.ci_lower <= 0.7 <= estimate.ci_upper)
    assert np.mean(estimates) == pytest.approx(0.7, abs=0.03)
    assert covered >= 26


@pytest.mark.parametrize("method", ["ips", "snips", "dr"])
def test_vectorized_target_policy_matches_event_interface(method: str) -> None:
    kwargs = {"reward_model": PerfectCrossFittedModel()} if method == "dr" else {}
    event = estimate_policy_value(
        _dataset(),
        FixedTarget([0.25, 0.75]),
        config=_config(method, n_resamples=50, seed=5),
        **kwargs,
    )
    batch = estimate_policy_value(
        _dataset(),
        BatchFixedTarget([0.25, 0.75]),
        config=_config(method, n_resamples=50, seed=5),
        **kwargs,
    )
    assert batch == event


def test_vectorized_target_policy_validates_matrix_contract() -> None:
    with pytest.raises(ValueError, match="shape"):
        estimate_policy_value(
            _dataset(),
            BatchFixedTarget([1.0]),
            config=_config(n_resamples=10),
        )
    with pytest.raises(ValueError, match="non-negative"):
        estimate_policy_value(
            _dataset(),
            BatchFixedTarget([-0.1, 1.1]),
            config=_config(n_resamples=10),
        )
    with pytest.raises(ValueError, match="sum to one"):
        estimate_policy_value(
            _dataset(),
            BatchFixedTarget([0.2, 0.2]),
            config=_config(n_resamples=10),
        )


def test_vectorized_dr_validates_reward_model_shape() -> None:
    class WrongShapeModel:
        def predict_event(self, event_index, context):
            return np.array([0.0])

    with pytest.raises(ValueError, match="one prediction per arm"):
        estimate_policy_value(
            _dataset(),
            BatchFixedTarget([0.5, 0.5]),
            config=_config("dr", n_resamples=10),
            reward_model=WrongShapeModel(),
        )


def test_extreme_importance_weights_report_weak_overlap() -> None:
    estimate = estimate_policy_value(
        LoggedBanditDataset(
            actions=np.array([0, 0]),
            rewards=np.array([0.0, 1.0]),
            propensities=np.array([0.001, 0.001]),
            n_arms=2,
        ),
        FixedTarget([1.0, 0.0]),
        config=_config(n_resamples=10),
    )
    assert estimate.overlap_status is OverlapStatus.WEAK


def test_overflowing_importance_weight_is_rejected() -> None:
    dataset = LoggedBanditDataset(
        actions=np.array([0, 0]),
        rewards=np.array([0.0, 1.0]),
        propensities=np.full(2, np.nextafter(0.0, 1.0)),
        n_arms=2,
    )
    with np.errstate(over="ignore"), pytest.raises(OverlapError, match="overflowed"):
        estimate_policy_value(
            dataset,
            FixedTarget([1.0, 0.0]),
            config=_config(n_resamples=10),
        )


def test_cluster_ids_activate_cluster_bootstrap() -> None:
    base = _dataset()
    dataset = LoggedBanditDataset(
        actions=base.actions,
        rewards=base.rewards,
        propensities=base.propensities,
        n_arms=base.n_arms,
        clusters=["user-a", "user-a", "user-b", "user-b"],
    )
    estimate = estimate_policy_value(
        dataset,
        FixedTarget([0.25, 0.75]),
        config=_config(n_resamples=100),
    )
    assert estimate.resampling_unit is ResamplingUnit.CLUSTER


def test_sequential_replay_non_contextual_and_contextual() -> None:
    replay = sequential_replay(
        RandomPolicy(n_arms=1),
        logged_actions=[0, 0],
        logged_rewards=[1.0, 0.0],
        logging_scheme="uniform",
        seed=1,
    )
    assert replay.n_events == replay.n_accepted == 2
    assert replay.acceptance_rate == 1
    assert replay.average_reward == 0.5
    assert not replay.selected_actions.flags.writeable
    assert not replay.accepted_actions.flags.writeable
    assert not replay.accepted_rewards.flags.writeable
    contextual = LogisticContextualBanditPolicy(n_arms=1, n_features=2)
    replay = sequential_replay(
        contextual,
        logged_actions=[0, 0],
        logged_rewards=[1.0, 0.0],
        contexts=np.array([[[1.0, 0.0]], [[0.0, 1.0]]]),
        logging_scheme="uniform",
        seed=1,
        clone_policy=False,
    )
    assert replay.n_accepted == 2
    assert not np.allclose(contextual.theta, 0)


def test_sequential_replay_validation() -> None:
    policy = RandomPolicy(n_arms=2)
    with pytest.raises(ValueError):
        sequential_replay(
            policy,
            logged_actions=[],
            logged_rewards=[],
            logging_scheme="uniform",
            seed=1,
        )
    with pytest.raises(ValueError):
        sequential_replay(
            policy,
            logged_actions=[2],
            logged_rewards=[1],
            logging_scheme="uniform",
            seed=1,
        )
    with pytest.raises(ValueError):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[np.nan],
            logging_scheme="uniform",
            seed=1,
        )
    contextual = LogisticContextualBanditPolicy(n_arms=1, n_features=2)
    with pytest.raises((TypeError, ValueError), match="contexts"):
        sequential_replay(
            contextual,
            logged_actions=[0],
            logged_rewards=[1],
            logging_scheme="uniform",
            seed=1,
        )


def test_replay_preserves_warm_state_when_requested() -> None:
    policy = FixedActionPolicy(n_arms=1, action=0)
    policy.updates = 5
    replay = sequential_replay(
        policy,
        logged_actions=[0],
        logged_rewards=[1.0],
        logging_scheme="uniform",
        clone_policy=False,
        reset_policy=False,
    )
    assert replay.n_accepted == 1
    assert policy.updates == 6

    policy.updates = 5
    sequential_replay(
        policy,
        logged_actions=[0],
        logged_rewards=[1.0],
        logging_scheme="uniform",
        clone_policy=False,
        reset_policy=True,
    )
    assert policy.updates == 1

    policy.updates = 5
    sequential_replay(
        policy,
        logged_actions=[0],
        logged_rewards=[1.0],
        logging_scheme="uniform",
        clone_policy=True,
        reset_policy=False,
    )
    assert policy.updates == 5


def test_nonuniform_replay_requires_propensities_and_is_reproducible() -> None:
    policy = FixedActionPolicy(n_arms=2, action=0)
    with pytest.raises(ValueError, match="requires propensities"):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="nonuniform",
        )
    kwargs = {
        "logged_actions": [0] * 100,
        "logged_rewards": [1.0] * 100,
        "logging_scheme": "nonuniform",
        "propensities": [0.25, 0.5] * 50,
        "seed": 12,
    }
    left = sequential_replay(policy, **kwargs)
    right = sequential_replay(policy, **kwargs)
    assert left.equals(right)
    assert left.logging_scheme is LoggingScheme.NONUNIFORM
    assert left.acceptance_scale == 0.25
    assert 50 < left.n_accepted < 100


def test_replay_rejects_invalid_policy_actions() -> None:
    with pytest.raises(ValueError, match=r"outside.*event 0"):
        sequential_replay(
            FixedActionPolicy(n_arms=2, action=99),
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="uniform",
        )


def test_replay_logging_contract_validation() -> None:
    policy = FixedActionPolicy(n_arms=2, action=1)
    no_matches = sequential_replay(
        policy,
        logged_actions=[0, 0],
        logged_rewards=[1.0, 1.0],
        logging_scheme="uniform",
    )
    assert no_matches.n_accepted == 0
    assert no_matches.average_reward is None

    with pytest.raises(ValueError, match="logging_scheme"):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="unknown",
        )
    with pytest.raises(ValueError, match="uniform logging propensities"):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="uniform",
            propensities=[0.9],
        )
    with pytest.raises(ValueError, match="minimum"):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="nonuniform",
            propensities=[0.2],
            acceptance_scale=0.3,
        )
    with pytest.raises(TypeError, match="boolean"):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="uniform",
            clone_policy="yes",
        )
    with pytest.raises(TypeError, match="contexts"):
        sequential_replay(
            policy,
            logged_actions=[0],
            logged_rewards=[1.0],
            logging_scheme="uniform",
            contexts=[[[1.0], [1.0]]],
        )
