from __future__ import annotations

import numpy as np
import pytest

from pymab.distributions import (
    BernoulliReward,
    BetaArmPrior,
    GaussianArmPrior,
    GaussianReward,
    UniformArmPrior,
    UniformReward,
    resolve_reward_model,
)
from pymab.environments import (
    AbruptShift,
    BanditEnvironment,
    GradualDrift,
    ProbabilityDrift,
    RandomArmSwap,
)
from pymab.metrics import moving_average


@pytest.mark.parametrize(
    ("model", "means"),
    [
        (GaussianReward(std=0.2), np.array([-1.0, 2.0])),
        (UniformReward(half_width=0.5), np.array([-1.0, 2.0])),
        (BernoulliReward(), np.array([0.1, 0.9])),
    ],
)
def test_reward_models_are_seeded_and_vectorized(model, means) -> None:
    left = model.sample(means, np.random.default_rng(3))
    right = model.sample(means, np.random.default_rng(3))
    np.testing.assert_array_equal(left, right)
    assert left.shape == means.shape
    assert isinstance(
        model.sample_one(mean=float(means[0]), rng=np.random.default_rng(1)), float
    )


@pytest.mark.parametrize(
    "means", [np.array([np.nan]), np.array([np.inf]), np.array([])]
)
def test_reward_models_reject_non_finite_or_empty_means(means) -> None:
    with pytest.raises(ValueError):
        GaussianReward().validate_means(means)


def test_bernoulli_validates_support() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        BernoulliReward().sample(np.array([1.1]), np.random.default_rng(1))


@pytest.mark.parametrize(
    "constructor",
    [lambda: GaussianReward(0), lambda: UniformReward(-1)],
)
def test_reward_model_parameter_validation(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()


def test_reward_model_resolution() -> None:
    assert isinstance(
        resolve_reward_model("normal", observation_scale=0.2), GaussianReward
    )
    assert isinstance(resolve_reward_model("bernoulli"), BernoulliReward)
    assert isinstance(resolve_reward_model("uniform"), UniformReward)
    instance = GaussianReward()
    assert resolve_reward_model(instance) is instance
    assert isinstance(resolve_reward_model(GaussianReward), GaussianReward)
    with pytest.raises(ValueError, match="unknown"):
        resolve_reward_model("poisson")


@pytest.mark.parametrize(
    "prior",
    [
        GaussianArmPrior(mean=1.0, std=0.1),
        BetaArmPrior(alpha=2.0, beta=3.0),
        UniformArmPrior(low=-1.0, high=2.0),
    ],
)
def test_arm_priors_are_seeded(prior) -> None:
    left = prior.sample(n_arms=4, rng=np.random.default_rng(4))
    right = prior.sample(n_arms=4, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError):
        prior.sample(n_arms=0, rng=np.random.default_rng(4))


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GaussianArmPrior(mean=np.inf),
        lambda: GaussianArmPrior(std=-1),
        lambda: BetaArmPrior(alpha=0),
        lambda: UniformArmPrior(low=2, high=1),
        lambda: UniformArmPrior(low=np.nan),
    ],
)
def test_arm_prior_parameter_validation(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()


def test_environment_from_prior_and_clone_are_independent() -> None:
    environment = BanditEnvironment.from_prior(
        n_arms=3,
        prior=GaussianArmPrior(mean=1.0, std=0.0),
        reward_model=GaussianReward(std=0.1),
        rng=np.random.default_rng(1),
    )
    clone = environment.clone()
    clone.means[0] = 99
    np.testing.assert_allclose(environment.means, np.ones(3))
    assert environment.n_arms == 3


def test_environment_validates_shape_and_dynamics_domain() -> None:
    with pytest.raises(ValueError, match="1D"):
        BanditEnvironment(means=np.ones((2, 2)))
    with pytest.raises(ValueError, match="does not support"):
        BanditEnvironment(
            means=np.array([0.2, 0.8]),
            reward_model=BernoulliReward(),
            dynamics=GradualDrift(),
        )


def test_real_dynamics_behaviors_and_validation() -> None:
    means = np.array([0.0, 1.0])
    drifted = GradualDrift(std=0.1).apply(means, step=1, rng=np.random.default_rng(1))
    assert not np.allclose(drifted, means)
    abrupt = AbruptShift(frequency=2, std=1.0)
    np.testing.assert_array_equal(
        abrupt.apply(means, step=0, rng=np.random.default_rng(1)), means
    )
    assert not np.array_equal(
        abrupt.apply(means, step=2, rng=np.random.default_rng(1)), means
    )
    np.testing.assert_array_equal(
        RandomArmSwap(probability=0).apply(means, step=1, rng=np.random.default_rng(1)),
        means,
    )
    np.testing.assert_array_equal(
        np.sort(
            RandomArmSwap(probability=1).apply(
                means, step=1, rng=np.random.default_rng(2)
            )
        ),
        means,
    )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GradualDrift(std=-1),
        lambda: AbruptShift(frequency=0),
        lambda: AbruptShift(std=-1),
        lambda: ProbabilityDrift(logit_std=-1),
        lambda: ProbabilityDrift(epsilon=0.5),
        lambda: RandomArmSwap(probability=2),
    ],
)
def test_dynamics_parameter_validation(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()


def test_probability_drift_stays_in_support() -> None:
    environment = BanditEnvironment(
        means=np.array([0.0, 0.5, 1.0]),
        reward_model=BernoulliReward(),
        dynamics=ProbabilityDrift(logit_std=5),
    )
    for step in range(20):
        environment.advance(step=step, rng=np.random.default_rng(step))
        assert np.all((environment.means > 0) & (environment.means < 1))


def test_environment_rejects_malformed_custom_dynamics() -> None:
    class BadShape(GradualDrift):
        def apply(self, means, *, step, rng):
            return np.array([1.0])

    environment = BanditEnvironment(means=np.array([0.0, 1.0]), dynamics=BadShape())
    with pytest.raises(ValueError, match="preserve"):
        environment.advance(step=1, rng=np.random.default_rng(1))


def test_moving_average_and_validation() -> None:
    np.testing.assert_allclose(moving_average(np.array([1.0, 2.0, 3.0]), 2), [1.5, 2.5])
    for data, window in [(np.array([1.0]), 0), (np.ones((2, 2)), 1), (np.ones(2), 3)]:
        with pytest.raises(ValueError):
            moving_average(data, window)
