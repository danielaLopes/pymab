import unittest

import numpy as np

from pymab.distributions import BernoulliReward, GaussianReward
from pymab.environments import AbruptShift, BanditEnvironment, GradualDrift
from pymab.metrics import moving_average
from pymab.policies import GreedyPolicy, UCBPolicy
from pymab.simulation import Experiment, ExperimentConfig


class DistributionTests(unittest.TestCase):
    def test_bernoulli_returns_scalar_float(self) -> None:
        rng = np.random.default_rng(1)
        reward = BernoulliReward().sample_one(0.8, rng)
        self.assertIsInstance(reward, float)
        self.assertIn(reward, {0.0, 1.0})

    def test_bernoulli_validates_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            BernoulliReward().sample(np.array([1.5]), np.random.default_rng(1))

    def test_gaussian_generation_is_seeded(self) -> None:
        left = GaussianReward().initial_values(
            mean=0, scale=1, n_arms=3, rng=np.random.default_rng(42)
        )
        right = GaussianReward().initial_values(
            mean=0, scale=1, n_arms=3, rng=np.random.default_rng(42)
        )
        np.testing.assert_allclose(left, right)


class EnvironmentTests(unittest.TestCase):
    def test_abrupt_shift_does_not_shift_at_zero_by_default(self) -> None:
        env = BanditEnvironment(
            q_values=np.array([1.0, 2.0]),
            dynamics=AbruptShift(change_frequency=2, change_magnitude=1.0),
        )
        env.advance(step=0, rng=np.random.default_rng(1))
        np.testing.assert_allclose(env.q_values, np.array([1.0, 2.0]))

    def test_gradual_drift_changes_values(self) -> None:
        env = BanditEnvironment(
            q_values=np.array([1.0, 2.0]),
            dynamics=GradualDrift(change_rate=0.1),
        )
        env.advance(step=1, rng=np.random.default_rng(1))
        self.assertFalse(np.allclose(env.q_values, np.array([1.0, 2.0])))


class SimulationTests(unittest.TestCase):
    def test_simulation_shapes_and_regret_are_expected_value_based(self) -> None:
        env = BanditEnvironment(q_values=np.array([0.0, 1.0]))
        result = Experiment(
            environment=env,
            policies=[GreedyPolicy(n_arms=2), UCBPolicy(n_arms=2)],
            config=ExperimentConfig(n_episodes=3, n_steps=4, seed=10),
        ).run()

        self.assertEqual(result.rewards.shape, (3, 4, 2))
        self.assertEqual(result.actions.shape, (3, 4, 2))
        self.assertEqual(result.cumulative_regret.shape, (4, 2))
        np.testing.assert_allclose(
            result.regret,
            result.optimal_values[:, :, np.newaxis] - result.expected_rewards,
        )

    def test_seed_reproducibility(self) -> None:
        def run_once():
            return Experiment(
                environment=BanditEnvironment(q_values=np.array([0.2, 0.8])),
                policies=[UCBPolicy(n_arms=2)],
                config=ExperimentConfig(n_episodes=2, n_steps=8, seed=123),
            ).run()

        first = run_once()
        second = run_once()
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_allclose(first.rewards, second.rewards)

    def test_moving_average_validation(self) -> None:
        np.testing.assert_allclose(
            moving_average(np.array([1.0, 2.0, 3.0]), 2), np.array([1.5, 2.5])
        )
        with self.assertRaises(ValueError):
            moving_average(np.array([1.0]), 2)


if __name__ == "__main__":
    unittest.main()
