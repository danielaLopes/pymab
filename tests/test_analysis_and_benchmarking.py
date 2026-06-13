import tempfile
import unittest
from pathlib import Path

import numpy as np

from pymab.benchmarking import compare, confidence_interval_margin
from pymab.environments import BanditEnvironment
from pymab.offline import replay_evaluate
from pymab.policies import LogisticContextualBanditPolicy, RandomPolicy, UCBPolicy
from pymab.simulation import Experiment, ExperimentConfig, SimulationResult


class ResultExportTests(unittest.TestCase):
    def _result(self) -> SimulationResult:
        return Experiment(
            environment=BanditEnvironment(q_values=np.array([0.0, 1.0])),
            policies=[RandomPolicy(n_arms=2), UCBPolicy(n_arms=2)],
            config=ExperimentConfig(n_episodes=3, n_steps=5, seed=7),
        ).run()

    def test_to_dict_is_json_ready(self) -> None:
        result = self._result()
        payload = result.to_dict()

        self.assertEqual(payload["policy_names"], list(result.policy_names))
        self.assertIsInstance(payload["rewards"], list)
        self.assertIsInstance(payload["actions"], list)

    def test_to_pandas_returns_tidy_rows(self) -> None:
        result = self._result()
        frame = result.to_pandas()

        self.assertEqual(
            len(frame), result.n_episodes * result.n_steps * result.n_policies
        )
        self.assertIn("policy_name", frame.columns)
        self.assertIn("regret", frame.columns)
        self.assertIn("selected_optimal_action", frame.columns)

    def test_npz_roundtrip_preserves_arrays(self) -> None:
        result = self._result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.npz"
            result.save_npz(path)
            loaded = SimulationResult.load_npz(path)

        np.testing.assert_allclose(loaded.rewards, result.rewards)
        np.testing.assert_array_equal(loaded.actions, result.actions)
        np.testing.assert_allclose(loaded.expected_rewards, result.expected_rewards)
        np.testing.assert_allclose(loaded.q_values, result.q_values)
        self.assertEqual(loaded.policy_names, result.policy_names)


class BenchmarkingTests(unittest.TestCase):
    def test_compare_summarizes_policies_and_identifies_winner(self) -> None:
        benchmark = compare(
            [RandomPolicy(n_arms=3), UCBPolicy(n_arms=3)],
            environment=BanditEnvironment(q_values=np.array([0.0, 0.2, 1.0])),
            n_episodes=30,
            n_steps=40,
            seeds=(1, 2, 3),
        )

        summary = benchmark.summary()
        self.assertEqual(len(summary), 2)
        self.assertEqual(benchmark.best_policy, summary[1]["policy_name"])
        self.assertLess(
            summary[1]["mean_cumulative_regret"],
            summary[0]["mean_cumulative_regret"],
        )

        frame = benchmark.to_pandas()
        self.assertEqual(
            list(frame["policy_name"]), [row["policy_name"] for row in summary]
        )
        self.assertIn("cumulative_regret_ci", frame.columns)

    def test_confidence_interval_margin_handles_single_seed(self) -> None:
        self.assertEqual(
            confidence_interval_margin(np.array([1.0]), confidence_level=0.95),
            0.0,
        )
        self.assertGreater(
            confidence_interval_margin(np.array([1.0, 2.0, 3.0])),
            0.0,
        )


class OfflineReplayTests(unittest.TestCase):
    def test_replay_evaluate_accepts_matching_logged_events(self) -> None:
        policy = RandomPolicy(n_arms=1)
        result = replay_evaluate(
            policy,
            logged_actions=np.array([0, 0, 0]),
            logged_rewards=np.array([1.0, 0.0, 1.0]),
            propensities=np.array([1.0, 1.0, 1.0]),
            seed=1,
        )

        self.assertEqual(result.n_accepted_events, 3)
        self.assertEqual(result.acceptance_rate, 1.0)
        self.assertEqual(result.average_reward, 2.0 / 3.0)
        self.assertEqual(result.ips_estimate, 2.0 / 3.0)
        np.testing.assert_allclose(result.cumulative_rewards, np.array([1.0, 1.0, 2.0]))

    def test_replay_evaluate_contextual_requires_contexts(self) -> None:
        policy = LogisticContextualBanditPolicy(n_arms=1, n_features=2)
        with self.assertRaises(ValueError):
            replay_evaluate(
                policy,
                logged_actions=np.array([0]),
                logged_rewards=np.array([1.0]),
            )

    def test_replay_evaluate_updates_contextual_policy_with_contexts(self) -> None:
        policy = LogisticContextualBanditPolicy(n_arms=1, n_features=2)
        contexts = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        result = replay_evaluate(
            policy,
            logged_actions=np.array([0, 0]),
            logged_rewards=np.array([1.0, 0.0]),
            contexts=contexts,
            seed=2,
            clone_policy=False,
        )

        self.assertEqual(result.n_accepted_events, 2)
        self.assertFalse(np.allclose(policy.theta, np.zeros((1, 2))))


if __name__ == "__main__":
    unittest.main()
