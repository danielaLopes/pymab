import unittest
import warnings

from pymab.game import EnvironmentChangeType, Game
from pymab.policies import GreedyPolicy


class GameCompatibilityTests(unittest.TestCase):
    def test_game_wrapper_runs(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            game = Game(
                n_episodes=2,
                n_steps=3,
                policies=[GreedyPolicy(n_arms=2)],
                n_bandits=2,
                Q_values=[0.1, 0.9],
                environment_change=EnvironmentChangeType.STATIONARY,
                seed=1,
            )
        game.game_loop()
        self.assertEqual(game.rewards_by_policy.shape, (2, 3, 1))
        self.assertEqual(game.total_rewards_by_step.shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
