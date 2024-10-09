from pymab.policies.ucb import StationaryUCBPolicy
from pymab.policies.bayesian_ucb import BayesianUCBPolicy
from pymab.game import Game


def main():
    Q_values = [0.1, 0.8, 0.3, 0.4, 0.9, 0.2, 0.25, 0.6, 0.5, 0.35]
    n_bandits = 10

    reward_distribution = "bernoulli"

    ucb_policy_0 = StationaryUCBPolicy(
        n_bandits=n_bandits, c=0, reward_distribution=reward_distribution
    )

    ucb_policy_1 = StationaryUCBPolicy(
        n_bandits=n_bandits, c=1, reward_distribution=reward_distribution
    )

    ucb_policy_2 = StationaryUCBPolicy(
        n_bandits=n_bandits, c=2, reward_distribution=reward_distribution
    )

    bayesian_ucb = BayesianUCBPolicy(
        n_bandits=n_bandits, reward_distribution=reward_distribution
    )

    game = Game(
        n_episodes=200,
        n_steps=100,
        Q_values=Q_values,
        policies=[ucb_policy_0, ucb_policy_1, ucb_policy_2, bayesian_ucb],
        n_bandits=n_bandits,
    )

    game.game_loop()

    for policy in game.policies:
        policy.plot_distribution()

    game.plot_average_reward_by_step()

    game.plot_average_reward_by_step_smoothed()

    game.plot_rate_optimal_actions_by_step()


if __name__ == "__main__":
    main()
