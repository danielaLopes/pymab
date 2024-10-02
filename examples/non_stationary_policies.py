from pymab import setup_logging
from pymab.policies.ucb import DiscountedUCBPolicy, SlidingWindowUCBPolicy, StationaryUCBPolicy
from pymab.game import EnvironmentChangeType, Game


def run_game(game: Game, game_name: str):
    game.game_loop()

    game.plot_average_reward_by_step(plot_name=game_name)
    game.plot_average_reward_by_step_smoothed(plot_name=game_name)
    game.plot_bandit_selection_evolution(plot_name=game_name)
    game.plot_cumulative_regret_by_step(plot_name=game_name)
    game.plot_Q_values_evolution_by_bandit_first_episode(plot_name=game_name)
    game.plot_rate_optimal_actions_by_step(plot_name=game_name)


def main():
    setup_logging(level="INFO")

    n_bandits = 10
    n_episodes = 1000
    n_steps = 2000
    # n_episodes = 100
    # n_steps = 200

    ucb_stationary_policy = StationaryUCBPolicy(
        n_bandits=10,
        c=1,
    )

    ucb_sliding_window_policy_50 = SlidingWindowUCBPolicy(
        n_bandits=10,
        c=1,
        window_size=50,
    )
    ucb_sliding_window_policy_100 = SlidingWindowUCBPolicy(
        n_bandits=10,
        c=1,
        window_size=100,
    )

    ucb_sliding_window_policy_200 = SlidingWindowUCBPolicy(
        n_bandits=10,
        c=1,
        window_size=200,
    )

    ucb_discounted_policy_0_9 = DiscountedUCBPolicy(
        n_bandits=10,
        c=1,
        discount_factor=0.9,
    )
    ucb_discounted_policy_0_5 = DiscountedUCBPolicy(
        n_bandits=10,
        c=1,
        discount_factor=0.5,
    )
    ucb_discounted_policy_0_1 = DiscountedUCBPolicy(
        n_bandits=10,
        c=1,
        discount_factor=0.1,
    )

    policies = [
            ucb_stationary_policy,
            ucb_sliding_window_policy_50,
            ucb_sliding_window_policy_100,
            ucb_sliding_window_policy_200,
            ucb_discounted_policy_0_1,
            ucb_discounted_policy_0_5,
            ucb_discounted_policy_0_9
        ]

    non_stationary_gradual_game_small_cahnge = Game(
        n_episodes=n_episodes,
        n_steps=n_steps,
        policies=policies,
        n_bandits=n_bandits,
        environment_change=EnvironmentChangeType.GRADUAL,
        change_params={'change_rate': 0.01}
    )
    run_game(non_stationary_gradual_game_small_cahnge, 'non_stationary_gradual_game_0_01')

    non_stationary_gradual_game_big_change = Game(
        n_episodes=n_episodes,
        n_steps=n_steps,
        policies=policies,
        n_bandits=n_bandits,
        environment_change=EnvironmentChangeType.GRADUAL,
        change_params={'change_rate': 0.5}
    )
    run_game(non_stationary_gradual_game_big_change, 'non_stationary_gradual_game_0_5')

    non_stationary_abrupt_game = Game(
        n_episodes=n_episodes,
        n_steps=n_steps,
        policies=policies,
        n_bandits=n_bandits,
        environment_change=EnvironmentChangeType.ABRUPT,
        change_params={'change_frequency': 100, 'change_magnitude': 0.5},
    )
    run_game(non_stationary_abrupt_game, game_name='non_stationary_abrupt_game_100_0_5')

    non_stationary_random_arm_swapping_game = Game(
        n_episodes=n_episodes,
        n_steps=n_steps,
        policies=policies,
        n_bandits=n_bandits,
        environment_change=EnvironmentChangeType.RANDOM_ARM_SWAPPING,
        change_params={'shift_probability': 0.2},
    )
    run_game(non_stationary_random_arm_swapping_game, game_name='non_stationary_random_arm_swapping_game_0_2')

    stationary_game = Game(
        n_episodes=n_episodes,
        n_steps=n_steps,
        policies=policies,
        n_bandits=n_bandits,
    )
    run_game(stationary_game, game_name='stationary_game')


if __name__ == "__main__":
    main()