import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List
from functools import lru_cache

from tqdm import tqdm
from joblib import Parallel, delayed

from pymab.policies.policy import Policy


class Game:
    n_episodes: int
    n_steps: int
    Q_values: np.ndarray
    set_Q_values_flag: bool
    Q_values_mean: float
    Q_values_variance: float
    policy: Policy
    n_bandits: int
    rewards: np.ndarray
    is_stationary: bool

    def __init__(
        self,
        n_episodes: int,
        n_steps: int,
        policies: List[Policy],
        n_bandits: int,
        Q_values: List[float] = [],
        Q_values_mean: float = 0.0,
        Q_values_variance: float = 1.0,
        is_stationary: bool = False,
    ) -> None:
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.policies = policies
        self.n_bandits = n_bandits
        self.Q_values = Q_values
        self.set_Q_values_flag = len(self.Q_values) == 0
        self.Q_values_mean = Q_values_mean
        self.Q_values_variance = Q_values_variance
        self.rewards_by_policy = np.zeros(
            (self.n_episodes, self.n_steps, len(self.policies)), dtype=float
        )
        self.actions_selected_by_policy = np.zeros(
            (self.n_episodes, self.n_steps, len(self.policies))
        )
        self.optimal_actions = np.zeros((self.n_episodes))

        self.is_stationary = is_stationary

        cmap = mpl.colormaps["Set1"]
        self.colors = cmap(np.linspace(0, 1, len(self.policies)))

    def new_episode(self, episode_idx: int) -> None:
        if self.is_stationary == False or episode_idx == 0:
            if self.set_Q_values_flag == True:
                self.generate_Q_values()
        for policy in self.policies:
            policy.Q_values = self.Q_values
            policy.reset()

    def generate_Q_values(self) -> None:
        # Default uses Gaussian distribution
        self.Q_values = np.random.normal(
            self.Q_values_mean, self.Q_values_variance, self.n_bandits
        )

    def plot_Q_values(self) -> None:
        cmap = plt.colormaps["Set1"].resampled(self.n_bandits).colors
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(range(len(self.Q_values)), self.Q_values, color=cmap)
        plt.title("Q values for each action")
        plt.xlabel("Actions")
        plt.ylabel("Q value")
        plt.show()

    def game_loop(self) -> None:
        for episode in range(self.n_episodes):
            self.new_episode(episode)
            self.optimal_actions[episode] = np.argmax(self.Q_values)

            # for policy_index, policy in tqdm(enumerate(self.policies), desc="Running game for each policy...", total=len(self.policies)):
            for policy_index, policy in enumerate(self.policies):
                # for step in tqdm(range(self.n_steps), desc="Running steps...", total=self.n_steps):
                for step in range(self.n_steps):
                    action, reward = policy.select_action()
                    self.rewards_by_policy[episode, step, policy_index] = reward
                    self.actions_selected_by_policy[episode, step, policy_index] = (
                        action
                    )

    @property
    @lru_cache(maxsize=None)
    def average_rewards_by_step(self) -> np.ndarray:
        return np.mean(self.rewards_by_policy, axis=0)

    @property
    @lru_cache(maxsize=None)
    def average_rewards_by_episode(self) -> np.ndarray:
        return np.mean(self.rewards_by_policy, axis=1)

    @property
    @lru_cache(maxsize=None)
    def total_rewards_by_step(self) -> np.ndarray:
        return np.cumsum(np.mean(self.rewards_by_policy, axis=0), axis=0)

    def plot_average_reward_by_step(self) -> None:
        fig = plt.figure(figsize=(18, 12), dpi=300)

        for policy_index, policy in enumerate(self.policies):
            plt.plot(
                self.average_rewards_by_step[:, policy_index],
                color=self.colors[policy_index],
                label=repr(policy),
            )

        plt.gca().tick_params(axis="both", labelsize=24)
        plt.title(
            f"Average reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
            fontsize=30,
        )
        plt.xlabel("Steps", fontsize=26)
        plt.ylabel("Average reward", fontsize=26)
        plt.legend(fontsize=16)
        plt.show()

    def plot_average_reward_by_episode(self) -> None:
        fig = plt.figure(figsize=(18, 12), dpi=300)

        for policy_index, policy in enumerate(self.policies):
            plt.plot(
                self.average_rewards_by_episode[:, policy_index],
                color=self.colors[policy_index],
                label=repr(policy),
            )

        plt.gca().tick_params(axis="both", labelsize=24)
        plt.title(
            f"Average reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
            fontsize=30,
        )
        plt.xlabel("Episodes", fontsize=26)
        plt.ylabel("Average reward", fontsize=26)
        plt.legend(fontsize=16)
        plt.show()

    def plot_total_reward_by_step(self) -> None:
        fig = plt.figure(figsize=(18, 12), dpi=300)

        for policy_index, policy in enumerate(self.policies):
            plt.plot(
                self.total_rewards_by_step[:, policy_index],
                color=self.colors[policy_index],
                label=repr(policy),
            )

        plt.gca().tick_params(axis="both", labelsize=24)
        plt.title(
            f"Cumulative reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
            fontsize=30,
        )
        plt.xlabel("Steps", fontsize=26)
        plt.ylabel("Cumulative reward", fontsize=26)
        plt.legend(fontsize=16)
        plt.show()

    def _moving_average(self, data: np.ndarray, smooth_factor: int):
        return np.convolve(data, np.ones(smooth_factor) / smooth_factor, mode="valid")

    def plot_average_reward_by_step_smoothed(self, smooth_factor: int = 50) -> None:
        fig = plt.figure(figsize=(18, 12), dpi=300)

        average_rewards = np.mean(self.rewards_by_policy, axis=0)

        for policy_index, policy in enumerate(self.policies):
            plt.plot(
                self._moving_average(
                    self.average_rewards_by_step[:, policy_index], smooth_factor
                ),
                color=self.colors[policy_index],
                label=repr(policy),
            )

        plt.gca().tick_params(axis="both", labelsize=24)
        plt.title(
            f"Average reward obtained during the {self.n_steps} steps for {self.n_episodes} episodes",
            fontsize=30,
        )
        plt.xlabel("Steps", fontsize=26)
        plt.ylabel("Average reward", fontsize=26)
        plt.legend(fontsize=16)
        plt.show()

    def plot_rate_optimal_actions_by_step(self) -> None:
        fig = plt.figure(figsize=(18, 12), dpi=300)

        optimal_actions_expanded = np.repeat(
            self.optimal_actions[:, np.newaxis], self.n_steps, axis=1
        )
        optimal_action_selections = (
            self.actions_selected_by_policy
            == optimal_actions_expanded[:, :, np.newaxis]
        )
        percentage_optimal_by_step = np.mean(optimal_action_selections, axis=0) * 100

        for policy_index, policy in enumerate(self.policies):
            plt.plot(
                percentage_optimal_by_step[:, policy_index],
                color=self.colors[policy_index],
                label=repr(policy),
            )

        plt.gca().tick_params(axis="both", labelsize=24)
        plt.title(
            f"Percentage of optimal actions chosen during the {self.n_steps} steps for {self.n_episodes} episodes",
            fontsize=30,
        )
        plt.xlabel("Steps", fontsize=26)
        plt.ylabel("% Optimal actions", fontsize=26)
        plt.ylim(0, 100)
        plt.legend(fontsize=16)
        plt.show()
