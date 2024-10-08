import numpy as np

class StationaryPolicyMixin:
    def _update_estimate(self, action_index: int, reward: float) -> None:
        self.actions_estimated_reward[action_index] += (
            reward - self.actions_estimated_reward[action_index]
        ) / self.times_selected[action_index]

class NonStationaryPolicyMixin:
    def _update_estimate(self, action_index: int, reward: float) -> None:
        self._recalculate_estimate(action_index)

    def _recalculate_estimate(self, action_index: int) -> None:
        raise NotImplementedError("Subclasses must implement this method")

class SlidingWindowMixin(NonStationaryPolicyMixin):
    """
    Maintains a fixed-size window of the most recent observations for each arm. This approach allows the algorithm to
    adapt to changes in the environment by forgetting older, potentially outdated information.
    Can adapt to abrupt changes in the environment. Provides a clear cut-off for old information. May discard useful
    information in slowly changing environments. Sliding window is expected to be more beneficial in environments
    with periodic and dramatic changes.
    """
    def __init__(self, *, window_size: int = 100):
        """
        :param window_size: The number of most recent observations to consider for each arm.
        """
        self.window_size = window_size

    def _recalculate_estimate(self, action_index: int) -> None:
        if len(self.rewards_history[action_index]) > self.window_size:
            self.rewards_history[action_index] = self.rewards_history[action_index][-self.window_size:]
        self.actions_estimated_reward[action_index] = np.mean(self.rewards_history[action_index])

class DiscountedMixin(NonStationaryPolicyMixin):
    """
    Gives more weight to recent observation, and less weight to older ones. With a discount factor close to 1, the
    algorithm has a longer memory, and changes slowly in response to new reward distributions. With a discount factor
    close to 0, the algorithm adapts quickly, but potentially overfits to noise in the environment.
    This approach is particularly useful in non-stationary environments where the reward distributions of arms may
    change over time. Discount factor is expected to be more beneficial in environments with gradual and continuous
    changes. May be less efficient in stationary environments compared to standard Policy.
    """
    def __init__(self, *, discount_factor: float = 0.9):
        """
        :param discount_factor: A value between 0 and 1 that determines how much weight is given to past observations.
        A smaller value gives more importance to recent rewards.
        """
        self.discount_factor = discount_factor

    def _recalculate_estimate(self, action_index: int) -> None:
        if len(self.rewards_history[action_index]) > 1:
            prev_estimate = self.actions_estimated_reward[action_index]
            latest_reward = self.rewards_history[action_index][-1]
            self.actions_estimated_reward[action_index] = (
                self.discount_factor * prev_estimate + (1 - self.discount_factor) * latest_reward
            )