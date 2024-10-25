.. PyMultiBandits documentation master file, created by
   sphinx-quickstart on Fri Aug  2 17:05:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

============================
Reward Distribution documentation
============================

Reward Distribution (reward_distribution.py)
--------------------------------------------

The Reward Distribution module is a crucial component of the PyMultiBandits library, responsible for simulating the stochastic nature of rewards in multi-armed bandit problems. Key features include:

- Various probability distributions for modeling reward outcomes (e.g., Bernoulli, Gaussian)
- Customizable parameters for each distribution type
- Support for both stationary and non-stationary reward scenarios
- Methods for sampling rewards and updating distribution parameters

This module provides the foundation for creating realistic and diverse bandit environments, allowing users to test policies under different reward conditions.

.. automodule:: pymab.reward_distribution
   :members:
   :undoc-members:
   :show-inheritance:



--------------------------------------------------------------------------
Example Usage
--------------------------------------------------------------------------

.. code-block:: python

   from pymab.reward_distribution import BernoulliDistribution, GaussianDistribution
   from pymab.game import Game
   from pymab.policies.epsilon_greedy import EpsilonGreedyPolicy

   Q_values = [0.3, 0.5, 0.7]
   rewards = []
   for q_value in Q_values:
      rewards.append(BernoulliRewardDistribution.get_reward(q_value=q_value, variance=2))
