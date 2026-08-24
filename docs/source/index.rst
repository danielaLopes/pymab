PyMAB documentation
===================

PyMAB provides reliable, reproducible multi-armed bandit experiments for
Python 3.11+. Version 2 uses stable named random streams, validated immutable
results, paired replicate-level inference, and support-aware environments.

Quick start
-----------

.. testcode:: quickstart

   import numpy as np

   from pymab.environments import BanditEnvironment
   from pymab.policies import EpsilonGreedyPolicy, UCBPolicy
   from pymab.simulation import Experiment, ExperimentConfig

   environment = BanditEnvironment(means=np.array([0.1, 0.4, 0.8]))
   result = Experiment(
       environment=environment,
       policies={
           "epsilon-greedy": EpsilonGreedyPolicy(n_arms=3, epsilon=0.1),
           "ucb": UCBPolicy(n_arms=3),
       },
       config=ExperimentConfig(horizon=20, n_replicates=3, seed=42),
   ).run()

   assert result.average_reward_by_step.shape == (20, 2)
   assert result.cumulative_regret.shape == (20, 2)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   simulation
   reliability
   types
   statistics
   benchmarking
   arcade
   decision_guide
   examples
   policy_assumptions
   environments
   policies
   offline
   distributions
   migration_v2

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
