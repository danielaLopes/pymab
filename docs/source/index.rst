PyMAB documentation
===================

PyMAB provides reproducible multi-armed bandit experiments for Python 3.11+.
The v1 API separates environments, policies, simulations, metrics, and plotting.

Quick start
-----------

.. code-block:: python

   import numpy as np

   from pymab.environments import BanditEnvironment
   from pymab.policies import EpsilonGreedyPolicy, UCBPolicy
   from pymab.simulation import Experiment, ExperimentConfig

   environment = BanditEnvironment(q_values=np.array([0.1, 0.4, 0.8]))
   result = Experiment(
       environment=environment,
       policies=[
           EpsilonGreedyPolicy(n_arms=3, epsilon=0.1),
           UCBPolicy(n_arms=3),
       ],
       config=ExperimentConfig(n_episodes=200, n_steps=500, seed=42),
   ).run()

   print(result.average_reward_by_step[-1])
   print(result.cumulative_regret[-1])

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   simulation
   benchmarking
   decision_guide
   examples
   environments
   policies
   distributions
   game
   reward_distribution

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
