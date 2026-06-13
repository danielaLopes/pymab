Benchmarking and Result Analysis
================================

Use ``compare`` when the question is not just "can this policy run?" but
"which policy won under repeated seeds?"

.. code-block:: python

   import numpy as np

   from pymab import compare
   from pymab.environments import BanditEnvironment
   from pymab.policies import RandomPolicy, ThompsonSamplingPolicy, UCBPolicy

   benchmark = compare(
       [
           RandomPolicy(n_arms=3),
           UCBPolicy(n_arms=3),
           ThompsonSamplingPolicy(n_arms=3),
       ],
       environment=BanditEnvironment(q_values=np.array([0.1, 0.3, 0.8])),
       n_episodes=100,
       n_steps=500,
       seeds=(1, 2, 3, 4, 5),
   )

   print(benchmark.best_policy)
   print(benchmark.summary())

Summary Columns
---------------

``BenchmarkResult.summary()`` returns one dictionary per policy:

``policy_name``
   Readable policy representation.

``mean_cumulative_regret``
   Mean final cumulative expected regret across top-level seeds. Lower is
   better.

``cumulative_regret_ci``
   Normal-theory confidence interval margin for final regret.

``mean_total_reward``
   Mean cumulative realized reward across top-level seeds. Higher is better.

``total_reward_ci``
   Confidence interval margin for total reward.

``mean_optimal_action_rate``
   Fraction of steps that selected the best current action.

DataFrame Export
----------------

Install the analysis extra to get pandas support:

.. code-block:: bash

   pip install "pymab[analysis]"

Then convert simulations or benchmarks into tidy tables:

.. code-block:: python

   result_frame = benchmark.combined.to_pandas()
   summary_frame = benchmark.to_pandas()

The simulation DataFrame has one row per episode, step, and policy. It includes
actions, realized rewards, expected rewards, regret, and whether the selected
action was optimal.

Persistence
-----------

Use compressed NumPy archives for reproducible experiment artifacts:

.. code-block:: python

   benchmark.combined.save_npz("results/benchmark.npz")

   from pymab.simulation import SimulationResult

   loaded = SimulationResult.load_npz("results/benchmark.npz")

``SimulationResult.to_dict()`` is useful for APIs and lightweight JSON
inspection. Prefer ``save_npz`` for larger arrays.

Standard Plots
--------------

Install the plot extra:

.. code-block:: bash

   pip install "pymab[plot]"

Then create standard comparison plots:

.. code-block:: python

   benchmark.plot_average_reward(output_path="results/average_reward.html")
   benchmark.plot_cumulative_regret(output_path="results/regret.html")
   benchmark.plot_optimal_action_rate(output_path="results/optimal_rate.html")

API Reference
-------------

.. automodule:: pymab.benchmarking
   :members:
   :undoc-members:
   :show-inheritance:
