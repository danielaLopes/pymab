Benchmarking and inference
==========================

``compare`` runs all policies inside one paired experiment. Its independent
unit is the replicate, not an individual time step.

.. testcode:: benchmarking

   import numpy as np

   from pymab import BanditEnvironment, ExperimentConfig, compare
   from pymab.benchmarking import BenchmarkConfig
   from pymab.policies import RandomPolicy, UCBPolicy
   from pymab.statistics import BootstrapConfig

   benchmark = compare(
       {"random": RandomPolicy(n_arms=3), "ucb": UCBPolicy(n_arms=3)},
       environment=BanditEnvironment(means=np.array([0.1, 0.3, 0.8])),
       config=ExperimentConfig(horizon=20, n_replicates=4, seed=7),
       analysis=BenchmarkConfig(
           bootstrap=BootstrapConfig(n_resamples=200, seed=91),
           baseline="random",
       ),
   )

   assert benchmark.lowest_mean_regret_policy in {"random", "ucb"}
   assert len(benchmark.summary()) == 2
   assert len(benchmark.compare_to_baseline()) == 1

``summary`` returns immutable ``PolicySummary`` records containing typed
``IntervalEstimate`` values for cumulative expected regret, realized total
reward, optimal-action rate, and final simple regret. ``compare_to_baseline``
returns ``PolicyComparison`` records and bootstraps paired replicate
differences, preserving the common-random-number design. ``to_dict`` is the
flat JSON-compatible boundary.

``lowest_mean_regret_policy`` is only a point-estimate ranking. Use paired
intervals, effect size, domain relevance, and adequate independent replication
before making a substantive claim.

Persistence and tables
----------------------

``SimulationResult.save_npz`` stores arrays plus versioned JSON metadata. Both
NPZ and JSON formats have matching load methods. ``to_pandas`` includes policy
IDs, replicate numbers, and replicate seeds on every tidy row.

.. automodule:: pymab.benchmarking
   :members:
   :show-inheritance:

Plot uncertainty bands use ``BootstrapConfig`` and a bounded-memory,
replicate-level bootstrap. Benchmark plot wrappers inherit the benchmark's
analysis configuration.

.. automodule:: pymab.plotting
   :members:
   :show-inheritance:
