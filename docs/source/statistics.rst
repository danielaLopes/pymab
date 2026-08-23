Statistical inference
=====================

``BootstrapConfig`` is the shared uncertainty configuration for scalar
analysis, benchmark summaries, offline estimators, and plotting. A fixed seed
produces the same result regardless of how the bounded resampling workspace is
chunked.

``bootstrap_mean_interval`` returns a typed ``IntervalEstimate`` rather than a
bare pair of bounds. The record includes the point estimate, bootstrap standard
error, percentile interval, observation count, confidence method, and
resampling unit.

.. testcode:: statistics

   import numpy as np

   from pymab.statistics import BootstrapConfig, bootstrap_mean_interval

   estimate = bootstrap_mean_interval(
       np.array([1.0, 2.0, 3.0, 4.0]),
       config=BootstrapConfig(n_resamples=200, seed=7),
   )

   assert estimate.estimate == 2.5
   assert estimate.ci_lower is not None
   assert estimate.n_observations == 4
   assert estimate.resampling_unit == "replicate"

Percentile intervals quantify Monte Carlo uncertainty under the selected
resampling unit; they do not correct bias or make dependent observations
independent. Use cluster IDs for dependent offline observations and use enough
independent experiment replicates for benchmark inference.

.. automodule:: pymab.statistics
   :members:
   :show-inheritance:
