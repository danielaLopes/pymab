Native backend performance
==========================

These measurements compare isolated Python-reference and release-mode Rust
workers on the same machine.
The report is generated from ``benchmarks/results/local.json``; timings
are medians and memory is sampled child-process RSS after imports.

.. list-table:: Runtime by canonical workload
   :header-rows: 1

   * - Case
     - Decisions
     - Python (s)
     - Rust (s)
     - Speedup
   * - stationary
     - 112,000
     - 3.1531
     - 0.2532
     - 12.45x
   * - bernoulli
     - 16,000
     - 4.8009
     - 0.0923
     - 52.02x
   * - nonstationary
     - 80,000
     - 5.1725
     - 0.2527
     - 20.47x
   * - contextual
     - 8,000
     - 1.1837
     - 0.0501
     - 23.65x

Aggregate results
-----------------

* Classic runtime speedup: **21.94x**
* Contextual runtime speedup: **23.65x**
* Incremental peak RSS ratio (Rust/Python): **0.351**

Memory evidence
---------------

State memory is capacity-aware on the Rust side and recursively measured
for the private Python reference objects after identical shared-fixture traces.
Incremental RSS excludes each worker's post-import baseline.

.. list-table:: Incremental peak RSS
   :header-rows: 1

   * - Case
     - Python (MiB)
     - Rust (MiB)
   * - stationary
     - 21.64
     - 17.75
   * - bernoulli
     - 62.78
     - 3.78
   * - nonstationary
     - 18.50
     - 14.66
   * - contextual
     - 4.38
     - 1.47

.. list-table:: Policy state after shared parity traces
   :header-rows: 1

   * - Policy
     - Python (bytes)
     - Rust (bytes)
     - Rust/Python
   * - bernoulli_bayesian_ucb
     - 1,615
     - 208
     - 0.129
   * - bernoulli_thompson_sampling
     - 1,542
     - 200
     - 0.130
   * - change_point_ucb
     - 2,866
     - 432
     - 0.151
   * - cusum_ucb
     - 2,859
     - 432
     - 0.151
   * - decaying_epsilon_greedy
     - 1,227
     - 128
     - 0.104
   * - discounted_bernoulli_thompson_sampling
     - 1,650
     - 200
     - 0.121
   * - discounted_ucb
     - 1,657
     - 208
     - 0.126
   * - epsilon_greedy
     - 1,068
     - 112
     - 0.105
   * - exp3
     - 1,551
     - 200
     - 0.129
   * - gaussian_bayesian_ucb
     - 1,674
     - 216
     - 0.129
   * - gaussian_thompson_sampling
     - 1,593
     - 208
     - 0.131
   * - gradient_bandit
     - 1,088
     - 112
     - 0.103
   * - greedy
     - 1,028
     - 120
     - 0.117
   * - kl_ucb
     - 1,304
     - 128
     - 0.098
   * - lin_ucb
     - 1,155
     - 184
     - 0.159
   * - linear_epsilon_greedy
     - 858
     - 96
     - 0.112
   * - linear_thompson_sampling
     - 1,167
     - 184
     - 0.158
   * - logistic_contextual_bandit
     - 925
     - 104
     - 0.112
   * - median_elimination
     - 1,883
     - 242
     - 0.129
   * - moss
     - 1,223
     - 128
     - 0.105
   * - page_hinkley_ucb
     - 2,866
     - 432
     - 0.151
   * - random
     - 1,036
     - 120
     - 0.116
   * - sliding_window_bernoulli_thompson_sampling
     - 2,779
     - 304
     - 0.109
   * - sliding_window_ucb
     - 2,384
     - 232
     - 0.097
   * - softmax
     - 1,072
     - 112
     - 0.104
   * - successive_elimination
     - 1,318
     - 146
     - 0.111
   * - ucb
     - 1,139
     - 120
     - 0.105

Measurement environment
-----------------------

* **implementation:** CPython
* **machine:** arm64
* **numpy:** 2.4.6
* **platform:** macOS-26.5.2-arm64-arm-64bit
* **processor:** arm
* **pymab:** 2.0.0
* **python:** 3.12.7
* **rust_core:** 2.0.0
