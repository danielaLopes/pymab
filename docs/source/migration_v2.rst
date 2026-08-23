Migrating from v1 to v2
=======================

Version 2 intentionally breaks interfaces that made experiment identity or
statistical interpretation ambiguous.

.. list-table:: API replacements
   :header-rows: 1

   * - Version 1
     - Version 2
   * - ``n_steps``
     - ``horizon``
   * - ``n_episodes``
     - ``n_replicates``
   * - sequence of policies
     - mapping of stable policy IDs to policies
   * - ``q_values``
     - ``means``
   * - ``reward_distribution``
     - ``reward_model``
   * - ``Game``
     - ``Experiment``
   * - ``replay_evaluate``
     - ``sequential_replay`` or ``estimate_policy_value``
   * - ``best_policy``
     - ``lowest_mean_regret_policy`` and paired baseline comparisons
   * - generic ``from_distribution``
     - ``from_prior`` with an explicit ``ArmPrior``

The compatibility modules, mixins, implicit results directory, and CamelCase
policy factory functions were removed. Import explicit policy classes such as
``BernoulliThompsonSamplingPolicy`` or ``GaussianBayesianUCBPolicy``.

Seeded numerical traces differ from v1 because v2 fixes random-stream coupling.
Within v2, traces are invariant to policy mapping order and to adding unrelated
policies.

Final v2 contract hardening
---------------------------

The finalized v2 API also removes ambiguous contracts that existed during v2
development:

* ``RewardModel.sample_one`` requires an explicit keyword-only ``rng``.
* ``sequential_replay`` requires ``logging_scheme``. Non-uniform replay requires
  logged-action propensities and uses propensity-aware rejection sampling.
* ``clone_policy`` and ``reset_policy`` are separate replay controls.
* Integer-labelled arrays reject floats, strings, and booleans instead of
  coercing them.
* Incompatible policy/environment capabilities raise ``CompatibilityError``;
  invalid logged support raises ``OverlapError``; persistence failures raise
  ``SerializationError``.
* Array-backed records expose ``equals``. Mutable policies and environments do
  not define value equality.
* Result schema 3 adds immutable runtime/component provenance and optional
  recorded contexts. Schema 2 payloads migrate with explicit unknown
  provenance.
* Sliding-window policies live in ``pymab.policies.nonstationary`` and
  change-point policies in ``pymab.policies.change_detection``. Importing all
  policy classes from ``pymab.policies`` remains the supported facade.
* UCB and MOSS expose ``reward_scale``; MOSS rejects ``horizon < n_arms``.
* Plotly is the sole dependency of the ``plot`` extra. Matplotlib was removed
  because PyMAB does not expose a Matplotlib backend.
* Bootstrap controls now live in ``pymab.statistics.BootstrapConfig``.
  ``BenchmarkResult`` receives a ``BenchmarkConfig``; ``compare`` receives it
  as ``analysis``; and ``estimate_policy_value`` receives an
  ``EstimatorConfig``. Plot helpers accept the same ``BootstrapConfig``.
* ``BenchmarkResult.summary`` and ``compare_to_baseline`` return immutable
  typed records. Use each record's attributes in Python and ``to_dict`` when a
  flat JSON-compatible representation is required.
* Import ``bootstrap_mean_interval`` and ``standard_error`` from
  ``pymab.statistics``. ``BootstrapBandConfig`` and the separate offline
  bootstrap keywords were removed.

Minimal v2 replacement
----------------------

Replace a v1 ``Game`` with an explicit environment, named policy mapping, and
required experiment seed:

.. testcode:: migration

   import numpy as np

   from pymab import BanditEnvironment, Experiment, ExperimentConfig
   from pymab.policies import UCBPolicy

   result = Experiment(
       environment=BanditEnvironment(means=np.array([0.2, 0.8])),
       policies={"ucb": UCBPolicy(n_arms=2)},
       config=ExperimentConfig(horizon=10, n_replicates=2, seed=7),
       metadata={"migration": "v1-to-v2"},
   ).run()

   assert result.policy_ids == ("ucb",)
   assert result.rewards.shape == (2, 10, 1)
   assert result.metadata["migration"] == "v1-to-v2"
