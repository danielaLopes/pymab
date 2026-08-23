Offline evaluation
==================

Offline evaluation is split into two distinct operations.

``sequential_replay`` samples actions from an adaptive online policy, accepts
logged events under an explicit logging design, and updates only on accepted
observations. Its acceptance rate is an essential diagnostic. Set
``logging_scheme="uniform"`` only when every action was logged with probability
``1 / n_arms``. For non-uniform logs, pass the propensity of each logged action;
after an action match, replay rejection-samples with probability ``c / p``.
The default ``c`` is the smallest observed propensity.

``clone_policy`` and ``reset_policy`` are independent. The defaults evaluate a
fresh clone. Set both to false for an in-place warm-start evaluation; setting
``clone_policy=False`` no longer resets the supplied policy implicitly.

``estimate_policy_value`` evaluates a fixed target probability rule from a
validated ``LoggedBanditDataset``. IPS and SNIPS require behavior propensities.
Doubly robust evaluation additionally requires out-of-fold predictions through
the ``CrossFittedRewardModel`` protocol.

Every estimate reports raw and post-clipping effective sample size, maximum and
mean importance weight, clipping fraction, overlap status, resampling unit, and
confidence method. Clipping trades variance for bias and never replaces the raw
overlap diagnostics. IPS and SNIPS raise ``OverlapError`` at zero support. DR
may return its direct-model component, marked ``model_only``.

The default percentile bootstrap treats events as independent. Supply cluster
IDs, such as user, session, or trajectory IDs, when events within a cluster are
dependent; PyMAB then resamples whole clusters. Reward predictions supplied to
DR must be out-of-fold or otherwise independent of the evaluated event.

.. automodule:: pymab.offline
   :members:
   :show-inheritance:
