Policy assumptions
==================

Use policies only with reward and environment assumptions they support. PyMAB
validates categorical incompatibilities before a run; theoretical assumptions
such as stationarity remain the experimenter's responsibility.

.. list-table:: Main policy families
   :header-rows: 1

   * - Family
     - Reward support
     - Environment
     - Evaluation objective
   * - Greedy, epsilon-greedy, softmax, gradient
     - finite real rewards
     - usually stationary
     - cumulative reward/regret
   * - UCB, MOSS, KL-UCB
     - UCB and MOSS require sub-Gaussian rewards with ``reward_scale`` set to a
       valid noise bound; KL-UCB is binary
     - stationary unless explicitly sliding-window, discounted, or change-point
     - cumulative reward/regret
   * - Bernoulli Thompson/Bayesian UCB
     - binary observations
     - stationary unless explicitly sliding-window or discounted
     - cumulative reward/regret
   * - Gaussian Thompson/Bayesian UCB
     - approximately Gaussian observations with known precision
     - stationary
     - cumulative reward/regret
   * - LinUCB, linear Thompson
     - finite real rewards with approximately linear conditional means
     - contextual
     - contextual cumulative reward/regret
   * - Logistic contextual
     - binary observations with a logit-linked conditional mean
     - contextual
     - contextual cumulative reward/regret
   * - EXP3
     - rewards in ``[0, 1]``
     - adversarial or non-stochastic
     - cumulative reward/regret
   * - Successive/median elimination
     - bounded or sub-Gaussian observations, depending on the guarantee used
     - stationary
     - best-arm identification and simple regret

Expected regret is appropriate only in simulations where true arm means are
known. Offline logged-data analysis must use replay or off-policy estimators and
must report overlap diagnostics.

Window and numerical semantics
------------------------------

Sliding-window UCB and Bernoulli Thompson Sampling retain observations from the
most recent ``window_size`` global decisions. An arm's old observation expires
even if that arm has not been selected again. Discounted variants instead apply
exponential forgetting on every update.

EXP3 requires rewards in ``[0, 1]``, ``gamma`` in ``(0, 1]``, and a learning
rate in ``(0, 1]``. It stores log weights and raises on non-finite numerical
state; it never silently resets learned weights. MOSS additionally requires
``horizon >= n_arms``.

The concentration guarantees used by successive and median elimination assume
bounded or appropriately sub-Gaussian rewards. Runtime reward-domain checks
cannot verify stationarity, linear realizability, independence, or a correctly
chosen noise scale; these remain study-design responsibilities.

Native numerical implementation
-------------------------------

All 27 built-in policy classes use Rust-owned learned state when the compiled
extension is available. Contextual policies store contiguous matrices and use
factorization/solve operations instead of explicit matrix inversion. Public
state arrays are read-only snapshots so Python and Rust state cannot diverge.
The private pure-Python implementations remain the reference backend for parity
testing and custom-component experiments.
