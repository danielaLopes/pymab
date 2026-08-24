Decision Guide
==============

Which Policy Should I Use?
--------------------------

Use this as a starting point, then benchmark with your reward assumptions.
For a visual introduction to epsilon-greedy and LinUCB, work through
:doc:`arcade` before comparing them in a full experiment.

``RandomPolicy``
   Baseline only. Include it in benchmarks so improvements have a sanity-check
   floor.

``EpsilonGreedyPolicy``
   Simple baseline when rewards are stationary and interpretability matters.
   It is easy to explain, but it keeps exploring forever and can waste traffic.

``UCBPolicy``
   Strong default for stationary Gaussian-like rewards. It is deterministic
   apart from tie-breaking, balances exploration through confidence bonuses,
   and is a good first policy for recommendation, pricing, or routing tests.

``BernoulliThompsonSamplingPolicy``
   Use for binary outcomes such as click/no-click, conversion/no-conversion,
   adverse-event/no-adverse-event, or trial success/failure. It is often a
   strong default for product experiments and clinical-trial style simulations.

``GaussianThompsonSamplingPolicy``
   Use when rewards are continuous and approximately Gaussian, such as revenue,
   latency savings, or quality scores.

``SlidingWindowUCBPolicy`` and ``DiscountedUCBPolicy``
   Use when the best arm changes over time. Sliding windows are easier to
   reason about when you have a natural recency horizon; discounting is smoother
   when older observations should decay continuously.

``LinUCBPolicy``
   Use when each action has features and the best arm depends on context, such
   as user segment, request geography, ad slot, server load, or patient cohort.

``LinearThompsonSamplingPolicy``
   Use for contextual problems where posterior sampling is preferred over UCB
   optimism. It can be more exploratory than LinUCB.

Which Environment Matches My Problem?
-------------------------------------

``BanditEnvironment``
   Use for classic K-armed bandits where each arm has one current expected
   reward. This fits A/B/n testing, arm-level pricing, model routing, and
   non-contextual recommendation tests.

``LinearContextualEnvironment``
   Use when each decision includes features and rewards are approximately
   linear in those features. This is the right starting point for user-specific
   recommendations, ad allocation by segment, server selection by request
   metadata, or cohort-dependent treatment effects.

``StationaryDynamics``
   Use when expected rewards do not move during the experiment.

``GradualDrift``
   Use for seasonality, slowly changing demand, model decay, or competitor
   effects.

``AbruptShift``
   Use for product launches, outages, pricing changes, supply changes, or other
   discrete events that alter rewards.

``RandomArmSwap``
   Use as a stress test for policies that claim to handle non-stationarity.

How To Judge Winners
--------------------

Prefer expected regret when the simulated ground truth is known. It is less
noisy than realized reward and directly answers how much value a policy left on
the table. Also check realized reward and optimal-action rate:

- Lowest cumulative regret: best decision quality.
- Highest total reward: best observed business outcome.
- Highest optimal-action rate: easiest diagnostic for learning behavior.
- Narrow confidence interval: more stable result across random seeds.

For public examples, always include ``RandomPolicy`` and at least one simple
baseline. A policy that cannot beat random or greedy in a controlled simulation
should not be trusted in a production experiment.
