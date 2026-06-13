Real-World Examples
===================

The ``examples/real_world_scenarios.py`` script contains copyable templates for
common bandit use cases:

Recommendation
   Compare recommendation modules using Bernoulli click rewards.

Ad Allocation
   Use ``LinearContextualEnvironment`` and ``LinUCBPolicy`` when the best ad
   depends on user segment or request features.

Pricing
   Simulate revenue-like continuous rewards across candidate price points.

Clinical Trials
   Use Bernoulli rewards and Thompson Sampling for binary treatment outcomes.

Proxy or Server Selection
   Treat latency savings, success rate, or quality score as rewards for routing
   decisions.

Non-Stationary Demand
   Compare stationary policies with sliding-window policies under abrupt shifts
   or gradual drift.

Run the examples with:

.. code-block:: bash

   python examples/real_world_scenarios.py

Each scenario intentionally includes a simple baseline. In your own experiments,
keep that habit: a sophisticated policy should beat a random or simple policy
before it is trusted.
