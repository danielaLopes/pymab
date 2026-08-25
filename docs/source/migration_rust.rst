Migrating to the Rust backend
=============================

PyMAB now keeps the public Python policy classes while storing built-in policy
state and running compatible experiments in Rust. Existing constructors,
updates, recommendations, result shapes, and statistical APIs remain available.

Backend selection
-----------------

``ExperimentConfig.backend`` accepts three values:

``"auto"``
   Use Rust when the environment, reward model, dynamics, context provider, and
   every policy are built-ins. Fall back to Python when custom callbacks are
   present. This is the default.

``"rust"``
   Require native execution. An incompatible custom component raises one
   aggregated compatibility error before the experiment starts.

``"python"``
   Run the private pure-Python behavioral reference. This is useful for parity
   investigation, not as the performance path.

Random streams
--------------

The Rust runner uses versioned Blake2b-derived ChaCha12 streams. It preserves
replay within the Rust backend, common-versus-independent reward coupling, and
policy-order isolation. Rust and Python backends intentionally publish different
RNG scheme identifiers, so identical seeds do not promise identical stochastic
samples across languages. Deterministic single-arm traces and all learned-state
transitions are covered by shared parity fixtures.

Custom components
-----------------

Custom Python policies, reward models, dynamics, and context providers continue
to work through ``backend="auto"`` or ``backend="python"``. Use
``experiment.backend_compatibility()`` to inspect every reason a configuration
cannot enter the native loop. No Python callback is invoked from the Rust hot
loop.

Packaging
---------

Published Python wheels contain the private native extension for CPython
3.11--3.14 on supported Linux, macOS, and Windows platforms. Source builds need
Rust 1.83 or newer. Rust applications can depend directly on the matching
``pymab`` crate; Python and Rust artifacts share one release version.
