PyMAB Arcade
============

PyMAB Arcade is an interactive companion to this documentation. It covers all
27 concrete policies exported by ``pymab.policies`` through short,
reproducible runs. Each lesson exposes the real constructor settings, policy
state, reward environment, and equivalent Python behind the decisions.

`Launch PyMAB Arcade <../demo/>`_

The Arcade is a static website. It has no account, server-side application,
analytics, or telemetry. Lesson completion and a small set of display
preferences remain in versioned browser local storage. Python executes locally
inside a Web Worker using a self-hosted Pyodide runtime and the wheel built from
the same repository commit.

Learning model
--------------

Each lesson offers a guided run, a fixed-seed parameter challenge, and free
play. The animated chamber is intentionally stateless: every crossroads is one
independent bandit round. Only the policy's learned estimates and cumulative
metrics carry forward.

This distinction matters. A contextual bandit such as LinUCB observes the
current light, echo, and tide before choosing a gate, but it does not navigate a
path or optimize a delayed sequence of rewards. Problems where an action changes
the next state belong to stateful reinforcement learning, not this demo.

Policy families and environments
--------------------------------

The mission atlas groups policies into Foundations, Optimism, Bayesian
methods, Changing environments, Best arm identification, Adversarial rewards,
and Contextual bandits. The grouping helps you compare related methods. Every
policy still has its own route and runs its own checked-out Python class.

Free play keeps policy parameters separate from the reward environment. You
can edit Bernoulli probabilities, Gaussian means and noise, nonstationary phase
schedules, adversarial reward tables, or contextual coefficient matrices. Best
arm lessons report a final recommendation. Other lessons emphasize cumulative
reward and expected regret.

Browser requirements
--------------------

Use a current release of Chromium, Firefox, or Safari with WebAssembly and
module Web Worker support. The first lesson load downloads the Python runtime,
NumPy, and PyMAB; subsequent loads can reuse the browser cache. A compatibility
message is shown when the required browser features are unavailable.

``BernoulliBayesianUCBPolicy`` needs SciPy for its posterior confidence bound.
The Arcade loads the pinned local SciPy package only when that lesson first
starts. The package is then cached for later runs.

Local development
-----------------

Install Node.js 24, npm, Python 3.12, and ``uv``. Then run:

.. code-block:: console

   make web-sync
   cd web && npm run dev

The production build creates and verifies all Python assets before Vite runs:

.. code-block:: console

   make web-build
   make web-e2e

To assemble and preview the complete GitHub Pages site with its production URL
structure, run:

.. code-block:: console

   make pages-build
   make pages-serve

The preview exposes the landing hub at ``/pymab/``, Arcade at
``/pymab/demo/``, and this documentation at ``/pymab/docs/``. The Pages build
also creates redirects from the former root documentation URLs into
``/pymab/docs/``.

Generated wheels, Pyodide files, caches, reports, and browser artifacts stay
under ignored directories in ``web/`` and are never included in the Python
package.

Python Lab safety
-----------------

The editable Lab starts a separate disposable worker. A run is stopped after
five seconds, combined output is capped at 64 KiB while it is written, and Stop
or navigation destroys the worker. It is a learning sandbox, not a security
boundary for running untrusted third-party code.
