# PyMAB Interactive Demo Design

Date: 2026-08-24
Status: Approved design

## Objective

Build a static educational website, **PyMAB Arcade**, in the PyMAB repository.
Its first release teaches `EpsilonGreedyPolicy` and `LinUCBPolicy` through the
stateless **Infinite Crossroads** game, while also exposing real PyMAB state and
an editable Python lab for developers.

The site serves two audiences through progressive disclosure:

- Beginners get a guided game, plain-language explanations, and restrained
  controls.
- Developers can open **Inspect PyMAB** to see the exact class, parameters,
  seed, inputs, diagnostics, and equivalent Python, or move the scenario into a
  full code editor.

Both views consume the same results produced by the current PyMAB wheel. The
site must not contain a TypeScript reimplementation of either policy.

## Repository and Product Structure

The demo lives in a sibling `web/` project in the existing repository. It is
not included in the `pymab` wheel and does not change the cadence or dependency
surface of `pip install pymab`.

The application contains four hash-routed surfaces:

- Home/campaign selection.
- `Epsilon Greedy: The Three Ancient Gates`.
- `LinUCB: The Infinite Crossroads`.
- `Lab`, an editable Python playground.

No account, backend, database, cloud API, or analytics is included. Lesson
completion, parameters, seed, reduced-motion choice, and inspector preference
are stored in versioned browser local storage. A corrupt or obsolete value is
discarded and replaced by defaults.

## Learning and Game Loop

Each lesson has three stages.

1. **Guided run:** a fixed seed produces a reproducible, step-by-step sequence.
   The real policy selects each gate. Before and after each selection, the UI
   explains the visible evidence, chosen action, reward, and update.
2. **Parameter challenge:** the player chooses the policy's main exploration
   parameter and sends it through a 20-chamber expedition. The player gets
   three attempts against the same scenario. Success requires both the reward
   and expected-regret target, preventing a lucky but poor strategy from
   passing.
3. **Free exploration:** the player may change the seed and supported
   parameters, step or auto-run, inspect every calculation, reset exactly, and
   open the equivalent code in Lab.

The animation suggests forward travel, but every chamber is an independent
bandit round:

```text
fresh context (when applicable) -> choose one gate -> observe reward -> reset
```

There is no position, inventory, combat state, path-dependent transition, or
route planning. Only the policy's learned parameters and cumulative lesson
metrics persist between chambers.

### Epsilon-greedy lesson

- Three Bernoulli gates have hidden success probabilities `[0.25, 0.50, 0.75]`.
- The guided run uses seed `42`, epsilon `0.20`, and 12 chambers.
- The challenge uses seed `7` and 20 chambers.
- The epsilon choices are `0`, `0.05`, `0.10`, `0.20`, `0.40`, and `0.80`.
- Challenge success requires at least 12 relics and cumulative expected regret
  no greater than `3.25`.
- The visible diagnostics are counts, value estimates, epsilon, current greedy
  set, whether the sampled branch was exploration or exploitation, selected
  gate, observed reward, and cumulative expected regret.

To explain the branch without replacing the real algorithm, the Python bridge
copies the NumPy generator state, inspects the next draw on the copy, then calls
`EpsilonGreedyPolicy.select_action` with the original generator. The diagnostic
therefore describes the exact real selection without consuming or changing its
random stream.

### LinUCB lesson

- Every chamber exposes three binary cues: blue versus red light, high versus
  low echo, and high versus low tide. The numerical feature vector is
  `[1, light, echo, tide]`, with each cue encoded as `-1` or `1`.
- The same feature vector is supplied to all three arms. Each arm learns its own
  parameter vector, matching PyMAB's disjoint `LinUCBPolicy` implementation.
- Rewards are Bernoulli samples from a `LogisticContextualEnvironment` with
  hidden arm parameters:

  ```python
  [[0.1, -1.2,  0.2, -0.8],
   [0.0,  1.0,  0.3,  1.0],
   [0.2,  0.0, -1.1,  0.2]]
  ```

- The guided run uses seed `31415`, alpha `1.0`, `l2=1.0`, and 12 chambers.
- The challenge uses seed `20260824` and 20 chambers.
- The alpha choices are `0.10`, `0.25`, `0.50`, `1.0`, `2.0`, and `4.0`;
  `l2` remains fixed at `1.0` in the lesson UI.
- Challenge success requires at least 10 relics and cumulative expected regret
  no greater than `3.25`.
- The visible diagnostics are the context matrix, learned coefficient vectors,
  predicted mean for each arm, uncertainty bonus, final upper-confidence score,
  selected gate, reward, optimal action after reveal, and regret.

The challenge constants are calibrated against the named PyMAB random streams:
the intended settings pass while deliberately under- or over-exploratory
settings demonstrate failure modes. The UI clearly states that one seeded run
is an illustration, not statistical evidence that a parameter is universally
best.

## Frontend Architecture

Use React, TypeScript, Vite, CSS modules with shared design tokens, semantic
HTML, and responsive SVG. Use SVG and CSS transitions rather than canvas so the
important state remains accessible and testable. Use CodeMirror 6 for Lab,
Vitest plus Testing Library for unit/component tests, and Playwright for real
browser tests. Use npm with a committed lockfile.

The presentation is a dark, atmospheric labyrinth with three persistent gate
identities. Colour is paired with symbols and text. The main lesson surface
contains one dominant chamber view, compact mission progress, and a collapsible
inspector. It does not present a dashboard of unrelated cards. At narrow widths
the inspector moves below the chamber and all three gates remain usable without
horizontal scrolling.

The default beginner layer uses short explanations and hides equations. Inspect
PyMAB reveals the exact numerical state, constructor call, package version,
source commit, and generated code. Changing an advanced parameter resets the
current run after an explicit confirmation so displayed history never mixes
incompatible configurations.

## Python Runtime and Worker Boundary

Build the checked-out source into a pure-Python wheel as part of the web build.
Do not commit generated wheels. Copy the wheel plus a pinned, self-hosted
Pyodide runtime into the static build inputs. Load Pyodide, NumPy, and that wheel
inside a module Web Worker so Python never blocks the UI thread.

The site-specific bridge lives under `web/python/`; it is not part of PyMAB's
public API. It owns the lesson fixtures, policy and environment instances, and
named NumPy generators. It serializes plain JSON-safe data and never returns a
live Python proxy to TypeScript.

The lesson worker accepts request-ID-bearing commands:

- `initialize`
- `startLesson`
- `step`
- `runToEnd`
- `reset`
- `dispose`

The TypeScript response union contains `ready`, `lessonStarted`, `stepCompleted`,
`runCompleted`, and `error`. Every state-changing response includes the request
ID, session ID, lesson ID, configuration, step number, action, reward, totals,
public context, explanation key, and policy-specific diagnostic object. The UI
derives prose from typed explanation keys; Python remains the source of numeric
truth.

The worker uses separate named streams for context, action selection, potential
rewards, and any environment dynamics. Reset reconstructs all objects and
streams from the original seed. A completed challenge response reveals hidden
environment parameters and optimal actions for the debrief; intermediate
challenge responses do not.

## Editable Lab

Lab is a separate route with a CodeMirror editor, Run, Stop, Reset, stdout,
stderr, execution status, and example picker. The initial examples reproduce
the current epsilon-greedy or LinUCB lesson configuration using public PyMAB
APIs. Sending a lesson to Lab preserves its seed and parameters.

Lab uses a separate disposable Web Worker and never shares Python globals with
a lesson. One Pyodide worker is active at a time. A run has a five-second wall
time and a 64 KiB combined output limit. Stop or timeout terminates the worker;
the next run creates a clean worker. Syntax/runtime errors are returned as
structured messages with a concise traceback. The page applies a restrictive
content-security policy, and the worker cannot access the DOM. Cross-origin
connections are disallowed; same-origin static assets remain readable.

## Loading and Failure Behavior

- Show distinct progress for runtime, NumPy, PyMAB wheel, and lesson startup.
- Keep navigation and explanatory content usable while Python loads.
- On initialization failure, show Retry and expandable technical details without
  discarding the chosen lesson.
- Allow only one active request per lesson session. Disable controls while it is
  pending and ignore responses with stale request or session IDs.
- A worker crash recreates the worker and offers to replay the current seed from
  step zero; partially reconstructed policy state is never presented as valid.
- Browsers without WebAssembly or module Worker support receive a compatibility
  explanation rather than a broken game.
- Animation can be skipped and honours `prefers-reduced-motion`; calculation and
  state updates are never coupled to animation completion.

## Verification

Python tests run the bridge directly under CPython and cover lesson creation,
fixture validation, all parameter choices, branch diagnostics, LinUCB score
decomposition, hidden-information timing, seeded snapshots, reset determinism,
and serialization.

TypeScript tests cover message validation, request/session cancellation, lesson
state transitions, local-storage migration, parameter confirmation, loading,
and errors. Component tests cover progressive disclosure and keyboard behavior.

Playwright tests use the real built wheel under real Pyodide. They complete both
guided runs and challenges, verify deterministic reset/replay, compare displayed
diagnostics with worker results, open both generated examples in Lab, and cover
successful execution, syntax error, timeout, Stop, and recovery. Browser checks
run at 360, 736, and 1024 CSS pixels and cover keyboard-only operation, focus
visibility/order, screen-reader names, non-colour status cues, and reduced
motion.

CI adds independent web jobs for formatting, type checking, unit tests,
production build, and Playwright. Existing Python gates remain unchanged. A web
change cannot merge unless both Python and web contracts pass.

## Deployment

Use GitHub Pages with a hash router so all routes work below the repository
subpath without rewrite rules. Pull requests build an inspectable artifact;
production deployment occurs from `main` only after all gates pass. The build
records the PyMAB version and Git commit shown in Inspect PyMAB.

Pin Pyodide and npm dependencies. Serve runtime files and the PyMAB wheel from
the same origin and rely on normal immutable browser caching; offline-first
service-worker behavior is not part of the first release. Link the production
demo from the README and Sphinx documentation only after the deployed browser
smoke test passes.

## Acceptance Criteria

- Both lessons complete end to end on current Chrome, Firefox, and Safari.
- Every action and diagnostic displayed by the game comes from the checked-out
  PyMAB wheel and matches the seeded CPython fixtures.
- The two challenge configurations and thresholds behave as specified.
- Reset and replay produce identical contexts, actions, rewards, and diagnostics.
- The inspector and generated Python match the active lesson configuration.
- Lab runs both generated examples and recovers from broken or non-terminating
  code without corrupting lesson state.
- The interface meets the responsive, keyboard, screen-reader, contrast, and
  reduced-motion requirements above.
- `pip install pymab` gains no web runtime dependency or bundled website asset.

## Explicit Non-goals

- Policies beyond epsilon-greedy and LinUCB.
- Accounts, synchronization, leaderboards, multiplayer, telemetry, or analytics.
- Persistent maze navigation or general reinforcement learning.
- User-authored package installation or unrestricted network access in Lab.
- Offline-first support, localization, or native mobile applications.
