# All-policy PyMAB Arcade

## Goal

Expand PyMAB Arcade from two lessons to all 27 concrete policies exported by
`pymab.policies`. Every policy must execute the checked-out Python class in the
browser, expose its real constructor settings, explain its decision state, and
produce equivalent Python.

The Arcade groups related policies into shared game worlds. The grouping helps
people browse the library. It does not merge policies or substitute one policy
for another.

## Policy inventory

The public catalog contains these concrete policies:

| Family | Policies |
| --- | --- |
| Foundations | `RandomPolicy`, `GreedyPolicy`, `EpsilonGreedyPolicy`, `DecayingEpsilonGreedyPolicy`, `SoftmaxPolicy`, `GradientBanditPolicy` |
| Optimism | `UCBPolicy`, `KLUCBPolicy`, `MOSSPolicy` |
| Bayesian methods | `BernoulliThompsonSamplingPolicy`, `GaussianThompsonSamplingPolicy`, `BernoulliBayesianUCBPolicy`, `GaussianBayesianUCBPolicy` |
| Changing environments | `SlidingWindowUCBPolicy`, `DiscountedUCBPolicy`, `SlidingWindowBernoulliThompsonSamplingPolicy`, `DiscountedBernoulliThompsonSamplingPolicy`, `ChangePointUCBPolicy`, `CUSUMUCBPolicy`, `PageHinkleyUCBPolicy` |
| Best arm identification | `SuccessiveEliminationPolicy`, `MedianEliminationPolicy` |
| Adversarial rewards | `EXP3Policy` |
| Contextual bandits | `LinearEpsilonGreedyPolicy`, `LinUCBPolicy`, `LinearThompsonSamplingPolicy`, `LogisticContextualBanditPolicy` |

`Policy`, `ActionValuePolicy`, and `ContextualPolicy` are abstract base classes
and do not appear as lessons.

## Navigation and visual model

The home page becomes a seven-family mission atlas. Each family card explains
the learning problem and lists its policies. A searchable all-policy view gives
experienced users direct access by class name.

Every policy has a stable route under `/lesson/<policy-id>`. Existing
`/lesson/epsilon-greedy` and `/lesson/linucb` routes remain valid.

The shared worlds are:

- The Ancient Gates for stationary foundations.
- The Observatory for optimistic confidence methods.
- The Oracle Vault for posterior methods.
- The Shifting Tides for nonstationary methods.
- The Trial Grounds for best arm identification.
- The Trickster's Arena for adversarial rewards.
- The Labyrinth of Signals for contextual methods.

The three portal names remain Moon, Sun, and Star across worlds. Shared names
make policy comparisons easier. Each world adds only the information its
policies observe or maintain:

- Stationary worlds show three portals and per-arm decision values.
- The Shifting Tides adds a round timeline and marks known environment phases.
  It does not reveal future probabilities during Guided or Challenge runs.
- The Trial Grounds marks active and eliminated portals. Its result emphasizes
  the recommended portal and simple regret rather than cumulative reward.
- The Trickster's Arena shows EXP3 sampling weights and the reward assigned to
  the selected portal. The unselected counterfactual reward vector stays hidden
  until the debrief.
- The Labyrinth keeps the Light, Echo, and Tide signal strip. Linear policies
  show learned reward estimates. Logistic contextual policy shows predicted
  reward probabilities.

## Catalog-driven architecture

Replace two-policy conditionals with one catalog on each side of the worker
boundary.

The TypeScript catalog owns presentation metadata:

- policy ID, Python class name, label, family, badge, route, and concise copy;
- reward model and learning objective;
- guided and challenge seeds, horizons, and targets;
- parameter definitions with field type, default, range, step, and help text;
- environment editor type;
- diagnostic renderer type.

The Python catalog owns execution metadata:

- policy class and constructor adapter;
- accepted parameters and cross-field validation;
- environment type;
- action selection and update adapter;
- policy-state diagnostics;
- generated-code adapter.

Tests compare both catalogs against the public concrete exports. A missing or
extra policy is a failure. Policy IDs are explicit constants, not names derived
at runtime, so routes and saved progress stay stable.

## Run configuration

The current single numeric `parameter` becomes a `parameters` record. The setup
panel renders fields from the selected policy's definitions.

Supported field controls are:

- linked slider and number field for bounded continuous values;
- number field for wide-range values and integer counts;
- switch for Boolean values;
- selector for closed string choices;
- optional number field for values such as EXP3 `learning_rate`.

`n_arms` stays fixed at three because the game has three portals. Contextual
policies use four features: an intercept plus Light, Echo, and Tide.

All other public constructor settings are visible. Advanced numerical settings,
including KL-UCB tolerance and iteration count, appear in a collapsed Advanced
section. MOSS planning horizon is the run horizon and is shown as a read-only
constructor value. LinUCB `l2`, previously fixed in the worker, becomes an
editable field.

Field validation follows the Python constructor contract. Cross-field rules,
such as minimum epsilon not exceeding initial epsilon, run in TypeScript for
immediate feedback and again in Python before a session starts. Invalid drafts
never replace the active session.

Guided and Challenge use curated parameter presets. Free play begins with the
public class defaults and allows edits. Users can restore defaults for the
selected policy.

## Environment models

Policy parameters and environment settings remain separate in the worker
request.

### Stationary Bernoulli

Used by binary stationary policies and as the default teaching environment for
general stationary policies. Free play exposes the existing three editable
portal probabilities. Seed-generated values use one low, one medium, and one
high band. Guided and Challenge keep fixed hidden values until the debrief.

### Stationary Gaussian

Used by Gaussian Thompson Sampling and Gaussian Bayesian UCB. Each portal has a
mean. One shared standard deviation controls reward noise. Free play exposes all
four values. The run shows numeric treasure values rather than binary relic or
empty outcomes.

### Changing Bernoulli

Used by sliding-window, discounted, and change-detection policies. The
environment has two or three deterministic phases. Each phase contains three
probabilities and a start round. Free play exposes the schedule with validation
for ordered phase boundaries. Guided and Challenge show the current phase but
hide future probabilities.

### Contextual linear and logistic

Each round draws Light, Echo, and Tide before the policy chooses. Linear methods
use finite numeric rewards generated from a linear conditional mean plus fixed
noise. Logistic contextual policy uses a Bernoulli reward with a logit-linked
probability. Free play exposes the three-by-four coefficient matrix in an
advanced environment editor. Seed-generated presets remain the default.

### Adversarial

EXP3 receives a deterministic reward matrix with one row per round and one
column per portal. Values are from 0 to 1. Guided runs use a rotating strategy
that punishes a policy that settles too quickly. Free play offers presets and an
editable table. Expected regret compares the selected reward with the best
reward available in that round.

### Best arm identification

Successive Elimination uses bounded numeric rewards. Median Elimination uses
unit-interval rewards. The environment has three fixed means and a sampling
budget. Completion reports the policy recommendation, whether it matches the
true best portal, and simple regret. Reward totals remain secondary.

## Generic session engine

A session composes three independent units:

1. An environment generates the current context, potential rewards, and true
   expected values from named deterministic random streams.
2. A policy adapter asks the real PyMAB object to choose, then updates it with
   only the selected outcome.
3. A lesson presenter converts environment and policy state into a JSON-safe
   snapshot without exposing hidden information early.

The environment never reads the selected action before it creates potential
outcomes. Context and nonstationary phase may depend on the round number, but an
action never changes the next state. The Arcade therefore remains a bandit demo,
not a reinforcement learning maze.

The worker protocol carries:

- `policyId` instead of a two-value lesson ID;
- mode, seed, parameter record, and typed environment object;
- family and objective in snapshots;
- numeric reward values rather than assuming binary rewards;
- a generic recommendation field for best arm policies;
- JSON-safe diagnostics whose renderer is selected by catalog metadata.

Old persisted version 1 data is migrated. Existing completion and recent values
for epsilon-greedy and LinUCB are retained. New policies receive catalog
defaults. Persistence moves to version 2 and stores parameter records by policy.

## Diagnostics

Diagnostics are calculated from state immediately before selection and after
update where both are educationally useful.

The standard diagnostic shapes are:

- estimates and counts for Random, Greedy, and epsilon methods;
- exploration probability or action probabilities for decaying epsilon,
  Softmax, Gradient, and EXP3;
- estimates, confidence bonuses, and indices for UCB variants;
- posterior parameters, posterior means, samples, or quantile bounds for
  Thompson Sampling and Bayesian UCB;
- effective counts, window contents, discounted counts, or change alarms for
  nonstationary methods;
- active set, confidence bounds, phase quota, and current recommendation for
  best arm methods;
- predicted means or probabilities, sampled scores, uncertainty bonuses, and
  learned coefficients for contextual methods.

The inspector labels values according to their meaning. A linear prediction is
not called a probability. A posterior draw is not called an estimate. Raw
validated diagnostics remain available in a disclosure panel.

## Guided explanations and challenges

Every policy has at least four guided explanation moments:

- its initial state or prior;
- its first decision rule;
- its first meaningful update;
- the family-specific tradeoff shown later in the run.

Repeated rounds use a short policy-specific fallback explanation. Copy names
the actual class behavior and avoids claims that one seeded run proves a
parameter is generally best.

Challenge scoring matches the policy objective:

- cumulative-reward policies use a reward target and expected-regret ceiling;
- best arm policies must recommend the true best portal within the budget and
  meet a simple-regret threshold.

Challenge presets are calibrated against the checked-out implementation and
pinned by golden tests. Free play never claims pass or fail.

## Generated Python and browser dependencies

Every snapshot contains a complete example that uses the public PyMAB class and
recreates the active environment, named seeds, horizon, policy parameters, and
metrics. Generated examples must parse and reproduce the browser session's final
metrics in CPython tests.

`BernoulliBayesianUCBPolicy` imports SciPy when selecting an action. The static
runtime therefore self-hosts the Pyodide SciPy package and its verified
dependencies. NumPy and SciPy artifacts are pinned by the existing manifest and
integrity checks. The loading screen names SciPy only when it is needed, and the
worker caches the loaded package for later lessons.

## Error handling

The TypeScript schema rejects unknown policy IDs, malformed environments, and
non-finite parameters before sending a request. Python repeats semantic checks
and lets the public constructor enforce its own final invariants.

Policy or environment failures return the existing structured worker errors.
The current session remains usable if a replacement session fails to start.
Unsupported optional runtime packages produce a direct message naming the
policy and dependency.

Generated environment tables have bounded horizons and three portals. The UI
does not allow a matrix large enough to create an unreasonable worker message.

## Accessibility and responsive behavior

Family cards and policy rows are keyboard links. Search results announce their
count. Parameter groups use fieldsets and legends. Sliders always have matching
number fields. Diagnostic charts have text or table equivalents.

Color never carries active, eliminated, selected, changed, or recommended state
by itself. Tooltips open on hover and keyboard focus. The family atlas, policy
picker, setup panel, chamber, timeline, environment tables, and inspector must
fit at 320 CSS pixels without horizontal page overflow.

Reduced motion applies to all new world effects.

## Documentation

The root hub and Arcade documentation state that all concrete public policies
are available. The policy decision guide links each class to its Arcade route.
The Arcade page explains the seven environment models, local execution, SciPy
load behavior, and the difference between cumulative and best arm objectives.

All new interface and documentation copy is checked with the human-writing
rules. Website prose contains no em dash characters.

## Tests and completion evidence

### Catalog coverage

- Import `pymab.policies.__all__`, remove the three abstract bases, and prove the
  Python catalog contains exactly the remaining 27 class names.
- Prove the TypeScript catalog contains the same 27 stable policy IDs through a
  checked generated manifest or a direct fixture comparison.
- Prove every catalog entry has copy, seeds, parameter definitions, environment,
  diagnostic renderer, and generated-code support.

### Python

- Instantiate every policy through its adapter with default settings.
- Start, step, reset, and complete Guided, Challenge, and Free play for every
  policy.
- Verify deterministic replay for every policy and environment family.
- Verify reward-domain, parameter, cross-field, and environment validation.
- Pin selected guided and challenge trajectories for every family.
- Verify every diagnostic is JSON-safe and does not reveal hidden truth early.
- Parse and execute generated Python for every policy, then compare final
  metrics and recommendation.
- Keep demo bridge branch coverage at or above the existing 95 percent gate.

### TypeScript and components

- Validate parameter records and every control type.
- Test version 1 to version 2 persistence migration.
- Test family filtering, policy search, routes, default restoration, draft
  isolation, environment editors, and diagnostic renderers.
- Test that all 27 policy routes render the expected class and constructor.

### Browser and accessibility

- Open every family and every policy route in Chromium.
- Complete at least one policy from each family in Chromium, Firefox, and
  WebKit.
- Run an accessibility scan on the atlas and one active lesson per family.
- Verify generated code can open in the Python Lab for every family.
- Verify no horizontal overflow at 320 CSS pixels for every distinct world and
  environment editor.
- Add visual snapshots for the atlas, each family picker, and each distinct
  chamber state. Review snapshots before committing them.

### Build and routes

- Run formatting, linting, TypeScript checks, unit tests, demo coverage, browser
  tests, and the production bundle budget check.
- Assemble the combined Pages artifact.
- Verify `/pymab/`, `/pymab/demo/`, and `/pymab/docs/` return successful
  responses and link to one another.
