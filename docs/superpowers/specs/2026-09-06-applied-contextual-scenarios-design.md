# Applied contextual scenarios

## Goal

Add two interactive scenarios to PyMAB Arcade:

1. a single-slot recommendation system;
2. proportionate defensive verification for web requests.

Both scenarios must use real PyMAB policies in the browser worker. They must
also state where the contextual bandit model fits and where it stops fitting.
The work includes the previously approved LinUCB tooltip copy, the slender
Christmas star for Star Path, and a repository guide that explains both use
cases in depth.

## Product structure

The home page gains an Applied scenarios section after the policy atlas. Its
cards link to these routes:

- `#/scenario/recommendations`
- `#/scenario/defensive-verification`

Scenarios are not inserted into the 27-policy catalog. A scenario is an
application of a policy, not another policy. Keeping separate registries
preserves the catalog's one-to-one relationship with public Python classes.

Policy lessons continue to use `startLesson`. Applied scenarios use a new
`startScenario` worker request. Both return the same validated snapshot shape,
with presentation metadata that removes the current assumption that every
action is a Moon, Sun, or Star path.

## Shared scenario experience

Each scenario reuses the existing application shell, setup panel styling,
progress display, decision history, run controls, outcome explanation,
developer inspector, and generated Python view.

The scenario header explains the real decision before introducing the policy.
The current context appears above the action columns. Each completed round
records:

- the context known before the decision;
- the value assigned to every available action;
- the selected action;
- the observed outcome for that action;
- the scalar reward used for learning;
- a short explanation of the update.

Unselected outcomes remain unknown. The simulation may calculate their
expected rewards to measure regret, but the history board never presents them
as observations.

Guided mode uses a fixed seed and explanation sequence. Challenge mode uses a
separate fixed seed and success threshold. Free Play exposes the scenario's
policy parameters, seed, and environment coefficients.

## Scenario 1: recommendations

### Decision

One visitor arrives and the system chooses one recommendation slot. The three
actions are:

- Article
- Product
- Tutorial

The context contains an intercept plus three binary features:

- Visitor: new or returning
- Engagement: low or high
- Visit: weekday or weekend

The context is shared by all three actions. Each action has its own coefficient
vector, so the same visitor can receive different predicted click
probabilities for Article, Product, and Tutorial.

### Policy and environment

The scenario uses `LogisticContextualBanditPolicy` with
`LogisticContextualEnvironment`. This is a direct match because the outcome is
binary and the expected reward follows a logistic link.

The selected recommendation receives reward `1` for a click and `0` for no
click. The policy updates only the selected action. Epsilon controls random
exploration, the learning rate controls the online gradient step, and L2
regularization limits coefficient growth.

The default coefficients create understandable, non-deterministic patterns:

- returning visitors with high engagement tend to prefer Product;
- new visitors with low engagement tend to prefer Tutorial;
- Article remains competitive across several contexts;
- weekend context changes the relative ordering without making any action
  universally best.

### Presentation

The history columns are Article, Product, and Tutorial. Context chips appear in
the round rail. Each cell shows the predicted click probability used for that
decision. The selected cell shows Click or No click.

The inspector includes predicted probabilities, the exploration branch, the
context vector, coefficient estimates before and after the update, and the
generated PyMAB example.

### Fit boundary

This is a natural contextual bandit problem because there is one decision, a
small action set, context known before selection, and prompt binary feedback.

The lesson does not claim to solve ranked lists, page-wide personalization,
delayed purchases, long sessions, inventory constraints, or long-term user
satisfaction. Those problems need additional models or a different decision
framework.

## Scenario 2: defensive verification

### Decision

An upstream risk system has already placed a request in an uncertain middle
range. The bandit chooses one of three approved actions:

- Allow
- Light check
- Strong verification

The context contains an intercept plus:

- a normalized request risk score;
- Account: new or established;
- Endpoint: routine or sensitive.

Requests that are clearly safe bypass the scenario. Requests that meet a hard
block rule are handled by deterministic security controls. The bandit never
experiments with hard blocking.

### Policy and environment

The scenario uses `LinUCBPolicy` with a linear contextual simulator. The reward
is a real-valued utility, not a probability:

- legitimate traffic proceeding successfully earns positive utility;
- stopping an abusive request earns positive utility;
- verification adds a friction cost;
- abandonment after a check adds a larger penalty;
- allowing abusive traffic adds a large penalty.

The simulator resolves a latent request type and an action outcome, then
returns the resulting utility. Its expected utility is constructed to be
approximately linear in the supplied context so LinUCB is a defensible teaching
model. The guide must state that a production anti-abuse system will rarely be
exactly linear.

### Presentation

The history columns are Allow, Light check, and Strong verification. Each cell
shows its LinUCB prediction, exploration bonus, and total UCB score. The
selected cell shows a plain outcome such as Allowed safely, Abuse passed,
Passed light check, or Abandoned during verification. It also shows the scalar
utility used in the update.

The inspector exposes the score decomposition, context vector, learned
coefficients, confidence matrices, and generated PyMAB example.

### Safety and fit boundary

The lesson is about allocating defensive friction. It does not model detection
evasion, human impersonation, fingerprint changes, or ways to bypass a
security system.

The scenario is a natural contextual bandit only inside a decision band where
all actions are reversible and approved. A production version would also need
delayed feedback, safety constraints, audit logs, conservative rollout rules,
privacy review, fairness checks, and a reliable source of later abuse labels.
The current Arcade remains an educational simulation.

## Presentation metadata

The snapshot gains a `presentation` object:

```text
presentation:
  experienceKind: policy | scenario
  experienceId: string
  arms: [{ name, shortName, symbol, symbolKind }]
  rewardPresentation: binary | numeric | utility
  positiveOutcomeLabel: string
  zeroOutcomeLabel: string
```

Existing policy lessons receive the current Moon, Sun, and Star defaults.
Scenario sessions supply their own values. `symbolKind` selects an approved
frontend icon rather than sending arbitrary markup across the worker boundary.

The decision history adapter reads presentation metadata from the snapshot.
It does not import the scenario registry or infer labels from a route.

## Worker and Python boundaries

The TypeScript protocol adds a validated `startScenario` request with scenario
ID, mode, seed, and parameters. The Python entrypoint validates the request and
constructs either `RecommendationScenarioSession` or
`DefensiveVerificationScenarioSession`.

Scenario sessions share the existing lifecycle: start, step, run to end, reset,
dispose, snapshot, and generated code. Random streams remain named and seeded
independently for context, action, latent outcome, and reward. Reset must replay
every public value exactly.

The recommendation session uses the public
`LogisticContextualBanditPolicy`. The defensive session uses the public
`LinUCBPolicy`. Diagnostics are computed from public policy state and checked
against public scoring methods where those methods exist.

## LinUCB lesson improvements

The existing cue tooltips keep their current first sentence and add a
default-environment explanation:

- Light: "Light can be red or blue. The policy sees it before choosing a path.
  In the default environment, red strongly favors Moon and blue strongly
  favors Sun."
- Echo: "Echo can be low or high. The policy sees it before choosing a path. In
  the default environment, high slightly favors Moon and Sun, while low
  strongly favors Star."
- Tide: "Tide can be low or high. The policy sees it before choosing a path. In
  the default environment, low strongly favors Moon and high strongly favors
  Sun. Tide has a smaller effect on Star."

The qualifier is required because Free Play can replace the default
coefficients.

Star Path uses the approved slender Christmas star SVG. The relic marker keeps
the compact gold `✦`, so path identity and reward outcome no longer share the
same shape.

## Repository guide

Add `docs/contextual-bandit-use-cases.md`. It is a repository reference and is
not added to the Sphinx navigation in this change.

The guide covers:

- a practical contextual bandit fit checklist;
- detailed mappings for both scenarios;
- why the selected PyMAB policies match their rewards;
- synthetic environment construction;
- baselines and experiment matrices;
- online and offline evaluation;
- rollout gates and monitoring;
- implementation gaps;
- privacy and safety constraints;
- conditions that require reinforcement learning, constrained optimization,
  supervised prediction, ranking, or deterministic rules instead.

## Error handling

Unknown scenario IDs, invalid parameters, incompatible reward domains, invalid
coefficient shapes, and non-finite utilities return validated worker errors.
The React route uses the existing error recovery component.

Scenario navigation disposes the previous worker session. Stale responses are
rejected by the existing request and session correlation logic.

## Accessibility and responsive behavior

Action columns use text labels and distinct icons. Outcomes are never conveyed
only by color. Context chips and selected outcomes are present in row-level
accessible descriptions.

Tooltips remain keyboard accessible, but the scenario's essential context is
also visible as text. Both scenario boards must work without page-level
horizontal overflow at 320 CSS pixels.

## Testing and acceptance

The work is complete when:

1. The home page shows two Applied scenarios cards without changing the count
   or identity of the 27 public policy lessons.
2. Both scenario routes support Guided, Challenge, and Free Play modes.
3. Both sessions use the real public PyMAB policy classes.
4. Seeded resets reproduce every context, action, outcome, reward, diagnostic,
   and total.
5. Recommendation predictions equal the public logistic policy output before
   selection.
6. Defensive-verification UCB decompositions equal the public LinUCB scores.
7. Unselected outcomes are never displayed as observations.
8. Scenario-specific action, context, and reward labels appear in the history
   board and accessible descriptions.
9. Hidden environment coefficients remain hidden until the existing reveal
   rules allow them.
10. Existing policy routes retain their current behavior and presentation.
11. The LinUCB cue tooltips include the approved qualified explanations.
12. Star Path uses the slender Christmas star while relics retain `✦`.
13. The repository guide describes both use cases, their experiments, and their
    limits.
14. Python tests, TypeScript tests, worker protocol tests, browser tests,
    accessibility checks, text checks, and the combined Pages build pass.
15. Published copy contains no em dash.
