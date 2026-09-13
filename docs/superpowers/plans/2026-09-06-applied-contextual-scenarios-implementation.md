# Applied contextual scenarios implementation plan

Date: 2026-09-06

## Goal

Implement the approved [applied contextual scenarios design](../specs/2026-09-06-applied-contextual-scenarios-design.md).

The release adds two real PyMAB browser scenarios:

- single-slot recommendations using `LogisticContextualBanditPolicy`;
- proportionate defensive verification using `LinUCBPolicy`.

It also improves the existing LinUCB tooltips, replaces the Star Path marker
with the approved slender Christmas star, and adds a detailed repository guide.

The scenarios must remain separate from the public policy catalog. They reuse
the Arcade's session lifecycle and visualization components through validated
presentation metadata.

## Task 1: Add scenario and presentation contracts test-first

Update:

- `web/src/engine/protocol.ts`
- `web/python/pymab_demo/protocol.py`
- `web/src/engine/WorkerClient.test.ts`
- protocol fixtures used by component tests

Add a `startScenario` request with:

- `scenarioId`: `recommendations` or `defensive-verification`;
- `mode`, `seed`, and scenario policy parameters;
- an optional validated environment.

Add snapshot presentation metadata with three safe icon identifiers, action
names, short names, reward presentation, and outcome labels. Keep defaults for
existing policy lessons so old snapshots stay valid during migration.

Tests must reject unknown scenarios, unknown icon identifiers, missing action
labels, non-finite parameters, and malformed environment values.

Verification:

```bash
cd web
npm test -- --run src/engine
npm run typecheck
```

## Task 2: Build deterministic Python scenario sessions test-first

Create:

- `web/python/pymab_demo/scenarios.py`
- `tests/demo/test_scenarios.py`

Update:

- `web/python/pymab_demo/sessions.py`
- `web/python/pymab_demo/entrypoint.py`
- `web/python/pymab_demo/codegen.py`
- `web/python/pymab_demo/fixtures.py`

### Recommendation session

Use four features: intercept, visitor type, engagement, and visit timing. Use
three arms: Article, Product, and Tutorial. Define a finite 3 by 4 coefficient
matrix and sample binary rewards with a logistic link.

Instantiate the public `LogisticContextualBanditPolicy`. Capture predicted
probabilities and the exploration branch before selection. Capture coefficients
before and after update. Use separate named streams for context, action, and
reward.

### Defensive verification session

Use four features: intercept, normalized risk, account state, and endpoint
sensitivity. Use three arms: Allow, Light check, and Strong verification.

Instantiate the public `LinUCBPolicy`. The simulator draws a latent request
type and an action outcome, then maps the outcome to a finite real-valued
utility. Its configured context range must keep each arm's expected utility
close enough to linear for the lesson's stated approximation.

Capture predicted means, confidence bonuses, UCB scores, and matrices before
and after update. Use separate named streams for context, latent request type,
action outcome, action selection, and utility noise if noise is included.

Tests must cover:

- exact public class types;
- valid context and coefficient shapes;
- score decomposition equality with public policy methods;
- selected-arm-only updates;
- hidden truth rules;
- JSON-safe snapshots;
- deterministic reset and golden seeded histories;
- challenge calibration;
- generated examples that parse and reproduce final metrics;
- invalid scenario IDs, parameters, and environments.

Verification:

```bash
uv run pytest tests/demo/test_scenarios.py tests/demo/test_diagnostics.py
uv run mypy src/pymab web/python
uv run ruff check web/python tests/demo
```

## Task 3: Dispatch scenarios through the browser worker

Update:

- `web/python/pymab_demo/entrypoint.py`
- `web/src/engine/lesson.worker.ts`
- `web/src/engine/WorkerClient.ts`
- worker tests

Route `startScenario` through the same initialized Pyodide bridge and session
map as lessons. Preserve one mutation in flight, session correlation, stale
response rejection, disposal, and recovery behavior.

Add a real-Pyodide smoke test for the first round of each scenario. Assert the
policy class, context, selected action, reward, and presentation metadata match
the CPython seeded fixture.

Verification:

```bash
cd web
npm test -- --run src/engine
npm run build
npx playwright test tests/runtime-smoke.spec.ts --project=chromium --grep "scenario"
```

## Task 4: Generalize the decision history presentation

Update:

- `web/src/components/game/decisionHistory.ts`
- `web/src/components/game/DecisionHistoryBoard.tsx`
- `web/src/components/game/decisionHistory.test.ts`
- `web/src/components/game/game.test.tsx`
- `web/src/styles/index.css`

Resolve action names and icons from snapshot presentation metadata. Keep Moon,
Sun, and the new slender Star as the fallback for policy lessons. Add approved
icons for Article, Product, Tutorial, Allow, Light check, and Strong
verification.

Render binary recommendation results as Click or No click. Render defensive
results with their outcome label and signed utility. The row's accessible name
must contain the context, selected action, outcome, and reward.

Update the LinUCB signal tooltips with the approved default-environment copy.
The additional text must remain qualified because Free Play can change the
coefficients.

Tests must prove:

- policy lessons keep their action labels;
- Star Path no longer shares the relic shape;
- recommendation and defensive rows use scenario labels;
- unselected outcomes remain unknown;
- utility values are not rendered as relics;
- tooltip content is keyboard accessible.

## Task 5: Add the scenario catalog, routes, and run state

Create:

- `web/src/catalog/scenarios.ts`
- `web/src/content/scenarios.ts`
- `web/src/routes/ScenarioRoute.tsx`
- focused unit tests

Update:

- `web/src/App.tsx`
- `web/src/routes/HomeRoute.tsx`
- `web/src/components/game/index.tsx`
- `web/src/state/runConfiguration.ts`
- `web/src/state/persistence.ts`

Add an Applied scenarios section after the policy atlas. The cards name the
decision, policy, reward, and fit boundary in plain language.

`ScenarioRoute` reuses the worker client and reducer lifecycle but owns a
scenario configuration type. It supports Guided, Challenge, and Free Play,
including deterministic seeds, parameter controls, environment coefficients,
restart, auto-run, debrief, error recovery, and worker disposal.

Persist recent scenario seed and parameters under a versioned optional field.
Migrate old persistence without losing policy progress.

Do not add the scenarios to policy search results or the 27-policy count.

## Task 6: Add scenario-specific setup and inspector copy

Create or extract focused components where needed:

- `ScenarioSetupPanel`
- `ScenarioContextSummary`
- scenario diagnostic tables

Recommendation setup exposes epsilon, learning rate, L2 regularization, seed,
and the 3 by 4 click coefficient matrix in Free Play.

Defensive setup exposes alpha, L2 regularization, seed, and its environment
coefficients in Free Play. It also displays the fixed utility table and safety
boundary. The hard block action is never configurable.

The developer inspector names the actual PyMAB policy, shows the current
context in human terms, and preserves the validated raw snapshot disclosure.

Use existing shadcn/ui primitives for form controls, disclosures, and tooltips.
Keep the history grid custom because it is a visualization.

## Task 7: Add the repository use-case guide

Create `docs/contextual-bandit-use-cases.md` with:

- a contextual-bandit fit checklist;
- the complete recommendation mapping;
- the complete defensive-verification mapping;
- policy and reward justification;
- example context and coefficient tables;
- simulation phases and experiment matrices;
- fixed, random, and contextual baselines;
- online metrics and segment checks;
- `LoggedBanditDataset`, sequential replay, IPS, SNIPS, and doubly robust
  evaluation where each applies;
- overlap and propensity requirements;
- staged rollout gates;
- monitoring and rollback conditions;
- privacy and fairness constraints;
- current PyMAB gaps;
- clear alternatives for ranking, delayed sequential decisions, hard safety
  constraints, and supervised risk prediction.

The guide remains repository-only for this release.

Run the human-writing review and the published punctuation checker. Do not add
em dashes.

## Task 8: Browser, accessibility, and regression verification

Add Playwright coverage for:

- both home-page scenario cards;
- direct loading of both scenario routes;
- one manual step and auto-run completion;
- Guided, Challenge, and Free Play;
- exact seeded replay after restart;
- context and outcome changes across rounds;
- scenario-specific labels and diagnostics;
- no hidden unselected outcomes;
- navigation between scenarios and policy lessons;
- stale-session protection during warm switching;
- keyboard tooltip access;
- axe checks;
- 320, 768, and 1440 CSS pixel layouts;
- combined `/pymab/`, `/pymab/demo/`, and `/pymab/docs/` routing.

Run the complete quality suite:

```bash
uv run ruff check .
uv run mypy src/pymab web/python
uv run pytest
cd web
npm run format:check
npm run lint
npm run typecheck
npm test -- --run
npm run build
npx playwright test
cd ..
make pages-build
```

Inspect both scenario routes in the production-path local server. Completion
requires every acceptance criterion in the design spec to have direct test or
rendered evidence.
