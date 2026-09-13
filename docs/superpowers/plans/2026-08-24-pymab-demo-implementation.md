# PyMAB Interactive Demo Implementation Plan

Date: 2026-08-24

## Goal

Implement the approved [PyMAB Interactive Demo design](../specs/2026-08-24-pymab-demo-design.md)
as a static React application under `web/`. The release is complete only when
the epsilon-greedy and LinUCB lessons, developer inspector, editable Python Lab,
real-Pyodide browser tests, and GitHub Pages deployment all satisfy the design's
acceptance criteria.

The lesson bridge must instantiate the real `EpsilonGreedyPolicy` and
`LinUCBPolicy` classes from the wheel built from the checked-out commit.

This plan does not broaden the first release to additional policies. It does
not change PyMAB's public API or add anything to the Python wheel.

## Fixed Technical Decisions

- Node.js 24 in CI; npm with `web/package-lock.json`.
- React `19.2.8`, React DOM `19.2.8`, React Router DOM `7.18.2`.
- TypeScript `6.0.3`, Vite `8.2.2`, `@vitejs/plugin-react` `6.1.0`.
- Pyodide `314.0.5`, pinned in `web/package.json` and self-hosted by the build.
- CodeMirror `6.0.2`, `@codemirror/state` `6.7.1`,
  `@codemirror/view` `6.43.9`, `@codemirror/commands` `6.11.0`,
  `@codemirror/language` `6.12.4`, and `@codemirror/lang-python` `6.2.1`.
- Zod `4.4.3` for runtime validation of worker messages and persisted state.
- Vitest and `@vitest/coverage-v8` `4.1.11`, Testing Library React `16.3.2`,
  DOM `10.4.1`, jest-dom `7.0.1`, user-event `14.6.6`, jsdom `30.0.1`,
  Playwright `1.62.1`, axe-core and `@axe-core/playwright` `4.13.0`.
- React types `19.2.18`, React DOM types `19.2.5`, and Node 24 types
  `24.13.3`.
- ESLint `10.9.1`, typescript-eslint `8.68.0`, React Hooks `7.1.1`, React
  Refresh `0.5.4`, globals `17.11.0`, and Prettier `3.9.6`. Accessibility
  enforcement uses semantic component tests plus axe because the current
  jsx-a11y release does not support ESLint 10.
- Hash routes: `#/`, `#/lesson/epsilon-greedy`, `#/lesson/linucb`, `#/lab`.
- Vite production base `/pymab/`; local development base `/`.
- Generated wheels and copied Pyodide assets live under `web/.generated/` and
  are never committed.

## Canonical Runtime Contracts

Define the protocol once in `web/src/engine/protocol.ts` and mirror its field
names in `web/python/pymab_demo/protocol.py`. Zod validates every response before
React receives it.

All requests contain `requestId`. Session commands also contain `sessionId`.

```text
InitializeRequest  { type, requestId }
StartLessonRequest { type, requestId, sessionId, lessonId, mode, seed, parameters }
StepRequest        { type, requestId, sessionId }
RunToEndRequest    { type, requestId, sessionId }
ResetRequest       { type, requestId, sessionId }
DisposeRequest     { type, requestId, sessionId }
```

Responses are a discriminated union:

```text
ready | lessonStarted | stepCompleted | runCompleted | disposed | error
```

`stepCompleted` and `runCompleted` contain a `LessonSnapshot` with:

- Identity: lesson, mode, seed, package version, source commit, session, step,
  horizon, and parameters.
- Outcome: selected arm, reward, total reward, instantaneous expected regret,
  cumulative expected regret, and completion status.
- Presentation: gate identities, visible cue labels, explanation key, and
  reduced public context.
- Decision diagnostics captured before selection and learning diagnostics
  captured after update.
- Hidden truth only when `reveal=true`, which occurs after guided explanation
  points and after a challenge/free run completes.

The epsilon diagnostic contains counts, estimates, greedy arms, epsilon,
selection branch, and sampled random value. The LinUCB diagnostic contains the
context matrix, learned theta estimates, predicted means, raw uncertainty,
alpha-scaled bonuses, UCB scores, and matrices after update.

Errors use `{code, message, recoverable, details?}`. Codes are
`BOOT_FAILED`, `INVALID_REQUEST`, `INVALID_SESSION`, `POLICY_FAILED`,
`STALE_RESPONSE`, `LAB_SYNTAX`, `LAB_RUNTIME`, `LAB_TIMEOUT`, and
`OUTPUT_LIMIT`.

## Task 1: Scaffold the Web Project and Quality Gates

Create:

- `web/package.json` and `web/package-lock.json` with the exact versions above.
- `web/index.html` with viewport metadata and the restrictive static CSP:
  `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self';
  connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline';
  font-src 'self'; object-src 'none'; base-uri 'self'`.
- `web/tsconfig.json`, `web/tsconfig.app.json`, `web/vite.config.ts`,
  `web/vitest.config.ts`, `web/playwright.config.ts`, `web/eslint.config.js`,
  and `web/.prettierrc.json`.
- Minimal `web/src/main.tsx`, `web/src/App.tsx`, `web/src/test/setup.ts`, and a
  shell test proving all four hash routes render.
- `.nvmrc` containing `24` and `.gitignore` entries for `web/node_modules/`,
  `web/dist/`, `web/.generated/`, Playwright output, and JS coverage.

Add npm scripts: `dev`, `format`, `format:check`, `lint`, `typecheck`, `test`,
`test:coverage`, `prepare:python`, `build`, `preview`, `e2e`, and `e2e:install`.
`build` must run asset preparation, type checking, and Vite in that order.

Add root Make targets `web-sync`, `web-format`, `web-lint`, `web-test`,
`web-build`, `web-e2e`, and `web-ci`; do not add Node work to the existing
Python-only `sync` or package-release targets.

Verification:

```bash
cd web && npm ci
npm run format:check
npm run lint
npm run typecheck
npm test -- --run
npm run build
```

Commit: `build: scaffold pymab arcade web app`

## Task 2: Build the Python Lesson Bridge Test-first

Create the browser-neutral package:

- `web/python/pymab_demo/protocol.py`: typed JSON-safe response dataclasses and
  serialization helpers.
- `web/python/pymab_demo/fixtures.py`: exact gates, cues, theta matrices,
  horizons, seeds, parameter choices, and thresholds from the design.
- `web/python/pymab_demo/sessions.py`: `EpsilonLessonSession` and
  `LinUCBLessonSession` implementing start, step, run-to-end, reset, and dispose.
- `web/python/pymab_demo/diagnostics.py`: pre-decision/post-update snapshots,
  epsilon RNG peeking, and LinUCB score decomposition.
- `web/python/pymab_demo/codegen.py`: complete public-API Python examples for
  each active configuration.
- `web/python/pymab_demo/entrypoint.py`: `dispatch_json(request_json) -> str`
  with no Pyodide-specific imports.

Use `pymab._random.generator` only inside this repository-owned bridge to match
PyMAB's named deterministic streams. Name them `context`, `action`, and
`reward`. Potential rewards are sampled for all arms before selecting the
chosen reward, so regret and replay remain well-defined.

Add `tests/demo/` tests before implementation. Cover:

- Fixture shape/domain validation and the exact allowed parameter values.
- Epsilon branch peeking does not mutate the action RNG and matches the action
  selected by the real policy.
- LinUCB predicted mean + alpha-scaled uncertainty equals the public
  `upper_confidence_bounds` output.
- Every response round-trips through JSON without NumPy scalars or arrays.
- Intermediate challenge snapshots do not reveal hidden probabilities, theta,
  or optimal actions; completed snapshots do.
- Guided and challenge golden snapshots for the specified seeds.
- Exact challenge outcomes used to calibrate success: epsilon `0.20` passes
  seed `7`; epsilon `0` and `0.80` fail at least one target. LinUCB alpha `1.0`
  passes seed `20260824`; alpha `0.25` and `2.0` fail at least one target.
- Reset reproduces every context, action, reward, branch, diagnostic, and total.
- Invalid commands and disposed/stale sessions return the specified errors.
- Generated examples parse with `ast.parse` and execute to the same final
  metrics as their session.

Generated examples use only public PyMAB APIs. To preserve exact named-stream
behavior without importing `pymab._random`, code generation emits the already
derived integer stream seeds as literals and constructs each stream with
`numpy.random.default_rng(numpy.random.SeedSequence(seed_literal))`.

Extend strict mypy to `web/python` and add a separate demo coverage command with
at least 95% statement and branch coverage.

Verification:

```bash
uv run pytest tests/demo --cov=web/python/pymab_demo --cov-branch --cov-fail-under=95
uv run mypy src/pymab web/python
uv run ruff check web/python tests/demo
```

Commit: `feat: add deterministic demo lesson bridge`

## Task 3: Prepare Self-hosted Python Assets

Create `web/scripts/prepare-python-assets.mjs` and a small unit-tested helper
module. It must:

1. Run `uv build --wheel --out-dir web/.generated/wheels` from the repository
   root.
2. Assert exactly one `pymab-*-py3-none-any.whl` was produced.
3. Copy `pyodide.mjs`, `pyodide.asm.mjs`, `pyodide.asm.wasm`,
   `python_stdlib.zip`, and `pyodide-lock.json` from
   `web/node_modules/pyodide` into `web/.generated/public/pyodide/`.
4. Parse `pyodide-lock.json`, resolve NumPy and all transitive dependencies,
   download their exact wheel filenames from the pinned Pyodide jsDelivr
   release, verify each lockfile SHA-256, and cache them under
   `web/.generated/cache/`.
5. Copy verified wheels beside the runtime, copy the PyMAB wheel to
   `web/.generated/public/wheels/`, and deterministically zip
   `web/python/pymab_demo/` as
   `web/.generated/public/python/pymab-demo-bridge.zip`.
6. Write `runtime-manifest.json` containing Pyodide version, Python version,
   NumPy filename, PyMAB filename/version, bridge filename, Git commit, and
   SHA-256 values for every generated or downloaded input.
7. Fail on missing hashes, version mismatches, duplicate wheels, dirty partial
   downloads, or attempts to copy outside `.generated`.

Vite serves `.generated/public` as its public directory. Repeated local builds
reuse verified cached downloads; `--clean` forces regeneration. No generated
binary is staged.

Tests mock download responses and cover dependency traversal, hash mismatch,
manifest generation, cache reuse, and path traversal. A build integration test
opens the real manifest, validates every referenced asset, and imports the
built wheel under CPython.

Verification:

```bash
cd web && npm run prepare:python -- --clean
npm test -- --run src/build
git status --short
```

The last command must show no generated assets.

Commit: `build: prepare self-hosted pyodide assets`

## Task 4: Implement the Typed Worker Runtime

Create:

- `web/src/engine/protocol.ts` with Zod request/response schemas and exported
  inferred types.
- `web/src/engine/lesson.worker.ts`: dynamically imports the self-hosted
  `pyodide.mjs`, reports staged progress, loads NumPy, installs the local PyMAB
  wheel and bridge zip into Pyodide's site-packages, and dispatches JSON
  requests.
- `web/src/engine/WorkerClient.ts`: request IDs, one in-flight mutation,
  session IDs, stale-response rejection, disposal, crash recovery, and progress.
- `web/src/engine/RuntimeProvider.tsx`: one active worker at a time and lazy Lab
  worker creation.
- `web/src/engine/support.ts`: WebAssembly/module Worker capability check.

Do not pass `PyProxy` objects across the worker boundary. Convert only the JSON
string returned by `dispatch_json`, validate it with Zod, and then post plain
structured-clone-safe data.

Install local Python artifacts without `micropip`: fetch and verify each file
against `runtime-manifest.json`, write it to Pyodide's virtual filesystem, and
use Python's `zipfile` plus `site.getsitepackages()[0]` to extract the PyMAB
wheel and bridge zip. Delete the temporary archives, import both packages, and
verify the imported PyMAB version before emitting `ready`.

Unit tests use a controllable fake Worker and cover boot progress, concurrent
command rejection, request correlation, stale sessions, protocol validation,
crash/reset, and disposal. The first Playwright smoke test must boot real
Pyodide, start epsilon guided mode, execute one step, and assert the package
version and selected action match the CPython fixture.

Verification:

```bash
cd web && npm run typecheck
npm test -- --run src/engine
npm run e2e -- tests/runtime-smoke.spec.ts --project=chromium
```

Commit: `feat: run real pymab in a browser worker`

## Task 5: Add Application State, Routing, and Persistence

Create:

- `web/src/routes/` for Home, Lesson, and Lab route components.
- `web/src/state/lessonReducer.ts` with explicit `loading`, `guided`,
  `challengeSetup`, `challengeRunning`, `debrief`, `freePlay`, `error`, and
  `unsupported` states.
- `web/src/state/persistence.ts` with Zod schema version `1` under local-storage
  key `pymab-arcade:v1`.
- `web/src/content/lessons.ts` containing beginner copy and explanation-key
  mappings, separate from numerical Python fixtures.

Persist only completion, attempt counts, most recent supported parameters/seed,
inspector preference, and reduced-motion override. Never persist worker state,
hidden truth, stdout, or arbitrary Lab code. Invalid or future-version data is
discarded safely.

Changing seed or policy parameters after the first step opens a confirmation;
confirming creates a new session, cancelling leaves everything unchanged.
Leaving a lesson disposes the session. Opening Lab disposes the lesson worker
after preserving the generated example in navigation state.

Tests cover every reducer transition, deep-link refresh, corrupted persistence,
version mismatch, confirmation behavior, disposal, and back/forward navigation.

Commit: `feat: add demo navigation and lesson state`

## Task 6: Build the Shared Infinite Crossroads Interface

Create focused components under `web/src/components/game/`:

- `AppShell`, `CampaignMap`, `MissionHeader`, `Chamber`, `Gate`, `CueStrip`,
  `OutcomeReveal`, `RunControls`, `ProgressTrail`, `ParameterChallenge`,
  `Debrief`, `LoadingStages`, `ErrorRecovery`, and `UnsupportedBrowser`.
- Responsive inline SVG art for the three gates and chamber; no raster assets or
  canvas.
- `web/src/styles/` tokens, reset, typography, focus, motion, and layout CSS.

Gate buttons remain native buttons with persistent names, symbols, and
accessible descriptions. Use a polite live region for results and an assertive
one only for unrecoverable errors. Initial focus enters the lesson heading;
after a step it remains on the initiating control. Auto-run exposes Pause and
never advances through hidden browser tabs.

Implement a CSS transition state machine (`idle`, `deciding`, `opening`,
`reward`, `learning`) driven by lesson state. Reduced motion skips spatial
movement but preserves the same ordered labels and outcomes.

Component tests cover keyboard order, names, result announcements, disabled
states, auto-run pause, reduced motion, and 360/736/1024 layouts. Add screenshot
baselines for the home page, both lesson chambers, inspector closed/open, and
both debriefs in dark mode; use a single intentional visual-regression project.

Commit: `feat: build infinite crossroads game interface`

## Task 7: Implement the Epsilon-greedy Lesson

Connect the shared UI to `epsilon-greedy` fixtures and diagnostics:

- Guided mode renders 12 fixed-seed steps and callouts for first observation,
  first definite exploration, exploitation, estimate update, and cumulative
  regret.
- Challenge mode offers only the six approved epsilon values, three attempts,
  20 chambers, and the two-part success rule.
- Free play allows the same epsilon choices plus any integer seed accepted by
  the bridge; step and auto-run share the same worker commands.
- The decision view plots counts and estimates per gate and labels the exact
  explore/exploit branch. It never labels a greedy action as exploitation based
  only on the selected arm.
- Debrief reveals true probabilities, actions, rewards, regret path, pass/fail
  reason, and the single-run statistical caveat.

Tests complete guided mode, verify the three challenge calibrations, exercise
all epsilon choices, reset/replay, attempt limits, and generated-code handoff.
The real-browser test compares every displayed action/reward/diagnostic against
the worker snapshots.

Commit: `feat: teach epsilon greedy through ancient gates`

## Task 8: Implement the LinUCB Lesson

Connect the shared UI to `linucb` fixtures and diagnostics:

- Translate feature values to visible light, echo, and tide cues without hiding
  the numeric vector from Inspect PyMAB.
- Guided mode renders 12 fixed-seed steps and callouts for initial uncertainty,
  context-dependent predictions, confidence bonus, update, and a case where a
  different context changes the recommended gate.
- Challenge mode offers only the six approved alpha values, fixes `l2=1.0`,
  allows three attempts, and applies the exact two-part target.
- Free play permits the approved alpha values and any integer seed.
- Before reveal, show only current cues and learned policy diagnostics. Debrief
  reveals logistic environment probabilities, optimal actions, and theta.
- The visual decomposes each UCB score into predicted mean plus uncertainty
  bonus using aligned bars and text; colour is not the only encoding.

Tests complete guided mode, verify alpha `1.0` passes and `0.25`/`2.0` fail the
calibrated challenge, assert cue-to-vector mapping, compare decomposition with
PyMAB, and verify no path/navigation state appears in the protocol or UI.

Commit: `feat: teach linucb through infinite crossroads`

## Task 9: Add Inspect PyMAB and Code Generation

Create `web/src/components/inspector/` with:

- Collapsible policy/configuration summary.
- Step input, decision calculation, learning update, metrics, package version,
  source commit, and generated code sections.
- Epsilon-specific estimates/counts/branch view.
- LinUCB context matrix, theta estimates, mean/uncertainty/UCB decomposition,
  and post-update matrices.
- Copy code and **Open in Lab** actions with accessible success feedback.

The inspector reads only the validated snapshot. It performs formatting but no
algorithm calculation. Numeric formatting uses four significant decimals and
exposes full precision through accessible details/copyable JSON. Generated code
comes from Python `codegen.py`; TypeScript must not synthesize policy code.

Tests verify progressive disclosure, exact constructor parameters, seed and
commit display, no hidden truth before reveal, copy behavior, and Lab handoff.

Commit: `feat: add developer policy inspector`

## Task 10: Implement the Disposable Python Lab

Create:

- `web/src/lab/lab.worker.ts` to boot a clean Pyodide runtime, load NumPy and
  the same PyMAB wheel, install a bounded stdout/stderr writer, run code, and
  return structured results.
- `web/src/lab/LabClient.ts` with a five-second wall timer. Stop, timeout, worker
  error, and navigation terminate the worker; the next run boots a clean one.
- `web/src/routes/LabRoute.tsx` with CodeMirror, epsilon/LinUCB examples, Run,
  Stop, Reset, execution status, and distinct stdout/stderr.

Cap combined output while writing, not after constructing an unbounded string.
Return at most 64 KiB plus an explicit truncation flag. Sanitize tracebacks to
remove internal virtual-filesystem prefixes while preserving Python exception,
user line number, and relevant stack frames. Do not expose package installation
controls.

Tests cover successful examples, stdout/stderr separation, syntax/runtime
errors, output truncation, infinite-loop timeout, manual Stop, rapid repeated
runs, navigation disposal, and clean recovery. Playwright performs all cases
with real Pyodide.

Commit: `feat: add editable browser python lab`

## Task 11: Complete Accessibility, Resilience, and Performance QA

Add automated axe checks to every route and meaningful lesson state. Run
keyboard-only Playwright scenarios for route navigation, lesson completion,
inspector, parameter confirmation, and Lab. Test system reduced motion and the
explicit override. Ensure all charts have concise text alternatives and all
matrices have semantic tables.

Test worker boot failure, bad manifest, wheel import failure, malformed Python
response, session mismatch, worker crash, Retry, and replay-from-zero recovery.
Verify controls never depend on animation events and stale responses cannot
mutate state.

Establish budgets measured on the production build:

- Main application JS excluding Pyodide: at most 350 KiB gzip.
- No main-thread task longer than 100 ms during Python initialization or a
  lesson step on the Playwright desktop profile.
- Warm-cache lesson worker ready within 2 seconds on the CI browser profile.
- No horizontal overflow at 320 CSS pixels and no clipped focus indicator.

Record production bundle sizes and fail CI when the JS budget is exceeded. Do
not set a cold-network time budget for the self-hosted Python runtime; instead
verify progressive loading, immutable caching, and a usable non-game shell
during download.

Commit: `test: harden demo accessibility and resilience`

## Task 12: Integrate CI and GitHub Pages

Extend `.github/workflows/ci.yml` with independent Node 24 jobs:

- `web-quality`: npm cache, `npm ci`, format check, ESLint, TypeScript, and
  Vitest coverage.
- `web-build`: Python 3.12 plus uv and Node 24; build wheel/runtime/site and
  upload `web/dist` as a normal pull-request artifact.
- `web-browser`: install Playwright browsers, restore prepared-runtime cache,
  run Chromium on pull requests, and run Chromium/Firefox/WebKit on pushes to
  `main` and manual workflow dispatch.

Use the repository's existing action-version conventions. Pin third-party
actions to immutable SHAs where the repository already does so.

Create `.github/workflows/pages.yml` with a build job and a separate deploy job.
The build repeats all quality gates, uses `actions/configure-pages@v6`, and
uploads `web/dist` with `actions/upload-pages-artifact@v4`. The deploy job needs
the build, uses the `github-pages` environment, grants `pages: write` and
`id-token: write`, and deploys with `actions/deploy-pages@v5`. Trigger production
only on successful pushes to `main` and manual dispatch; pull requests never
deploy.

Add an after-deploy Playwright smoke command that opens the emitted Pages URL,
boots Pyodide, completes one seeded step from each policy, and opens Lab. Keep
README/Sphinx links out until that smoke test succeeds on the production URL.

Commit: `ci: test and deploy pymab arcade`

## Task 13: Documentation and Final Acceptance

After the first successful production deployment:

- Add the live Arcade URL and a one-sentence description to `README.md`.
- Add `docs/source/arcade.rst` explaining supported browsers, local development,
  runtime download, privacy/no analytics, lesson scope, and the distinction
  between contextual bandits and stateful reinforcement learning.
- Link Arcade from `docs/source/index.rst` and the policy decision guide.
- Add contributor commands and asset-pipeline troubleshooting to
  `.github/CONTRIBUTING.md`.
- Add a changelog entry without changing the Python package version solely for
  the website.

Run the complete completion audit:

```bash
make format
make lint
make test
make docs
make web-ci
make web-build
make web-e2e
git diff --check main...HEAD
git status --short
```

Then verify, from the deployed URL, every acceptance criterion in the approved
design on Chromium, Firefox, and WebKit, followed by a manual smoke run in the
current desktop Safari release on macOS. Capture the CI URLs, deployed commit,
PyMAB version shown in the inspector, seeded lesson results, Lab recovery
result, accessibility report, and responsive screenshots as release evidence.

Commit: `docs: publish pymab arcade guide`

## Implementation Order and Merge Discipline

Execute Tasks 1-13 in order. Tasks 2 and 3 may be developed in parallel only
after Task 1, but Task 4 must not start until both are green. Tasks 7 and 8 may
be developed in parallel after Tasks 5-6. Task 9 follows both lessons so it
cannot invent a lowest-common-denominator contract. Task 10 follows the shared
runtime asset pipeline but remains isolated from lesson session state.

Keep the listed commit boundaries unless a review fix must amend the immediately
preceding commit. Do not commit generated runtime assets, wheels, test output,
coverage, screenshots outside intentional baselines, or local persistence.

When a contract changes, update Python serialization tests, Zod schemas,
TypeScript tests, and the real-Pyodide smoke fixture in the same commit. A mocked
worker test never substitutes for the required real-Pyodide browser evidence.

## Authoritative References

- [Using Pyodide in a Web Worker](https://pyodide.org/en/latest/usage/webworker.html)
- [Loading custom Python wheels](https://pyodide.org/en/stable/usage/loading-custom-python-code.html)
- [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
