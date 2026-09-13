# Readable policy labels and immediate switching

## Goal

Make Python policy class names easier to scan in lesson headers and remove the
confusing gap between the policy selected in the run setup and the policy that
is still active in the lesson.

## Header presentation

Lesson headers keep the policy family and complete Python class name. PascalCase
class names are split into readable words, including acronym boundaries. For
example, `LogisticContextualBanditPolicy` appears as
`Logistic Contextual Bandit Policy`, while `UCBPolicy` appears as `UCB Policy`.
Code-focused areas such as the setup panel, constructor preview, and developer
view continue to show the exact Python identifier.

## Policy switching

Selecting a different policy immediately navigates to that policy's lesson and
starts a fresh run. The current mode is preserved. In Free play, the selected
policy's saved parameters and seed are restored. Guided and Challenge use that
policy's configured defaults.

Changing parameters, environment values, or the seed within the active policy
remains a draft operation. Those edits take effect only through the existing
restart button.

The transition stops auto-run, clears the stale active configuration, and uses
the existing navigation-state handoff so the new lesson starts exactly once.

## Motion control

No motion behavior changes are included. The existing control continues to
offer system, reduced, and full modes.

## Verification

- Unit-test PascalCase formatting for ordinary names and acronyms.
- Update the setup-panel test to expect immediate policy selection callbacks.
- Update browser coverage so selecting a policy changes the route without a
  second confirmation click and preserves the selected mode.
- Confirm the previous policy cannot advance after another policy is selected.
- Run formatting, lint, type checks, frontend tests, browser smoke tests, the
  published-text check, and the combined Pages build.
