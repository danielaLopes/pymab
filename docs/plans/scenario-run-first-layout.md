# Scenario run-first layout

## Goal

Make the applied contextual-bandit scenarios easier to follow by treating the decision history as the primary interface.

## Layout

The recommendation and defensive-verification pages use this order:

1. Mission introduction
2. A compact two-part scope boundary
3. Collapsed run settings
4. Full-width run progress and decision history
5. Outcome and playback controls
6. Full-width Developer View

The policy lesson pages keep their existing two-column layout.

## Run settings

Run settings collapse automatically after a run starts successfully, including the initial seeded run. The collapsed bar shows the scenario, mode, and current policy parameters. Opening it reveals the existing scenario selector, mode controls, parameter controls, seed, and free-play environment settings.

## Scope boundary

The recommendation scenario separates the decision that fits a contextual bandit from cases that require a different model. The defensive-verification scenario separates the policy's narrow choice among approved checks from deterministic security rules.

## Accessibility and responsive behavior

The settings control exposes its expanded state and names the region it controls. Developer View remains collapsed by default. At narrow widths, summaries and boundary cards stack without creating page-level horizontal overflow; the history itself retains its internal horizontal scrolling.
