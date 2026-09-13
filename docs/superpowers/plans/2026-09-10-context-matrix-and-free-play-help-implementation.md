# Context matrix and Free Play help implementation

## Scope

Implement the approved design in `docs/superpowers/specs/2026-09-07-context-matrix-and-free-play-help-design.md` without changing the recommendation simulation or policy calculations.

## Tasks

1. Add component tests for the Free Play guidance and mode-specific start button label.
2. Add reusable descriptions for each recommendation context encoding.
3. Extend the matrix table with optional table and column help using the existing shadcn tooltip component.
4. Enable that help only for the recommendation scenario's current context matrix.
5. Add focused styles for the information controls and tooltip copy.
6. Run unit tests, type checking, linting, formatting checks, the production build, published-text checks, and browser tests.
7. Verify in a browser that Free Play stays on the recommendation route and that the matrix help works with pointer and keyboard input.

## Expected result

Free Play remains a configuration mode. The interface tells users what it unlocks and that they must start the configured run. The context matrix explains why its rows repeat and how each displayed value maps back to the visitor signals.
