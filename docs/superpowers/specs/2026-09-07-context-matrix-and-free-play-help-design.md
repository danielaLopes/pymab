# Context matrix and Free Play help

## Goal

Make the recommendation scenario easier to understand without changing how its policy or simulation works. Explain what the context matrix represents, translate encoded values into familiar terms, and make the Free Play workflow clear.

## Current behaviour

Each recommendation round generates one visitor visit. The same visitor context is repeated for every candidate because every candidate is evaluated for that visitor. A run contains several visitor visits and preserves one policy's learned state across them. Starting a new run resets that learned state.

The context sequence is generated from the run seed. Binary signals are encoded as `-1` or `+1`. Numeric signals are scaled to the range from `-1` to `+1`. The current table exposes these model inputs but does not explain them.

Selecting Free Play changes the draft configuration. It unlocks the random seed and hidden simulation coefficients, but it does not change the active run until the user starts the configured run. The current interface does not make that second step obvious.

## Design

### Free Play guidance

When Free Play is selected, show this text below the mode selector:

> Free play unlocks the random seed and simulation coefficients. The current run will not change until you start the configured run.

Change the configuration button label from `Start configured run` to `Start free play run` while Free Play is selected. Keep the existing label for Guided and Challenge modes.

Free Play must remain on the current scenario route. It must not open or navigate to Python Lab. The separate `Open in Python Lab` button remains in Developer view.

### Context matrix help

Add an accessible information control beside the `Current context matrix` caption. It opens a tooltip on hover or keyboard focus and can be activated on a touch device.

Use this explanation:

> One round represents one visitor visit. Every candidate is evaluated for the same visitor, so the rows repeat. Binary signals use -1 and +1. Numeric signals are scaled between them. A new visitor context is generated for the next round.

Each context column also receives a short tooltip based on its feature metadata:

- `Base`: Always 1. It lets each candidate have a starting preference before visitor signals are applied.
- Binary signals: Show both labels and their encodings, such as `new = -1, returning = +1`.
- Numeric signals: State the original range and explain that it is scaled to `-1` through `+1`. For Engagement, include a concrete example: `70/100 becomes 0.4`.

The learned coefficient table does not receive these tooltips because its numbers have a different meaning. This avoids suggesting that a coefficient is a visitor value.

### Accessibility and components

Use the existing shadcn/ui conventions. Add the shadcn tooltip primitive if the project does not already contain it. The information control must have an accessible name, visible focus styling, and usable content without requiring precise pointer movement.

The explanatory text should remain available to assistive technology. Tooltips must not be the only way to identify the table or its columns.

## Data flow

The matrix component will accept optional help content and optional column descriptions. The recommendation scenario will derive column descriptions from the existing context feature metadata. Other policy screens can continue using the matrix component without help content.

No protocol or Python changes are required. The displayed matrix values, generated context, policy updates, and simulation coefficients remain unchanged.

## Testing

Add component coverage for:

- Free Play guidance appearing only when Free Play is selected.
- The Free Play apply button reading `Start free play run`.
- The matrix information control exposing the main explanation.
- Binary and numeric column descriptions using the correct encodings.
- Existing matrix tables rendering normally when no help content is supplied.

Run the TypeScript tests, lint checks, production build, and the existing browser tests. In a browser, confirm that selecting Free Play stays on the recommendation scenario and that the tooltip works with pointer and keyboard input.

## Out of scope

- Manually choosing the visitor values for each round.
- Tracking named visitors or retaining per-visitor history.
- Changing feature encoding or policy mathematics.
- Redesigning the learned coefficient table.
- Changing Python Lab.
