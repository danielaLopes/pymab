# Portal probabilities and LinUCB cues

## Goal

Let users configure each portal's relic chance in ε-greedy Free play. Make LinUCB's context signals and learned estimates easier to understand without revealing the correct portal during a run.

## Scope

This change has three parts:

- Seed-generated, editable portal probabilities for ε-greedy Free play.
- Visible learned reward estimates for LinUCB during each round.
- Accessible explanations for the light, echo, and tide context signals.

Guided and Challenge keep their current environments and scoring. LinUCB's true relic probabilities remain hidden until the debrief.

## ε-greedy Free play interface

Add an inline section called "Portal relic chances" to the existing run setup panel. It appears only when ε-greedy and Free play are selected.

The section has one linked slider and number field for each portal:

- Moon Gate
- Sun Gate
- Star Gate

Values are displayed as percentages from 0% to 100% in steps of 0.1%. The application stores them as normalized floating-point values from 0.0 to 1.0.

The section also shows whether the values are "Generated from seed" or "Custom." A button labelled "Use seed-generated values" restores the defaults for the current seed.

During an active ε-greedy Free play run, each portal shows its configured relic chance. Guided and Challenge do not show the hidden probabilities before completion.

## Generated defaults

The same safe integer seed always generates the same three default probabilities in every supported browser.

Generation produces one value in each of three bands:

- Lower chance: 10% to 35%
- Middle chance: 40% to 65%
- Higher chance: 70% to 95%

The seed also determines which portal receives each band. This keeps the portals meaningfully different without making one named portal permanently best.

Generated values use 0.1% increments. Custom values may use any valid 0.1% increment from 0% to 100%.

When the values are still generated, changing the random seed generates a new set. Editing any probability changes the source to Custom. Custom values remain unchanged when the seed changes. The reset button generates defaults from the current seed and changes the source back to Generated.

## Run configuration and worker contract

Portal probabilities are environment settings, not ε-greedy policy parameters.

Extend the run configuration with an optional three-value probability tuple for ε-greedy Free play. The draft configuration also records whether the values are generated or custom. Algorithm and mode switching preserve the draft values when possible.

Extend the worker start request with a separate environment object. The Python session validates that:

- The environment is accepted only for ε-greedy Free play.
- Exactly three finite probabilities are provided.
- Every probability is between 0.0 and 1.0.
- Every probability uses a 0.001 increment, equivalent to 0.1%.

Guided and Challenge continue using 0.25, 0.50, and 0.75. The ε-greedy session uses the submitted Free play probabilities for reward sampling, expected regret, hidden truth, and equivalent Python code.

The active configuration changes only after the worker returns a valid initial snapshot. A failed start leaves the previous session available and keeps the draft visible.

## LinUCB run window

After each LinUCB decision, show the learned reward estimate used for each portal. Label it "Learned estimate" rather than "Probability" because the linear prediction is not constrained to the 0 to 1 range.

The run window uses the existing `predictedMeans` diagnostic. The true context-dependent probabilities and hidden coefficients remain in the completed debrief.

Before the first decision, show "No estimate yet" for each portal.

## Context signal tooltips

Add a small help control to each LinUCB context signal. The controls work with pointer hover and keyboard focus.

Use this copy:

- Light: "Light can be red or blue. LinUCB sees it before choosing a portal."
- Echo: "Echo can be low or high. LinUCB sees it before choosing a portal."
- Tide: "Tide can be low or high. LinUCB sees it before choosing a portal."

Use the shadcn/ui Tooltip component so behavior, focus handling, and visual treatment match the existing component library.

## Validation

Probability fields may be empty while a user is editing. The start button is disabled until all three fields are valid. Show a short inline error for an invalid value and never round silently.

Slider input is clamped to its documented range. Number input must match the 0.1% step.

## Tests

Add unit tests for:

- Deterministic generation from the same seed.
- Different seeds changing values or portal assignment.
- The three generated probability bands.
- Draft validation and percentage conversion.
- Generated values updating when the seed changes.
- Custom values surviving seed changes.
- Resetting custom values from the current seed.

Add Python tests for:

- Environment validation.
- Reward sampling and expected regret using submitted probabilities.
- Hidden truth using submitted probabilities.
- Generated Python code containing the submitted probabilities.
- Guided and Challenge retaining the fixed environment.

Add browser and accessibility tests for:

- Editing the three Free play controls and starting a run.
- Showing configured chances on ε-greedy Free play portals.
- Keeping draft changes separate from the active run.
- Showing LinUCB learned estimates without true probabilities.
- Opening each tooltip by hover and keyboard focus.
- Narrow layouts without horizontal overflow.

Refresh the affected visual snapshots after reviewing the changes.
