# Run configuration panel design

## Goal

Make the active algorithm, mode, parameter, and seed obvious on every lesson page. Users should be able to change a value without triggering an unexpected reset, then start a run with one explicit action.

## Current problem

Algorithm choice currently lives on the mission map. Parameter controls appear only in Challenge and Free play, while Guided hides the values it uses. Mode buttons immediately reset the session with little visual feedback. Challenge mode also stops accepting new attempts after three runs, even though the interface does not show that limit.

The click handlers work, but the interaction looks broken because the result is easy to miss.

## Interface

Add a persistent configuration panel between the lesson header and the game. It contains:

- An algorithm selector for ε-greedy and LinUCB.
- A mode selector for Guided, Challenge, and Free play.
- A parameter slider linked to a number field.
- A seed field.
- A summary of the active run.
- A primary button labelled "Start new run" or "Restart with these settings".

The ε-greedy slider covers 0 to 1 in steps of 0.01. The LinUCB alpha slider covers 0.05 to 4 in steps of 0.05 because the policy requires a positive value. The number field accepts the same range and allows precise keyboard entry.

The seed is always visible. Guided and Challenge show their fixed seed as read-only. Free play allows any safe integer seed.

On narrow screens, the panel becomes one column. The game remains below it, and the developer inspector keeps its existing position.

## Interaction

The panel maintains a draft configuration separately from the active run. Changing the algorithm, mode, parameter, or seed updates the draft without touching the current game.

When the draft differs from the active run, the panel shows that changes have not been applied. The primary button says "Start new run" before the first run and "Restart with these settings" afterward. Pressing it applies every draft value at once and starts a fresh run. Because this action is explicit, it does not open a confirmation dialog.

Selecting another algorithm changes the draft only. It preserves the selected mode, uses the target algorithm's default parameter, and preserves the Free play seed. Pressing the primary button navigates to the matching lesson route. The application carries the draft configuration through router state so the destination lesson starts once with the requested settings.

The debrief actions remain shortcuts. "Start challenge" uses the active parameter and the lesson's fixed challenge seed. "Start free play" uses the active parameter and the most recent Free play seed. Their labels describe the action they perform.

Remove the three-attempt challenge lock. Attempts may still be recorded for local progress, but they never disable or reject a new challenge run.

## Components and state

Create a `RunSetupPanel` presentation component with a typed configuration value and callbacks for draft changes and applying the draft. Keep worker communication and routing in `LessonRoute`.

`LessonRoute` owns:

- `activeConfiguration`, which matches the current worker session.
- `draftConfiguration`, which drives the form.
- Input validation and the unapplied-change state.
- Same-algorithm restarts through `startMode`.
- Cross-algorithm navigation through router state.

Starting a worker session updates the active configuration only after the worker returns a valid initial snapshot. A failed start keeps the draft visible and uses the existing error recovery screen.

## Validation

Clamp slider movement to its documented range. Number fields may be temporarily empty while being edited, but a run cannot start until every value is valid. Show a short inline error next to an invalid field.

Accept only safe integer seeds. Do not silently round parameter values or seeds.

## Copy

Use direct labels such as "Algorithm", "Run mode", "Exploration chance", "Confidence width", "Random seed", and "Current run". Avoid promotional language, em dashes, and unexplained changes of state.

## Tests

Add component tests for:

- Slider and number field synchronization.
- Algorithm and mode selection.
- Invalid parameter and seed states.
- The active-run summary and unapplied-change message.

Add browser tests for:

- Starting Guided, Challenge, and Free play from the persistent panel.
- Starting another challenge after more than three completed attempts.
- Switching from ε-greedy to LinUCB and back.
- Preserving the current game until "Start new run" is pressed.
- Starting Challenge and Free play from the debrief shortcuts.
- Keyboard access, narrow layouts, and accessibility checks.

Refresh the affected visual snapshots after confirming the changes are intentional.
