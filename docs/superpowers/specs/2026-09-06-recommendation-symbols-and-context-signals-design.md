# Recommendation symbols and context signals

## Goal

Give the recommendation scenario more visual identities and let users choose which predefined context signals the policy observes. Keep the editor specific to recommendation problems rather than turning the Arcade into a general simulation builder.

## Candidate symbols

Candidate names remain unrestricted, non-empty, and unique. The visual-type catalogue expands to article, product, tutorial, video, podcast, newsletter, course, event, tool, message, offer, and download.

The selected type controls the symbol shown in Run Settings, Decision History, and Developer View. It does not change the candidate's coefficients or simulation behavior. Symbols come from the application catalogue. The interface does not accept uploaded SVG, HTML, or arbitrary icon definitions.

## Context signal catalogue

The base feature is always active and is not shown as a generated signal. Visitor type, engagement, and visit timing remain active in the default configuration.

Users can add or remove these predefined signals:

- visitor type: new or returning;
- engagement: a numeric score displayed from 0 to 100;
- visit timing: weekday or weekend;
- device: desktop or mobile;
- account age: a numeric number of days;
- recent activity: a numeric interaction count;
- price sensitivity: a numeric score displayed from low to high;
- session depth: a numeric page count;
- traffic source: organic or paid.

At most eight visible signals can be active. Together with the base feature, this produces one to nine model features. The scenario starts with three visible signals and four model features, matching the existing implementation.

Binary signals produce `-1` or `+1`. Numeric signals are generated inside a stated display range and normalized to `-1` through `+1` before they reach the policy. Decision History shows the understandable display value. Developer View shows the normalized context matrix used by the policy.

## Configuration model

Each signal has a stable catalogue ID, display name, type, symbol, value labels or numeric range, normalization rule, and deterministic generation rule. The configuration stores active signal IDs in display order.

Candidate coefficients are stored by feature ID rather than as a fixed tuple. Each candidate always has a base coefficient. Adding a signal creates a deterministic coefficient for every candidate using the run seed, candidate ID, and signal ID. Adding a candidate creates coefficients for the base feature and all active signals.

Removing a signal from the active list does not delete its stored coefficients. Adding it again restores the previous values. Reordering signals changes the context and matrix column order without attaching coefficients to the wrong feature. Changing a candidate name or symbol leaves its coefficients unchanged.

Existing saved configurations without signal metadata are migrated in memory to the default signal set. Their four current coefficients map to base, visitor type, engagement, and visit timing in that order.

## Run settings

The Candidates editor keeps the custom name field and replaces the three-option visual selector with the expanded catalogue. Add controls remain focused on creating a candidate; a new candidate can then receive any name and visual type.

A Context signals section appears after Candidates. It lists active signals with a remove control and lists inactive signals in an Add signal selector. The default signals can be removed. The base feature is described beside the section count but cannot be removed.

The simulation coefficient editor creates one row per candidate and one column per active model feature. Columns use feature names instead of fixed positions. Numeric fields keep the current direct editing behavior. Wider matrices scroll inside their own container without causing page-level horizontal overflow.

Changes remain drafts until the user starts the configured run. Starting a run resets history because the policy dimensions or interpretation may have changed.

## Simulation

The Python scenario derives `n_features` from the active signal count plus the base feature. On every round it generates one value per active signal from seed-isolated random streams, builds one feature vector, and repeats that vector for every candidate.

The environment coefficient matrix is assembled in active feature order. The true click probability for each candidate is the sigmoid of the coefficient and feature dot product. The policy receives the same matrix shape and learns one coefficient vector per candidate.

The snapshot exposes signal presentation metadata, visible display values, the normalized context matrix, candidate presentation data, and variable-width learned coefficient matrices. Generated Python includes the active signal names and constructs the policy with the derived feature count.

## Validation and recovery

The TypeScript editor and Python worker both validate the configuration. A valid recommendation run has two to eight candidates, zero to eight visible signals, unique candidate IDs and names, unique supported signal IDs, supported visual types, finite coefficients for every required feature, and no unknown coefficient keys in the active matrix.

Invalid drafts disable the start button and show a specific message next to the affected control. Worker validation remains the final boundary for malformed requests. A rejected draft leaves the current run available.

## Testing

TypeScript tests cover the expanded symbol catalogue, default signal migration, binary and numeric signal definitions, deterministic coefficient generation, add and remove behavior, coefficient restoration, feature ordering, and derived feature counts.

Python tests cover default compatibility, runs with no optional signals, runs with eight signals, deterministic context generation, numeric normalization, matrix dimensions, validation failures, reset replay, and generated Python.

Component and browser tests cover candidate symbol selection, adding and removing signals, starting a configured run, visible values in Decision History, normalized values in Developer View, dynamic coefficient columns, narrow layouts, and accessibility.

## Out of scope

This change does not accept custom SVG, image uploads, arbitrary executable generators, free-form feature schemas, multi-category features, delayed rewards, or context supplied by an external service. It does not change the feature model of policy lessons or defensive verification.
