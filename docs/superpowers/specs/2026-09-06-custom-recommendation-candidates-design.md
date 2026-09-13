# Custom recommendation candidates

## Goal

Allow a user to run the recommendation scenario with two to eight candidates instead of a fixed set of three. The default run remains Article, Product, and Tutorial. A user can add candidates such as Tutorial 2, rename any candidate, choose its visual type, reorder it, remove it, and edit its simulation coefficients.

This feature applies only to the recommendation scenario. Defensive verification and policy lessons keep their existing three-action designs.

## Candidate model

Each recommendation candidate has:

- a stable internal ID;
- a user-visible name;
- a visual type: article, product, or tutorial;
- four environment coefficients: intercept, returning visitor, engagement, and weekend visit.

Names must be non-empty after trimming and unique without regard to letter case. The interface accepts between two and eight candidates. Internal IDs, rather than names or list positions, preserve identity while a candidate is renamed or reordered.

The initial candidates retain the current coefficient rows exactly. A newly added candidate receives a deterministic coefficient row derived from the current run seed and its stable ID. The generated values use bounded ranges that produce plausible differences without making every outcome certain. Repeating the same configuration and seed produces the same coefficients.

Changing a name, visual type, or seed after creation does not regenerate an existing row or overwrite manual edits. A user can edit every coefficient in Free Play. Removing and adding a new candidate creates a new stable ID and a newly generated row.

## Run settings interface

The expanded Run Settings panel gains a Candidates section above Policy parameters. It lists candidates as compact editable rows. Each row contains:

- a text field for the display name;
- an article, product, or tutorial type selector;
- controls to move the candidate up or down;
- a remove control that is disabled when only two candidates remain.

Three compact controls, Add article, Add product, and Add tutorial, append a candidate until the maximum of eight is reached. The initial name uses the selected type and the first available number, such as Tutorial 2 or Product 2. The user can replace it immediately.

Validation appears beside the relevant field. Start configured run remains disabled while a name is empty, duplicated, or while the candidate count is outside the supported range. Reordering, adding, or removing candidates affects only the draft until the user starts the configured run. A successful start collapses Run Settings and resets the history because the policy arm mapping has changed.

The collapsed settings summary shows the candidate count alongside the mode and policy parameters. It does not list all candidate names.

## Environment coefficients

Simulation coefficients remain an advanced Free Play control. The matrix is generated from the candidate list instead of assuming three rows. Each row is labeled with the current candidate name and contains the four existing coefficients.

Adding a candidate adds its deterministic row. Reordering candidates reorders coefficient rows with them. Removing a candidate removes its row. Renaming or changing a visual type preserves its row.

The scenario configuration sent to the worker contains the candidate presentation data and matching coefficient matrix in every mode. Guided and Challenge hide coefficient editing but still need these values for custom candidates. The snapshot exposes the complete environment only in Free Play. The worker validates the count, IDs, names, visual types, coefficient dimensions, numeric finiteness, and uniqueness before constructing a policy.

## Policy and worker behavior

`RecommendationScenarioSession` derives `n_arms` from the candidate list and constructs `LogisticContextualBanditPolicy` with that value. It repeats the current four-feature visitor context once per candidate and computes one probability per coefficient row.

The snapshot returns a variable-length presentation list, gate ID list, context matrix, prediction vector, learned coefficient matrix, history cells, and hidden truth. Selected-arm and recommendation indexes are validated against the current candidate count instead of the previous fixed range of zero to two.

Generated Python uses the configured candidate count and includes a short candidate-name list so arm indexes can be interpreted. Developer View derives `n_arms` from the snapshot presentation rather than displaying three unconditionally.

The worker protocol accepts between two and eight presentation arms. Policy lesson snapshots and defensive-verification snapshots remain valid with their three arms.

## Decision history

The decision-history board renders one column per candidate. It sets its table column count and grid template from the presentation list.

At two to four candidates, the board fills the available width. At five to eight candidates, the action columns keep a readable minimum width and scroll horizontally. The Round and context rail remains sticky at the left while the user scrolls across candidates. Column headers remain sticky while the user scrolls through rounds.

The choice trail calculates horizontal positions from the active candidate count. Every selected candidate, observed click or no-click result, score, and unobserved outcome retains the existing semantics.

## Compatibility

Existing saved progress and requests without candidate data use the default three candidates. No migration prompt is needed. The recommendation route continues to start automatically with the current default experience.

No generic arbitrary-action framework is introduced. Candidate configuration is owned by the recommendation scenario because its names, visual types, reward meaning, and coefficient editor are specific to that environment.

## Error handling

The interface prevents expected input errors before a request is sent. The worker independently rejects malformed or incompatible data with the existing recoverable invalid-request response.

Errors include:

- fewer than two or more than eight candidates;
- empty or case-insensitively duplicated names;
- missing or duplicated stable IDs;
- unsupported visual types;
- a coefficient matrix with the wrong number of rows or columns;
- non-finite coefficient values;
- a selected-arm index outside the current candidate list.

If a configured run fails to start, Run Settings remains open and the current run is not presented as having accepted the draft.

## Testing

Python tests cover policy construction and full seeded runs with two, three, and eight candidates. They verify deterministic coefficient generation, exact reset replay, matching context and coefficient dimensions, generated Python, validation failures, and stable behavior after reorder and removal.

TypeScript tests cover the variable-length protocol, candidate validation, default compatibility, choice-trail geometry, dynamic table metadata, and configuration updates by stable ID.

Browser tests cover adding and naming a candidate, choosing its type, reordering, removing, editing its coefficients, starting the run, advancing a round, and seeing the same candidate order in Decision History and Developer View. Accessibility checks cover error association and candidate controls. Visual tests cover four and eight candidates at desktop width plus the eight-candidate board at narrow width without page-level overflow.

## Out of scope

This change does not add arbitrary icons, file uploads, remote catalogs, persistent server storage, candidate-specific feature schemas, ranked recommendations, delayed rewards, or dynamic candidates during an active run. It does not change the candidate count for other scenarios or policy lessons.
