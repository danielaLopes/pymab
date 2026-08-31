# Universal decision history board

## Goal

Replace the Arcade's single-round portal chamber with one shared decision history
board for all 27 policies. The board must retain the playful Moon, Sun, and Star
world while making every past choice, observed reward, and family-specific
decision signal visible.

The board is a visual history of independent bandit rounds. It must not imply
that choosing one path changes the next chamber or that the policy observes
rewards from paths it did not choose.

## Chosen approach

The history board becomes the primary run visualization for every policy. It
replaces the large three-portal chamber rather than appearing below it or behind
a view switcher.

The shared layout has:

- three fixed columns for Moon Path, Sun Path, and Star Path;
- one row for every completed turn;
- a larger current row at the bottom;
- one highlighted chosen cell per completed turn;
- a choice trail connecting selected cells in chronological order;
- a relic for a positive binary reward;
- an empty diamond for a zero binary reward;
- a labelled numeric token for continuous or adversarial rewards;
- a question mark in every unchosen cell because its reward was not observed;
- a compact legend explaining each mark.

The current row shows the latest decision state. Before the first choice, it
shows the three available paths and an awaiting-decision message. After a
choice, it shows the selected path, the observed reward, and the values the
policy used to decide when those values are available.

## Information by policy family

The board shell is identical across families. A presentation adapter supplies
the row label, context rail, cell values, selection reason, and state badges.

### Foundations

- Random shows equal choice probabilities.
- Greedy shows current value estimates.
- Epsilon policies show estimates and an Explore or Exploit badge.
- Softmax and Gradient show action probabilities.

### Optimism

Each cell shows the learned estimate, confidence bonus when available, and the
resulting index. The selected cell emphasizes the index used for that turn.

### Bayesian methods

Each cell shows the decision quantity appropriate to the policy: posterior
sample, posterior mean, or credible upper bound. The label must not call a
sample an estimate.

### Changing environments

The row rail shows the current environment phase when that phase is public.
Change-detection policies add an alarm marker on the turn where a change is
detected. Future phases and hidden probabilities remain hidden in Guided and
Challenge modes.

### Best arm identification

Cells show active, eliminated, or recommended state. The board records samples
and numeric rewards, while the final row emphasizes the policy's recommendation
and simple-regret result.

### Adversarial rewards

Cells show EXP3 selection probabilities. Only the selected reward is shown in
the row. Unchosen rewards remain unknown during the run.

### Contextual bandits

The row rail contains Light, Echo, and Tide for that turn. Each cell shows the
current learned reward estimate or predicted probability, plus uncertainty when
the policy exposes it. Values change between rows because the context changes.

## Fixed and changing environment information

For a public Free Play stationary environment, each column header may show its
configured reward chance or mean because the user entered those values. Guided
and Challenge runs do not reveal hidden environment values before the debrief.

Contextual expected rewards are shown only when they are already public or are
policy predictions. The presentation must distinguish a true configured reward
chance, a learned prediction, a posterior draw, and an optimism index.

There is no hidden-outcome reveal in the first implementation. The simulation
and policy continue to observe only the selected reward.

## Data model

The existing snapshot history is the source of truth. Every history event
already contains:

- selected arm;
- observed reward;
- instantaneous expected regret;
- visible context cues;
- public context;
- explanation key;
- diagnostic state for that turn.

The frontend adds a pure presentation layer that converts a snapshot and its
history events into `DecisionHistoryRow` objects. A row contains only display
data and never reconstructs hidden rewards.

The adapter interface returns:

- turn number and optional family marker;
- visible context items;
- one selected arm;
- observed reward kind and formatted label;
- optional values for all three cells;
- optional per-cell secondary values;
- selection reason;
- active, eliminated, alarm, or recommendation badges.

The protocol changes only if a required diagnostic is missing from historical
events. Any new field must be JSON-safe, validated on the TypeScript boundary,
and generated from policy state captured at the relevant turn.

## Components

`DecisionHistoryBoard` owns the accessible grid, scrolling window, choice trail,
legend, and current-row treatment.

`DecisionHistoryRow` renders the turn rail and three path cells.

`DecisionCell` renders the selected state, reward mark, decision values, and
family badges.

`ContextRail` renders contextual cues, phase markers, or a plain turn label.

`RewardMark` handles binary and numeric rewards without assuming every positive
number is a Bernoulli success.

`PolicyDecisionValues` receives presentation data from the catalog-selected
adapter. It does not inspect raw diagnostics directly.

Existing shadcn/ui primitives provide tooltips and disclosures. The game board
itself remains a custom semantic grid because it is a data visualization rather
than a form control.

## Scrolling and animation

The board keeps all completed rows in the DOM inside a bounded vertical scroll
area. It follows the newest row while rounds advance automatically, but stops
following if the user scrolls upward. A Jump to current turn control restores
automatic following.

The newest row uses a short entrance animation. A thin, non-interactive SVG
choice trail connects the centre of each selected cell. The legend calls it
Choice trail so it reads as history, not movement between causally connected
states. Reduced-motion users receive no path animation.

## Responsive behavior

Desktop shows the turn rail, three columns, and legend together.

Tablet keeps the three path columns visible and moves the legend below the
board.

Mobile keeps the turn rail compact and allows the three path columns to scroll
horizontally as one unit. Column headers remain sticky. The current row and run
controls must remain usable at 320 CSS pixels without page-level horizontal
overflow.

## Accessibility

The board exposes table semantics with column headers, row labels, and cells.
Each chosen cell has an accessible description that includes the turn, path,
observed reward, and decision value. Question marks are announced as Reward not
observed. Color is never the only signal for choice, reward, elimination, or an
alarm.

Tooltips are supplementary. All essential information is present in accessible
text and keyboard-focusable row details.

## Copy

Use Round consistently in the product rather than mixing Round and Turn. Use
plain terms such as Chosen, Relic found, No relic, Reward not observed, Explore,
and Exploit. Avoid em dashes and claims that unobserved rewards are known.

## Testing and acceptance

The implementation is complete when:

1. Every one of the 27 policy routes renders the shared history board.
2. Advancing a round adds exactly one row with the correct selected path and
   observed reward.
3. Unchosen cells never display a simulated reward during a run.
4. Contextual rows preserve their own Light, Echo, and Tide values.
5. Binary, Gaussian, linear, adversarial, and best-arm rewards use the correct
   reward representation.
6. Family-specific values use accurate labels and are derived from the
   historical diagnostic for that row.
7. Guided, Challenge, and Free Play modes retain their existing hidden-value
   rules.
8. Auto-run follows the latest row, manual scrolling can pause following, and
   Jump to current turn restores it.
9. Keyboard and screen-reader labels identify every chosen action and observed
   outcome.
10. The board has no page-level horizontal overflow at 320, 768, and 1440 CSS
    pixels.
11. Deterministic unit, browser, and visual tests cover at least one policy from
    each family, with route-level smoke coverage for all policies.
12. Published copy passes the existing text checker and contains no em dash.

The existing setup panel, run controls, outcome explanation, diagnostics,
debrief, generated Python, and route structure remain in place.
