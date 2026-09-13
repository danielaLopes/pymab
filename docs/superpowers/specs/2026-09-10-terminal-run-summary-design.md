# Terminal Run Summary Design

## Goal

Make the collapsed and expanded scenario settings summary easier to scan and more visibly technical.

## Approved direction

Use a single terminal-style status line instead of separate rounded pills. The line starts with a prompt marker, then presents each setting as a `label: value` pair separated by vertical dividers.

Example:

```text
>_ mode: free_play | candidates: 3 | signals: 3 | exploration: 0.96 | learning: 0.65 | l2: 0.36
```

## Visual treatment

- Use the existing monospace interface font for the prompt, labels, and values.
- Use muted text for labels and brighter text for values.
- Keep the prompt marker mint-colored so the line reads as a technical status display.
- Use one compact rectangular container with a modest corner radius.
- Keep the existing Edit settings or Close settings control visually separate from the status line.

## Responsive behavior

The status line stays on one line and scrolls horizontally when space is limited. On the existing small-screen breakpoint, detailed settings may remain hidden so the settings toggle stays compact.

## Scope

This changes only the scenario settings summary in `ScenarioSetupPanel`. It does not change the configuration controls, values, simulation behavior, or validated snapshot data.

## Accessibility

The visible terminal punctuation is decorative. Screen readers receive a natural-language status summary through an accessible label, while all current button semantics remain intact.

## Verification

- Add a component test for the terminal summary and its accessible label.
- Run the component tests, lint, type checking, formatting checks, and the production build.
- Inspect the summary at desktop and narrow viewport widths.
