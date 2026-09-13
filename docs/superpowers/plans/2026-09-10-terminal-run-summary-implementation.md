# Terminal Run Summary Implementation Plan

## Objective

Replace the pill-based scenario configuration summary with the approved terminal-style status line while preserving existing settings behavior.

## Task 1: Render semantic status entries

Update `web/src/components/game/ScenarioSetupPanel.tsx`.

- Build status entries for mode, candidate count, signal count, and scenario parameters.
- Render the entries as label and value spans inside one terminal-style container.
- Give the terminal line a concise natural-language accessible label.
- Keep Edit settings, Close settings, and the expand icon outside the terminal line.

## Task 2: Replace pill styling

Update `web/src/styles/index.css`.

- Remove the rounded pill treatment from summary values.
- Add the shared terminal container, prompt, key, value, and separator styles.
- Prevent wrapping inside the terminal line.
- Allow safe horizontal overflow when the available width is narrower than its contents.
- Preserve the existing small-screen behavior for the settings toggle.

## Task 3: Add regression coverage

Update `web/src/components/game/ScenarioSetupPanel.test.tsx`.

- Verify the terminal status element is present.
- Verify its accessible summary reflects the current mode and settings.
- Keep the existing Free Play behavior tests.

## Task 4: Validate

From `web/`, run:

```bash
npm test -- ScenarioSetupPanel.test.tsx
npm run lint
npm run typecheck
npm run format:check
npm run build
```

Refresh the combined Pages build and inspect the existing local preview at desktop and narrow widths.
