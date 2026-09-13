# Validated Snapshot Viewer Implementation Plan

1. Add a focused `JsonSnapshotViewer` component under `web/src/components/game/`.
2. Format primitive arrays compactly while preserving exact JSON values.
3. Render safe React text nodes with syntax token classes and line numbers.
4. Add a Copy JSON action with accessible success and failure feedback.
5. Replace the raw snapshot `pre` element in `InspectPanel` with the new component.
6. Add the code-viewer, scrolling, syntax, toolbar, and responsive styles to `web/src/styles/index.css`.
7. Add unit tests for formatting, rendering, and copying.
8. Run all unit tests, lint, type checking, formatting checks, and the production build.
9. Refresh and visually inspect the combined local Pages preview.
