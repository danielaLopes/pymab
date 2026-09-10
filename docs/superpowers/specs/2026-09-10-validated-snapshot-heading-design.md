# Validated Snapshot Viewer Design

## Goal

Replace the sprawling raw JSON dump with a compact code viewer that remains useful for developers.

## Approved direction

Use the approved code-viewer treatment:

- DM Mono for the disclosure heading and JSON.
- JSON syntax colors for keys, strings, numbers, booleans, and null values.
- Line numbers in a muted gutter.
- Primitive arrays rendered on one line when they fit, including matrix rows and score vectors.
- A fixed maximum height with vertical and horizontal scrolling.
- A Copy JSON action that copies the complete snapshot.
- No rounding, truncation, or mutation of the underlying values.

The viewer remains inside the existing native disclosure control. The rest of Developer view keeps its current typography and layout.

## Accessibility

The scrollable code region has an accessible label and keyboard focus. Copy feedback is announced through a polite live region. Syntax color is decorative and does not replace the JSON text.

## Verification

Confirm formatting and syntax tokenization with focused unit tests. Check copying, keyboard access, overflow, the production build, and the combined local preview.
