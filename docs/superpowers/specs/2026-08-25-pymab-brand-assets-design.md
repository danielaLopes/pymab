# PyMAB Brand Assets Design

## Objective

Replace PyMAB's unrelated and letter-based icons with a coherent visual system derived from the approved **Lucky Lever** slot-machine illustration. The system must feel playful and literal while remaining readable in documentation navigation and at favicon sizes.

## Approved Direction

The brand uses two related assets:

1. **Lucky Lever illustration** — the approved full slot-machine artwork, with its deep-teal cabinet, mint trim, gold frame and lever, three white reels, and three gold stars.
2. **Machine Face mark** — the approved third companion concept: a cropped, simplified slot-machine face with exactly three star reels, a gold frame, a small portion of the teal cabinet, and one small right-side lever.

The shared palette is:

- deep teal `#04191D`
- mint `#99E7C6`
- warm gold `#F2C56B`
- off-white `#EFFCF7`

Neither asset contains letters, words, a Python logo, weapons, or additional casino imagery.

## Asset Formats and Ownership

The repository owns one canonical copy of each source asset under `assets/`:

- `assets/pymab-lucky-lever.png` — optimized transparent PNG for the detailed illustration
- `assets/pymab-mark.svg` — deterministic vector recreation of the Machine Face mark
- `assets/pymab-mark.png` — raster export for consumers that cannot use SVG

Consumer-specific copies are generated or synchronized from these canonical assets. Existing `assets/icon.png` and `docs/source/_static/icon.png` are replaced or retired so they cannot silently diverge.

The illustration remains raster because its dimensional highlights are intentional. The companion mark is recreated as simple vector geometry because a generated bitmap cannot reliably serve 16-pixel favicons, high-density displays, and arbitrary documentation sizes.

## Usage Map

### Full Lucky Lever Illustration

- centered above the title and introductory copy in `README.md`
- featured on the root `/pymab/` hub as part of the Arcade destination when that hub is assembled
- available to documentation landing content where a larger editorial illustration is appropriate

The full illustration is not used as a favicon or small navigation glyph.

### Machine Face Companion Mark

- Arcade header, replacing the current letter `P` mark
- Arcade favicon, replacing the inline data-URL favicon
- Sphinx `html_logo` and `html_favicon`
- root hub header mark when the hub is assembled
- default project mark for future social preview metadata

Accessible text remains separate from the image. Decorative instances use empty alternative text or `aria-hidden`; meaningful standalone instances use `PyMAB` as alternative text.

## Visual Behavior

The vector companion mark uses flat fills, a bold dark outline, and no small highlights. The silhouette must remain identifiable at 16, 32, 64, and 128 pixels. At the smallest sizes, the three light reel columns and the right-side lever must remain distinct; minor cabinet detail may be omitted.

The README illustration is constrained to a reasonable display width and retains its native aspect ratio. It must not dominate the first viewport on typical desktop displays and must scale down to the available width on mobile.

## Implementation Boundaries

This change updates brand assets and existing references only. It does not redesign the Arcade layout, Sphinx theme, or README prose. The pending GitHub Pages hub may consume the assets later, but its deployment restructuring remains a separate implementation concern.

No remote image hosting is introduced. All README, documentation, and web references use repository-relative or build-relative paths so forks and offline builds remain intact.

## Validation

Implementation is complete when:

- the full illustration has real transparency and renders correctly on both light and dark backgrounds;
- the Machine Face SVG contains no embedded raster image, text, or external dependency;
- the mark is visually checked at 16, 32, 64, and 128 pixels;
- README asset links resolve from GitHub's renderer;
- the Sphinx warnings-as-errors build succeeds and its generated logo/favicon references resolve;
- the Arcade production build succeeds with the new favicon and header mark;
- unit, accessibility, and relevant screenshot tests pass or are deliberately updated for the approved brand change;
- a repository-wide search finds no remaining references to the retired icon or inline `P` favicon.

## Source Provenance

The approved Lucky Lever artwork and Machine Face concept were generated in the project brainstorming session on 2026-08-25. Final repository assets are derived from those approved concepts and are stored locally in the repository; runtime builds do not depend on the generation service.
