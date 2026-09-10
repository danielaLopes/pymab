# Validated Snapshot Heading Implementation Plan

1. Add a dedicated class to the Full validated snapshot summary in `web/src/components/game/index.tsx`.
2. Apply DM Mono and medium weight to that class in `web/src/styles/index.css`.
3. Add focused regression coverage for the class if an existing inspector component test is available.
4. Run tests, lint, type checking, formatting checks, and the production build.
5. Refresh and inspect the combined local Pages preview.
