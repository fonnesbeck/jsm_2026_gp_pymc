# README Header Design

## Goal

Replace the existing README title and course-subtitle lines with one integrated header that identifies the course and instructor while preserving the rest of the README unchanged.

## Layout

Use a centered HTML block at the start of `README.md`. In order, it contains:

1. The existing PyMC Labs logo at `assets/pymc-labs-logo.png`, displayed at a controlled width with descriptive alt text.
2. The existing course title as the primary heading.
3. The existing *JSM 2026 Continuing Education Course* subtitle.
4. **Chris Fonnesbeck**.
5. Three affiliations, each on its own line:
   - Principal Data Scientist, PyMC Labs
   - Senior Data Scientist, Leeds United FC
   - Adjoint Associate Professor, Vanderbilt University Medical Center

## Scope

Only the opening header in `README.md` changes. The course description and all subsequent sections remain unchanged.

## Verification

Confirm that the logo path resolves in the repository, the rendered HTML structure is valid for GitHub Markdown, all requested text appears exactly once in the new header, and the previous standalone title and subtitle are removed.
