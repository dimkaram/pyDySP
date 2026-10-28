# pyDySP: instructions for Claude Code

## Current task

We are adding a `Campaign` class to pyDySP.

At the start of every session, read these two files before doing anything else:

1. `campaign-brief/README.md` (goal, constraints, stages)
2. `campaign-brief/PROGRESS.md` (what is done, decisions, open questions)

Then read the design file named in the README for the stage you are on. Do not
rely on memory of earlier sessions: the files are the source of truth.

## Working rules

- Work on the feature branch (`feature/campaign`, or the branch created for
  this session). Never commit to `main`.
- Work one stage at a time. Each stage is its own commit. **Stop after each
  stage and wait for my review** before starting the next.
- Before each commit, `pytest`, `ruff` and `black` must pass.
- After each stage, update `campaign-brief/PROGRESS.md`.
- **Flag, don't decide.** If the brief is unclear, contradictory or impossible,
  stop and ask. If a change to `Channel` or `Test` seems necessary, stop and ask
  first. Any such change must be backward-compatible.
- You may push commits to the working branch only. **Never** push to `main`,
  create or push git tags, or upload to PyPI. A pushed `v*` tag triggers
  `.github/workflows/release.yml`, which publishes to PyPI immediately. I do
  merging, tagging and releasing myself.
- `CLAUDE.md` and `campaign-brief/` are working files and will be removed before
  the pull request. Do not edit them, except `campaign-brief/PROGRESS.md`.
- Do not modify the behaviour of existing `Channel` and `Test` methods. Existing
  tests must pass unmodified.

## Project facts

- Source: `src/pydysp/`. Tests: `tests/`. Docs: `docs/source/` (Sphinx, hosted on
  Read the Docs). Examples: `examples/`.
- Conventions (follow `src/pydysp/test.py`): dataclass style, numpy-style
  docstrings, new objects returned rather than mutating.
- Dependencies: numpy, scipy, matplotlib, tabulate only. No pandas.
- Target Python: 3.11 or newer (see `campaign-brief/delivery.md`).
- Setup and checks (confirm against the repo before relying on them):
  - `pip install -e .[dev]`
  - `pytest -q`
  - `ruff check .` and `black --check .`
  - Docs: use the docs `Makefile` if there is one, otherwise
    `sphinx-build -b html docs/source docs/_build/html`. Record the number of
    warnings before you start, so we know the baseline.
