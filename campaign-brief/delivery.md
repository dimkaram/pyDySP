# Delivery: tests, docs, notebooks, release (Stages 4 to 6)

## Tests (written within every stage)

Use the repo's pytest setup. Use synthetic `Test` objects built in code, so no
data files are needed, and the Agg backend for plot tests. Cover at least:

- Length mismatches between parallel lists.
- `number` handling: default, `number_from`, `numbers=`, gaps, continuing
  numbering.
- Selection keeping numbers.
- Natural sort order and case-insensitive matching in `from_folder`.
- The CSV round trip.
- Missing channels (NaN and `strict=True`).
- Result transforms.

## Documentation (Stage 4)

Sphinx on Read the Docs.

- New `docs/source/campaign.rst` (autodoc, like the `Channel` and `Test` pages),
  added to the toctree and API reference in `pydysp.rst`.
- A "Campaigns" section in `quickstart.rst`: load a folder, select, `metric`,
  plot, and the CSV round trip.
- A bullet in `index.rst` Features, and entries in the README feature list and
  Quick Start.
- Run the Sphinx build locally and make sure it finishes without new warnings
  (compare with the baseline recorded in `PROGRESS.md`).

## Notebooks (Stage 5)

Two notebooks, aligned in style.

### New `examples/campaign_demo.ipynb`

Uses only simulated data, so it runs on Binder and in the public repo. Outline:

1. Introduction and imports (print the version).
2. A simulated campaign of about 20 tests of an oscillator whose natural
   frequency slowly drops as "damage" accumulates, mixing hammer, white-noise
   and record tests, with an input and a response channel. Save them as CSV files
   in a temporary folder so nothing is left in the repo.
3. Load the folder with `Campaign.from_folder`, with descriptions and tags, and
   print the summary.
4. Selection: exclude hammer tests, pick a range by number, and show `index`
   against `number`.
5. Scalar metrics: `metric("Response", "max_abs")`, `highlight=`, and one result
   against another.
6. Transfer functions: an overlay of a few tests, the heatmap for the
   white-noise tests, and `peak_frequency` against test number.
7. CSV round trip: `test_info_to_csv` then `with_test_info_from_csv`.
8. `check_channels()` on a campaign where one channel is missing from one test.

### Existing `examples/demo.ipynb`: light touch only

Align intro wording, headings and figure settings (`layout="constrained"`) with
the new notebook, add a closing pointer to it, and re-run on 0.3.0 so the printed
version and outputs are current. Content and order stay as they are.

### Shared style for both notebooks

- Title and intro; an imports cell that prints the version.
- A short markdown paragraph per section.
- A one-line comment at the start of each code cell.
- A closing "Next steps" section.
- Outputs saved. Python 3.11 kernel metadata.

Add both notebooks to the docs toctree (follow how `demo.ipynb` is handled now)
and a Binder link for each in the README.

## Release preparation (Stage 6)

Claude Code prepares the release but never performs it. The author tags and
publishes after merging.

- **Python 3.11 minimum.** Python 3.10 reaches end of life on 31 October 2026,
  and current numpy and scipy already require 3.11 or newer. Set
  `requires-python = ">=3.11"` in `pyproject.toml`, remove the 3.10 classifier,
  and update tool settings (ruff, black) and any docs that mention 3.10.
- **CI matrix.** In `.github/workflows/tests.yml`, change the Python matrix from
  3.10, 3.11, 3.12 to 3.11, 3.12, 3.13.
- **Version 0.3.0** in `pyproject.toml` and wherever `__version__` is defined;
  keep them in sync, since `docs/source/conf.py` imports `__version__`. Add a
  changelog entry if one exists. This is a new, backward-compatible feature, so a
  minor bump is right.
- **Checks.** Run `python -m build` and `twine check`, and the local Sphinx
  build. Do not upload anything.
- **Commit separately.** The Python change and the version bump are separate
  commits, so each can be reviewed or adjusted on its own.

### Release flow after merge (for the author)

Merge to `main`; Read the Docs rebuilds "latest"; check the docs; then the
author tags `v0.3.0`, which triggers `release.yml` and publishes to PyPI.
