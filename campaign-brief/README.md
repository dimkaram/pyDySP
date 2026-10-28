# Campaign brief: index

Add a `Campaign` class to pyDySP (this fork of `dkaramitros/pyDySP`, currently
v0.2.2) that holds an ordered series of `Test` objects from one experimental
campaign and supports analysis and plotting across tests.

## How to use this folder

| File | Contents |
|---|---|
| `README.md` | This index: goal, constraints, non-goals, stages |
| `design-campaign.md` | Loading, numbering, selection, metadata, test-info CSV, channel checks |
| `design-results.md` | Scalar results (`metric`), plotting conventions, curve results |
| `delivery.md` | Tests, documentation, notebooks, release preparation |
| `PROGRESS.md` | Progress log: update it after every stage |

Read this file and `PROGRESS.md` at the start of every session, then the design
file for your current stage.

## Goal and context

`Channel` and `Test` cover a single recording. Nothing covers the whole
campaign, so the author keeps Python lists of files, names and descriptions and
rewrites the same list comprehensions and multi-panel plot blocks in every
notebook.

Typical campaign: shaking-table tests, tens of tests with tens of channels each
(one campaign has 66 channels), loaded from `.mat` files each time. Channels
normally stay the same across tests, but a sensor can occasionally be replaced
and wired to a different channel.

What the class must make easy:

- Load a whole folder of tests, with descriptions and tags (for example hammer,
  white-noise, record).
- Plot a per-test value (max, min, RMS, residual) against test number, with the
  current test highlighted; accumulate it; plot one value against another
  (force against displacement).
- Overlay transfer functions of a few tests, or show how a resonance moves
  through the campaign.
- Keep the test information in a CSV that can be edited and reloaded.
- Handle a sensor change mid-campaign without special cases in analysis code.

This becomes a pull request to the original repository, which the author also
maintains.

## Constraints

- Minimum Python is 3.11 (see `delivery.md`). New code must run on 3.11, 3.12
  and 3.13.
- No new required dependencies: only numpy, scipy, matplotlib and tabulate. No
  pandas.
- Existing behaviour of `Channel` and `Test` does not change, and existing tests
  pass unmodified.
- Follow `test.py` conventions: dataclass style, numpy-style docstrings, new
  objects returned rather than mutating.
- New module `src/pydysp/campaign.py`, exported so that `pydysp.Campaign` works.
- All tests are held in memory. Lazy loading is not needed now, but nothing
  should prevent adding it later.

## Non-goals for this version

- Lazy loading of tests.
- Response spectra across tests.
- A helper that lays several results out on a multi-panel grid. Users build
  grids with `ax=`.
- Modal analysis across a campaign.
- An optional `to_dataframe()`.
- Any plot styling, captions or figure sizes.

## Stages

One commit per stage, each with its tests. Stop after each stage for review.

| # | Stage | Read |
|---|---|---|
| 1 | **Core `Campaign`:** `from_tests`, `from_files`, `from_folder`, numbering, selection, metadata lists, `info()`, the test-info CSV pair, `check_channels()`, and the channel-info method for a range of tests | `design-campaign.md` |
| 2 | **Scalar results:** `metric`, the result class, transforms, export and plotting | `design-results.md` (scalar results, plotting) |
| 3 | **Curve results:** transfer functions, spectra, overlay, heatmap and `peak_frequency` | `design-results.md` (curve results) |
| 4 | **Documentation:** API page, quickstart, README and landing page | `delivery.md` (documentation) |
| 5 | **Notebooks:** the new campaign demo and the light touch on the existing demo | `delivery.md` (notebooks) |
| 6 | **Release preparation:** Python 3.11, the CI matrix and the 0.3.0 bump | `delivery.md` (release preparation) |

Tests are written within each stage; see `delivery.md` (tests) for what they
must cover.
