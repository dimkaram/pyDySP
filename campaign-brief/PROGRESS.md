# Campaign progress log

Claude Code: update this file after every stage, before you stop for review.
Keep entries short and factual. Do not delete earlier entries.

## Environment baseline (fill in at the start of Stage 1)

- Python version in use: 3.13.17 (container default). Stage 1 suite also run on
  3.11.17: all pass. New code also parses with Python 3.10 grammar, since CI
  still tests 3.10 until Stage 6.
- `pytest -q` result before any changes: 52 passed. Note: the `pytest` on PATH is
  a separate uv tool without numpy or matplotlib, so use `python -m pytest`.
- `ruff check .` and `black --check .` result before any changes: **both already
  fail on existing code.** ruff 0.16.9, no project config: 130 errors (mostly
  UP045, I001, UP037; 63 in `test.py`, 32 in `channel.py`). black 26.5.1: would
  reformat 5 files (`__init__.py`, `helpers.py`, `test_helpers.py`,
  `test_channel.py`, `test_test.py`). CI runs only pytest. New files are kept
  clean; existing files are left as they are (see Open questions).
- Sphinx build warnings before any changes: 2 (`make html` in `docs/`); one is
  `html_static_path entry '_static' does not exist`.
- Where `__version__` is defined: `src/pydysp/__init__.py` (hard-coded
  "0.2.2"), and separately `version` in `pyproject.toml`. `docs/source/conf.py`
  imports it.
- How `demo.ipynb` is included in the docs (copy, symlink or Sphinx setting):
  **it is not included.** There is no nbsphinx/myst-nb and no toctree entry; the
  only link is the Binder badge in `README.md`. Relevant to Stage 5.

## Stages

- [x] Stage 1: Core `Campaign` (`design-campaign.md`)
- [x] Stage 2: Scalar results and plotting (`design-results.md`)
- [x] Stage 3: Curve results (`design-results.md`)
- [x] Stage 4: Documentation (`delivery.md`)
- [ ] Stage 5: Notebooks (`delivery.md`)
- [ ] Stage 6: Release preparation (`delivery.md`)

Mark a stage done only when its tests pass, `ruff` and `black` are clean, and it
is committed.

## Decisions

Record every point where the brief was unclear and the author decided, with the
date and the decision.

| Date | Stage | Question | Decision |
|---|---|---|---|
| 2026-10-28 | 1 | Where is `number` stored? | Parallel `Campaign.numbers: list[int]` field, kept in step with `tests`. `Test` unchanged. |
| 2026-10-28 | 1 | Default of `number_from` in `from_folder` | `None` (the brief's value was an example). |
| 2026-10-28 | 1 | How `number_from` matches | Glob style, case-insensitive, against the file name **without extension**. `{number}` = one or more digits; leading zeros fine. Exactly one `{number}`. |
| 2026-10-28 | 1 | Duplicate test numbers | `ValueError`. |
| 2026-10-28 | 1 | Merging with what the loader sets | Names and descriptions are replaced only when the campaign is given them; otherwise the loader's values are kept. Tags are added to the loader's tags. Meta is merged, given keys win; `None` values skipped. |
| 2026-10-28 | 1 | `names=` and `name_template=` both given | `ValueError`. |
| 2026-10-28 | 1 | `select()` semantics | Criteria combine with AND; result keeps campaign order. Missing numbers are ignored (ranges with gaps). An unknown name raises `KeyError`. An empty result emits a `UserWarning`. |
| 2026-10-28 | 1 | Indexing | `campaign[i]` returns a `Test`, `campaign[a:b]` a `Campaign`. `get(number=)` or `get(name=)`, `KeyError` if not found. |
| 2026-10-28 | 1 | "Metadata lists" | Means the constructor arguments `descriptions`, `tags`, `meta`. Also added read-only `names`, `descriptions`, `tags` and `meta_values(key, default)` (Stage 2 uses `meta_values` for x-axes). |
| 2026-10-28 | 1 | `info()` / `check_channels()` output | Both return strings (like `Test.info`, `channel_health`). New `missing_channels()` returns `{channel: [missing test numbers]}` for Stage 2 NaN handling and tests. |
| 2026-10-28 | 1 | CSV export columns (**correction to brief**) | Write every column with at least one filled cell; never drop identical columns. `number` and `name` always written. |
| 2026-10-28 | 1 | CSV import details | Empty cell leaves value unchanged; `source_file` not imported; `timestamp` updatable; `tags` cell replaces tags; numeric meta column is `int` if every filled cell is a whole number without a decimal point, else `float`; "short text" is at most 80 characters, no line breaks. |
| 2026-10-28 | 1 | `check_channels()` labels | Tests labelled by real numbers. Ranges `a-b` = 3 or more consecutive tests in campaign order. Reports the shorter of "missing in" / "present only in". Duplicate channel names within a test are also reported. |
| 2026-10-28 | 1 | Campaign name default | Folder name for `from_folder`, empty string otherwise. |
| 2026-10-28 | 1 | Python 3.10 syntax | Avoid 3.11-only features until Stage 6 so CI stays green. |
| 2026-10-29 | 1 | The six Stage 1 choices Claude Code made (`numbers=` + `number_from=` raises; `from_tests(numbers=)`; recursive search sorted by relative path; clashing meta keys skipped with a warning; `with_channel_info_from_csv` warns on no match; channels identified by `name_user`, else `name_input`, case-insensitive) | Approved; keep all. |
| 2026-10-29 | all | Pre-existing ruff/black/Sphinx issues | Fix them in three separate housekeeping commits before Stage 2 (black, then ruff, then Sphinx). Ruff defaults only, no stricter rules; anything not provably identical gets a targeted `noqa` and is listed here for review. |
| 2026-10-29 | 4, 5 | Notebooks in the docs (**overrides delivery.md**) | Sphinx does not build `demo.ipynb` today, so add no notebook extension and no new docs dependency. In Stage 4, add `docs/source/examples.rst` (in the toctree) linking both notebooks on GitHub and their Binder links. This replaces "add both notebooks to the docs toctree". |
| 2026-10-29 | 6 | Lint in CI | When editing the CI matrix in Stage 6, also add `ruff check` and `black --check` steps to `.github/workflows/tests.yml`, as a separate commit so it can be dropped. |
| 2026-10-29 | all | Scope of "ruff and black must pass" | Applies to the whole repo from now on. |
| 2026-10-29 | all | Housekeeping commits `326cf83`, `59ab231`, `7d1dd05` and Stage 2 `a329bac` | Approved. |
| 2026-10-29 | 2 | `"max_abs"` sign | Keep as implemented: signed value of the sample with the largest magnitude, exactly like `Channel.max_abs()` (the author plots the signed peak on the time history). Docs must say it is signed and that `.abs()` gives the magnitude; a test locks it in. `"max"` and `"min"` stay signed. |
| 2026-10-29 | 2 | `"residual"` window | Default `n_last=50` samples (a count, not seconds), the final samples including the last one. Signed; users call `.abs()`. |
| 2026-10-29 | 2 | `cumsum()` and missing values | Agreed: NaN adds nothing and stays NaN. |
| 2026-10-29 | 2 | Line breaks in `plot()` (**overrides design-results.md**) | Break only at NaN values (missing channel or data), never at jumps in test number. A skipped number shows as wider x spacing. Markers stay (`marker="o"` default). The `gaps=` option is removed. |
| 2026-10-29 | 2 | Combining results | Same tests in the same order. `+`/`-` require identical units, else `ValueError`. `*`/`/` combine units into a compound string. |
| 2026-10-29 | all | Plot return values (**overrides the brief and design-results.md**) | New code: a method that draws on one set of axes returns that `ax`; a method that builds a multi-panel figure returns `(fig, ax)`. The figure is `ax.figure`. `CampaignResult.plot()` returns `ax`. Stage 3 overlay and heatmap follow the same rule. Existing methods are not changed. |
| 2026-10-29 | 2 | `response_spectrum` default-periods aliasing | Not fixed on this branch; listed under "Found but not fixed" for a separate issue. |
| 2026-10-29 | all | The 11 `noqa` comments from `59ab231` | Reviewed and accepted as they are. |
| 2026-10-29 | 3 | Stage 3 `1cf18af` and Claude Code's six choices: one class for scalars and curves; complex values shown as magnitude by default with `part=`; heatmap with a colour bar; `tests=[...]` raises `KeyError` for an unknown number; units derived from the channels; linear overlay with legend | Approved. |
| 2026-10-29 | 3 | Units of curve and scalar results | Always derived from the channels' own units (`"g/g"`, `"kN/mm"`, `"g²/Hz"`), never hard-coded. A test with non-g units locks this in. |
| 2026-10-29 | 3 | Heatmap colour scaling | Users can pass `cmap=` and `norm=` (for example `LogNorm()` for a log scale); documented. |
| 2026-10-29 | 3 | Heatmap orientation (**overrides design-results.md and the earlier docstring**) | Like a spectrogram: test number along x, increasing to the right; frequency along y, increasing upward. Each column is centred on its true test number and extends to the midpoints between neighbouring numbers, so a skipped number gives a wider column. A scalar result such as `peak_frequency` can be drawn on top with `ax=`. Missing tests are blank columns. The colour bar stays. |
| 2026-10-29 | 4, 5 | `docs/source/examples.rst` (**overrides delivery.md and the earlier decision**) | Moves from Stage 4 to Stage 5, because the new notebook does not exist yet and its links would be broken. |

## Open questions for the author

List anything waiting for an answer. Remove it once answered, and move the answer
to Decisions.

- None at present.

## Found but not fixed

Problems in existing code found while working on this branch. They are out of
scope here; the author will raise separate issues.

- `Channel.response_spectrum` (and `plot_response_spectrum`): the default
  `periods=np.linspace(0.05, 5.0, 100)` is created once, at import, and
  `response_spectrum` passes it through `np.asarray` (no copy) into the returned
  `ResponseSpectrum.T`. Editing `spec.T` in place therefore changes the default
  grid for every later call. Fix: `periods=None` and build the grid inside, or
  copy the array.
- `Test.to_csv` / `Test.from_csv` do not round-trip channel names when a
  channel has units. With the default `include_axis_labels=True` the column
  header is `label_axis` (for example `"A1 [g]"`), and `from_csv` then names the
  channel `"A1 [g]"`; with `include_axis_labels=False` the headers are
  `Ch1, Ch2, ...`. So `test["A1"]` fails after a CSV round trip. Found while
  checking the Stage 4 doc snippets; Campaign code is not affected. (Possible
  fix: write `name_user` as the header and units in a second row or a sidecar
  file.)

## Plot-method audit (2026-10-29)

Rule for new code: one set of axes returns `ax`; a multi-panel figure returns
`(fig, ax)`. Checked by calling each method with and without `ax=`.

| Method | `ax` given | `ax=None` | Follows rule |
|---|---|---|---|
| `Channel.plot` | same `Axes` | new `Axes` | yes |
| `Channel.plot_fourier` | same `Axes` | new `Axes` | yes |
| `Channel.plot_psd` | same `Axes` | new `Axes` | yes |
| `Channel.plot_arias` | same `Axes` | new `Axes` | yes |
| `Channel.plot_response_spectrum` | same `Axes` | new `Axes` | yes |
| `FourierSpectrum.plot` | same `Axes` | new `Axes` | yes |
| `WelchSpectrum.plot` | same `Axes` | new `Axes` | yes |
| `AriasResult.plot` | same `Axes` | new `Axes` | yes |
| `ResponseSpectrum.plot` | same `Axes` | new `Axes` | yes |
| `Test.plot_transfer_function` | same `Axes` | new `Axes` | yes (with `phase=True` it adds a twin axis but returns the main one) |
| `Test.plot_grid` | no `ax=` parameter | `(Figure, ndarray of Axes)`, plus `str` if `make_caption=True` | yes (multi-panel) |
| `Test.plot_channels` | no `ax=` parameter | as `plot_grid` | yes (multi-panel) |
| `CampaignResult.plot` (new) | same `Axes` | new `Axes` | yes (after Stage 2 fix-up) |

No existing method breaks the rule.

## Stage log

For each finished stage: commit hash, files added or changed, tests added, and
anything the author should look at first.

### Stage 1

Done 2026-10-28. Commit: see `git log` (message "Add Campaign class (Stage 1)").

- Added `src/pydysp/campaign.py`: `Campaign` dataclass (`tests`, `numbers`,
  `name`, `meta`), `from_tests`, `from_files`, `from_folder`, `__len__`,
  `__iter__`, `__getitem__`, `get`, `select`, `names`, `descriptions`, `tags`,
  `meta_values`, `info`, `test_info_to_csv`, `with_test_info_from_csv`,
  `with_channel_info_from_csv(numbers=)`, `missing_channels`, `check_channels`.
  Private helpers for natural sort, case-insensitive glob, `{number}` patterns.
- Changed `src/pydysp/__init__.py`: import and export `Campaign`.
- Added `tests/test_campaign.py`: 49 tests (101 in total, all pass on 3.13 and
  3.11). They cover length mismatches, every numbering source, gaps, duplicates,
  selection, natural sort and case-insensitive matching, recursive search, the
  CSV round trip and header aliases, channel-info CSVs by number range, and the
  missing-channel report.
- `ruff` and `black` clean on the new files. `Channel` and `Test` untouched.
- Look first at: number handling in `from_files`, `with_test_info_from_csv`
  type parsing, and the `check_channels()` wording.
- Approved 2026-10-29. Commit `2640cd9`.

### Housekeeping (before Stage 2)

Three separate commits, 2026-10-29. `python -m pytest`: 101 passed after each,
with the existing tests' assertions unchanged.

- `326cf83` Format existing files with black: removes one blank line in each of
  5 files. Formatting only.
- `59ab231` Fix ruff findings with default rules: 130 to 0. 119 safe auto-fixes
  (annotation syntax, import order, two merged if/elif branches). Manual
  changes, behaviour identical: `ClassVar` annotations on the three `Channel`
  `*_DEFAULTS` dicts; `in dict` instead of `in dict.keys()`; one
  `except Exception` around `int(str)` narrowed to `ValueError`; three unused
  unpacked test variables prefixed with `_`. 11 `noqa` (see Open questions).
  Public signatures, defaults, dataclass fields and type aliases were compared
  before and after: identical.
- `7d1dd05` Fix the two Sphinx build warnings: remove the `html_static_path`
  entry for the missing `_static` folder, and add the blank line before a
  bullet list in the `AriasResult` docstring. Build now has 0 warnings.

### Stage 2

Done 2026-10-29. Commit: see `git log` (message "Add Campaign.metric and
CampaignResult (Stage 2)").

- Added `src/pydysp/result.py`: `CampaignResult` (values, numbers, names, label,
  units, what, meta), `value(number)`, `axis_label`, `abs()`/`abs`/negation,
  `cumsum()`, `+ - * /` with numbers and other results, `to_array()`,
  `to_table()`, `to_csv()`, `plot(x=, ax=, highlight=, origin=, gaps=, xlabel=,
  ylabel=, label=, **plot_kwargs)` returning `(fig, ax)`.
- Changed `src/pydysp/campaign.py`: `Campaign.metric(channel, what,
  processed=True, strict=False, label=None, units=None, **kwargs)` with
  built-ins `max`, `min`, `max_abs`, `rms`, `residual`; a missing channel gives
  NaN and one warning, or `KeyError` with `strict=True`.
- Changed `src/pydysp/__init__.py`: export `CampaignResult`.
- Added `tests/test_result.py`: 22 tests (123 in total). They cover each
  built-in, custom functions, missing channels (NaN and `strict=True`), every
  transform and arithmetic case, export, and plots (labels, `ax=`, gaps,
  `highlight`, `x=` result and meta key, `origin`, NaN).
- Whole repo: `ruff` and `black` clean; Sphinx 0 warnings; Python 3.13 and 3.11.
- Look first at: the `max_abs` sign and `residual` default (Open questions),
  and the look of `highlight` in a real figure.
- Approved 2026-10-29, with changes applied in a fix-up commit ("Stage 2
  fix-up: ..."): `residual` default 50; lines break only at NaN (`gaps=`
  removed); `+`/`-` require equal units; `plot()` returns `ax`; docstrings state
  that `max_abs` is signed; new tests lock in the signed `max_abs`, the 50-sample
  residual window, the unit check and connected lines across skipped numbers
  (125 tests in total).

### Stage 3

Done 2026-10-29. Commit: see `git log` (message "Add curve results:
transfer_function, spectrum, overlay, heatmap (Stage 3)").

- Changed `src/pydysp/result.py`: `CampaignResult` gains curve mode (`freq`,
  `is_curve`, `curve()`, `common_freq()`, `peak_frequency()`), curve
  `to_array()` (2D) and `to_csv(part=)` (frequency down the rows, one column per
  test), and `plot(kind="overlay"|"heatmap", fmin=, fmax=, part=, legend=,
  colorbar=)`. Every plot returns the `ax` it drew on.
- Changed `src/pydysp/campaign.py`: `transfer_function(x, y, kind="H1",
  tests=None, processed=True, strict=False, **csd_kwargs)`,
  `spectrum(channel, method="welch"|"fourier", tests=None, ...)`, and private
  helpers `_for_tests`, `_find_channels` (shared with `metric`).
- Added `tests/test_curves.py`: 20 tests (143 in total; pass on 3.13 and 3.11).
  Synthetic white-noise tests with a resonance that drops through the
  campaign. They cover agreement with `Test.transfer_function` and
  `Channel.welch_psd`/`fourier`, `tests=`, missing channels (NaN/None and
  `strict=True`), `peak_frequency` tracking the resonance, curve transforms and
  type errors, different grids (overlay works; heatmap, `to_array` and `to_csv`
  raise), curve CSV (magnitude and phase), overlay labels and options, heatmap
  rows, labels, colour bar and masking.
- Whole repo: `ruff` and `black` clean; Sphinx 0 warnings.
- Look first at: the one-class design (Open questions), the heatmap's y-axis
  (stripes in campaign order labelled with real numbers), and complex values
  shown as magnitude by default.
- Approved 2026-10-29, with changes applied in a fix-up commit ("Stage 3
  fix-up: ..."): heatmap reoriented (test number along x at the true numbers,
  frequency along y; `peak_frequency` can be drawn on top), explicit `cmap=`
  and `norm=`; tests for cell edges, a single test, the overlay with
  `peak_frequency`, `cmap`/`norm`, blank columns, and units from non-g
  channels (147 tests in total). Units were already taken from the channels;
  the new test locks that in.

### Stage 4

Done 2026-10-29. Commit: see `git log` (message "Document Campaign and
CampaignResult (Stage 4)").

- Added `docs/source/campaign.rst`: "Campaign API" page with a Conventions
  section (signed `max_abs` like `Channel.max_abs()`, `.abs()` for magnitude;
  `n_last` is a count of samples; plot methods return the `Axes`, figure is
  `ax.figure`; missing channels), the test-info CSV format with an example, and
  autodoc for `pydysp.campaign` and `pydysp.result`.
- Changed `docs/source/pydysp.rst`: `Campaign` in the class list and `campaign`
  in the API toctree.
- Changed `docs/source/quickstart.rst`: new section "Working with a Campaign"
  (load a folder, select, metric and transforms, plots with `ax=`, transfer
  function overlay, heatmap with the spectrogram-like orientation and
  `peak_frequency` drawn on top, `cmap`/`norm`, test-info CSV round trip with an
  example file, sensor changes and `check_channels`).
- Changed `docs/source/index.rst`: Features bullet; API line mentions `Campaign`.
- Changed `README.md`: Campaign in the "It provides" list, a "Campaigns" Quick
  Start snippet, a "Campaign" feature list, and table-of-contents entries.
- No changelog file exists in the repo, so none was added. Version not bumped.
- `examples.rst` deferred to Stage 5 (see Decisions).
- Sphinx: 0 warnings (`make html`). `fail_on_warning` unchanged.
- The quickstart and README calls were run end to end on a synthetic folder
  (CSV files loaded with `Test.from_csv`): all ran without warnings, and the
  CSV round trip restored names, tags and `amp`. That check found the
  `to_csv`/`from_csv` naming issue listed under "Found but not fixed".
- Look first at: the quickstart Campaign section (wording and length), the
  Conventions block in `campaign.rst`, and the README snippet.

## Next step

State exactly what the next session should do first.

Wait for the author's review of Stage 4. Then read `delivery.md` (notebooks)
and start Stage 5: the new `examples/campaign_demo.ipynb` (simulated data,
temporary folder, outline in delivery.md, heatmap in the spectrogram
orientation with `peak_frequency` on top), the light touch on `demo.ipynb`,
`docs/source/examples.rst` in the toctree linking both notebooks on GitHub and
Binder (no notebook extension), and a Binder link for each notebook in the
README.
