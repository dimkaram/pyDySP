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
- [ ] Stage 3: Curve results (`design-results.md`)
- [ ] Stage 4: Documentation (`delivery.md`)
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

## Open questions for the author

List anything waiting for an answer. Remove it once answered, and move the answer
to Decisions.

- **Ruff findings left with `noqa` (housekeeping commit `59ab231`), for you to
  decide.** Each has a reason in the code:
  - `src/pydysp/__init__.py`: RUF022, `__all__` not sorted. It is grouped by
    category with comments; sorting would lose that.
  - `src/pydysp/channel.py` (`response_spectrum`, `plot_response_spectrum`):
    B008, `periods=np.linspace(0.05, 5.0, 100)` default. Changing it to
    `None` alters the public signature. Note: `response_spectrum` stores this
    shared default array as `ResponseSpectrum.T` without copying, so a user who
    edits `spec.T` in place changes the default for later calls. A real fix
    (`periods=None` and build the grid inside) is small; say if you want it.
  - `src/pydysp/test.py`: UP007 on `ChannelKey` and `ChannelSelector`, and UP035
    on the `typing` import that carries `Sequence`. Using `|` or
    `collections.abc.Sequence` would change the runtime value of these public
    aliases.
  - `src/pydysp/test.py`: BLE001 (`except Exception`) in four MAT-header parse
    helpers in `from_sofsi_mat` and `from_equals_mat`. Header values are
    arbitrary MAT objects, so narrowing the exceptions could make loading fail
    on files that load today.
  - `src/pydysp/test.py`: BLE001 in `channel_health`, which deliberately reports
    any per-channel failure as a table row.
- **Stage 2 choices made without an explicit ruling (please confirm):**
  - `"max_abs"` returns the value **with its sign**, as `Channel.max_abs()[1]`
    does. For a symmetric signal it is often negative. Use `.abs()` for the
    magnitude. Alternative: make `"max_abs"` return `|value|`.
  - The residual metric is named `"residual"`, with `n_last=100` samples by
    default (`camp.metric("L1", "residual", n_last=500)`). Is 100 a sensible
    default, or should `n_last` be required?
  - Result class `CampaignResult` in new module `src/pydysp/result.py`, exported
    as `pydysp.CampaignResult`. Results store a copy of each test's `meta`, so
    `plot(x="amp")` works without the campaign.
  - Built-ins take `processed=True` (passed to the `Channel` method); a custom
    function gets the channel plus any extra keyword arguments.
  - `.cumsum()` skips NaN: a missing test adds nothing and stays NaN, rather than
    making every later value NaN.
  - Arithmetic between results requires the same test numbers in the same order,
    otherwise `ValueError`. Labels and units combine (`"F1 / L1"`, `"kN/mm"`); for
    `+` and `-` the units are kept only if equal.
  - `plot()` uses `marker="o"` by default (overridable), so isolated points
    between gaps stay visible. Gaps (`gaps=True`) apply only to the test-number
    x-axis; any jump in numbers breaks the line, including after a `select`.
  - `highlight=` accepts one number or a list. It draws a hollow square in the
    line's colour, 1.6 times the marker size. An unknown number raises `KeyError`.
  - With a test-number x-axis, ticks are set to integers (`MaxNLocator`).
  - Plots return `(fig, ax)` as the brief says. Note that
    `Test.plot_transfer_function` actually returns only `ax`; it is unchanged.

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

## Next step

State exactly what the next session should do first.

Wait for the author's review of Stage 2 and answers to the open questions. Then
read `design-results.md` (curve results) and start Stage 3: `transfer_function`,
`spectrum`, overlay and heatmap plots, `peak_frequency`, and curve `to_csv`. The
result class will need a frequency axis and 2-D values.
