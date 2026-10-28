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
- [ ] Stage 2: Scalar results and plotting (`design-results.md`)
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

## Open questions for the author

List anything waiting for an answer. Remove it once answered, and move the answer
to Decisions.

- Stage 1 choices made by Claude Code without an explicit ruling (please confirm):
  - `numbers=` and `number_from=` both given raises `ValueError` (the brief lists
    them as a priority order; silently ignoring one seemed error-prone).
  - `from_tests` accepts `numbers=` as well as `name=`.
  - `from_folder` with `recursive=True` matches the pattern against file names
    and sorts by path relative to the folder (sub-folder, then file, natural
    order).
  - Meta keys that clash with standard CSV columns (for example `meta["date"]`)
    are not exported, with a warning.
  - `with_channel_info_from_csv(numbers=...)` warns if no test matches.
  - Channels are identified across tests by `name_user`, else `name_input`,
    case-insensitively (as `Test.__getitem__` does).
- Pre-existing ruff/black failures (see baseline): should the "ruff and black
  must pass" rule apply to new and changed code only? A repo-wide fix would touch
  `Channel`/`Test` files and is out of scope unless you want a separate commit.
- `demo.ipynb` is not in the Sphinx docs at present, so Stage 5's "follow how
  `demo.ipynb` is handled now" has nothing to follow. Decide before Stage 5
  whether to add nbsphinx or myst-nb (new docs dependency) or just link the
  notebooks.

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

## Next step

State exactly what the next session should do first.

Wait for the author's review of Stage 1 and answers to the open questions. Then
read `design-results.md` (scalar results, plotting) and propose a plan for
Stage 2.
