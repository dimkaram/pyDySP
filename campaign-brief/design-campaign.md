# Design: the `Campaign` class (Stage 1)

## Loading and numbering

`Campaign` is a dataclass holding `tests: list[Test]`, plus its own `name` and
`meta`. It adds no per-test store of its own: `Test.name`, `description`, `tags`
and `meta` carry the per-test information.

### Constructors

```python
Campaign.from_tests([test1, test2, ...], name="...")

Campaign.from_files(
    files,                  # list of paths
    loader=Test.from_sofsi_mat,
    names=None, descriptions=None, tags=None, meta=None, numbers=None,
    name_template=None, number_from=None, name="...",
)

Campaign.from_folder(
    folder, pattern="*.mat", recursive=False,
    loader=Test.from_sofsi_mat,
    number_from="SH2601_{number}_*",
    name_template="Shallow Foundation - Test {number:03d}",
    descriptions=None, tags=None, meta=None, numbers=None, name="...",
)
```

- **`loader`** is any callable `path -> Test`. This lets the full chain be
  passed, for example
  `lambda f: Test.from_sofsi_mat(f).with_channel_info_from_csv(...).drift_corrected()`.
- **`from_folder`** matches the pattern ignoring case (`.MAT` and `.mat` both
  match, as on Linux CI and Binder) and sorts files in natural order, so
  `Test_2` comes before `Test_10`.
- **Optional parallel lists** (`names`, `descriptions`, `tags`, `numbers`) are
  matched to the files in sorted order. A length mismatch raises an error that
  names the list at fault.
- **`name_template`** builds names from `{number}` (format specs such as
  `{number:03d}` work) and `{stem}` (file name without extension). Without a
  template, names default to the file stem, as `Test` does.

### Two numbers per test

- **`index`**: the Python position, starting at 0, in the campaign currently
  held. Used for `campaign[0]` and `campaign[-1]`. It is recomputed after every
  selection.
- **`number`**: the test's own number, the one people say aloud. It stays with
  the test through every selection and is the default x-axis and label in
  plots. `campaign.get(number=21)` looks a test up by it.

### Where `number` comes from, in priority order

1. `number_from`: a pattern with a `{number}` placeholder matched against the
   file name, for example `"SH2601_{number}_*"`. It works for `Shear_001_1000Hz`
   and `SH2601_29_...` alike, and it handles continuing numbering (29, 30, 31)
   and gaps (a test run but not recorded). A file that does not match raises a
   clear error.
2. `numbers=[...]`: an explicit list. This is the escape hatch, and the way to
   handle file names that contain no number.
3. Default: `index + 1` at load time, so the first test is 1. The same default
   applies to `from_tests`.

Iterating a `Campaign` and `len(campaign)` work, and `campaign.info()` prints a
summary table (number, name, description, tags, duration, channel count) using
`tabulate`.

## Selection, metadata and the test-info CSV

### Selection

Selection returns a new `Campaign` and never mutates, as `Test` does. Tests keep
their `number` through every selection.

```python
camp.select(indices=slice(2, None))
camp.select(numbers=range(29, 40))
camp.select(names=["Test 003"])
camp.select(tags={"white-noise"})            # any of the tags
camp.select(exclude_tags={"hammer"})
camp.select(where=lambda t: t.meta["amp"] > 0.1)
```

This replaces patterns such as the hard-coded `np.arange(2, len(tests))` used to
skip hammer tests.

### Metadata in the constructors

`descriptions` is a list of strings, `tags` is a list of sets, and `meta` is
accepted either as columns (`{"amp": [...], "record": [...]}`, documented as the
main form) or as rows (`[{"amp": 0.06}, ...]`). A dict means columns and a list
means rows. Values are stored in each test's existing `Test.meta`. There is no
`tagger=` function: tags are always given explicitly.

### Test-info CSV

Channel-level CSV already exists (`channel_info_to_csv` and
`with_channel_info_from_csv`). Test-level and campaign-level CSV do not. Mirror
the channel pair:

```python
camp.test_info_to_csv("tests.csv", overwrite=True)
camp = camp.with_test_info_from_csv("tests.csv")   # returns a new Campaign
```

Example file:

```csv
number,name,description,tags,amp
1,Test 001,Hammer Test: Foundation,hammer,
3,Test 003,White Noise 0.06g,white-noise,0.06
4,Test 004,Record 901,record,0.20
```

- **Export** writes `number`, `name`, `description`, `tags`, `timestamp`,
  `source_file` and simple `meta` values (numbers and short text). It skips
  bulky entries such as the SoFSI file header, and it omits columns that are
  empty or identical across tests, as `channel_info_to_csv` does.
- **Import** matches rows to tests by `number` first, or by `name` when there is
  no `number` column. Header matching is flexible (case and underscores
  ignored), with aliases as in `with_channel_info_from_csv`. An unmatched row
  raises `ValueError`, and tests without a row stay unchanged.
- **Extra columns** go into `Test.meta`. A column becomes numbers if every
  filled cell parses as a number, and text otherwise. Tags are comma-separated
  in one cell.
- **Workflow.** Define the lists in a notebook once, export the CSV, and from
  then on load the files and call `with_test_info_from_csv`.

No test-level CSV is added to `Test` itself: a single test would give one row,
and the campaign CSV already carries that information.

## Channel checks and sensor changes

A sensor change is handled by naming, not by an alias map. The existing
channel-info CSV maps DAQ channels to logical names (`name_input` to
`name_user`), so a replaced sensor gets the same `name_user` in a second CSV, and
analysis code keeps using one name.

### Applying a channel-info CSV to a range of tests

```python
camp = camp.with_channel_info_from_csv("channel_info_a.csv", numbers=range(1, 15))
camp = camp.with_channel_info_from_csv("channel_info_b.csv", numbers=range(15, 36))
```

This reuses `Test.with_channel_info_from_csv` for each selected test. Without
`numbers=`, it applies to all tests. If a channel is added or removed so that
positions shift, the CSV should match on `name_input` instead of `idx`, which
the existing reader already supports. The dead channel must be renamed or
dropped (`drop_channels`), because two channels with the same `name_user` in one
test make lookups ambiguous.

### `check_channels()`

Prints a short report of channels missing from some tests, for example
`Channel A12: missing in tests 14, 15` and
`Channel A4b: present only in tests 14-35`. Use it after applying the CSVs to
confirm that every logical channel exists in every test.

### Missing channels in results

When a metric or curve needs a channel that a test lacks, the default is NaN in
that position plus a warning, which shows as a gap in plots. `strict=True`
raises an error instead.

Reference channels by name in campaign code, not by integer index: integer
indices shift silently if the channel layout changes.
