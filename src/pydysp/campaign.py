# campaign.py
from __future__ import annotations

import csv
import math
import os
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from tabulate import tabulate

from .channel import Channel
from .result import CampaignResult, _combine_units
from .test import Test

# Loader signature: path -> Test
_Loader = Callable[[str], Test]

# Metric signature: Channel -> number
_MetricFunction = Callable[..., float]

# Maximum length of a text ``meta`` value written by ``test_info_to_csv``.
_MAX_TEXT_LEN = 80

# Standard columns of the test-info CSV, in export order.
_CSV_FIELDS = ("number", "name", "description", "tags", "timestamp", "source_file")

# Header aliases for ``with_test_info_from_csv`` (normalised with ``_norm_header``).
_CSV_ALIASES = {
    "number": {"number", "num", "no", "nr", "#", "testnumber", "testno", "testnum"},
    "name": {"name", "testname", "title"},
    "description": {"description", "desc", "testdescription"},
    "tags": {"tags", "tag"},
    "timestamp": {"timestamp", "date", "datetime", "time"},
    "source_file": {"sourcefile", "source", "file", "filename", "path"},
}

_INT_RE = re.compile(r"[+-]?\d+")


@dataclass
class Campaign:
    """
    Ordered series of :class:`Test` objects from one experimental campaign.

    A ``Campaign`` holds the tests of a campaign in order, together with a
    test ``number`` for each of them. The ``number`` is the test's own label
    (the one people say aloud): it stays with the test through every
    selection. The ``index`` of a test is its 0-based position in the
    campaign currently held, as used by ``campaign[0]``.

    Per-test information is carried by the tests themselves
    (:attr:`Test.name`, :attr:`Test.description`, :attr:`Test.tags` and
    :attr:`Test.meta`). Selections and updates return new ``Campaign``
    objects; the original is never modified.

    Parameters
    ----------
    tests : list of Test, optional
        Ordered list of :class:`Test` objects.
    numbers : list of int, optional
        Test numbers, one per test and all different. If empty, tests are
        numbered ``1, 2, ..., len(tests)``.
    name : str, optional
        Name of the campaign.
    meta : dict, optional
        Free-form campaign-level metadata.
    """

    # Ordered list of tests.
    tests: list[Test] = field(default_factory=list)
    # Test numbers, parallel to ``tests``.
    numbers: list[int] = field(default_factory=list)
    # Name of the campaign.
    name: str = ""
    # Free-form campaign-level metadata.
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """
        Normalise containers and validate tests and numbers.

        Raises
        ------
        TypeError
            If an entry of ``tests`` is not a :class:`Test`.
        ValueError
            If ``numbers`` does not match ``tests`` in length or contains
            duplicates.
        """
        self.tests = list(self.tests)
        for i, t in enumerate(self.tests):
            if not isinstance(t, Test):
                raise TypeError(f"Campaign.tests[{i}] is not a Test instance.")
        if not self.numbers:
            self.numbers = list(range(1, len(self.tests) + 1))
        else:
            self.numbers = [_as_int(n, "numbers") for n in self.numbers]
        if len(self.numbers) != len(self.tests):
            raise ValueError(
                f"numbers has {len(self.numbers)} entries but there are "
                f"{len(self.tests)} tests."
            )
        _check_unique(self.numbers)
        self.name = self.name or ""
        self.meta = dict(self.meta)

    def __repr__(self) -> str:
        return (
            f"Campaign(name={self.name!r}, n_tests={len(self.tests)}, "
            f"numbers={self.numbers!r})"
        )

    # ------------------------------------------------------------------ #
    # Container protocol and lookup
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the number of tests in the campaign."""
        return len(self.tests)

    def __iter__(self) -> Iterator[Test]:
        """Iterate over the tests in order."""
        return iter(self.tests)

    def __getitem__(self, key: int | slice) -> Test | Campaign:
        """
        Return a test by index, or a sub-campaign by slice.

        Parameters
        ----------
        key : int or slice
            0-based position (negative values count from the end), or a
            slice of positions.

        Returns
        -------
        Test or Campaign
            The test at ``key``, or a new :class:`Campaign` for a slice.
            Test numbers are kept.

        Raises
        ------
        TypeError
            If ``key`` is neither an integer nor a slice.
        """
        if isinstance(key, slice):
            return self._subset(range(len(self.tests))[key])
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return self.tests[key]
        raise TypeError(
            f"Campaign.__getitem__ only supports int or slice keys, got {type(key)!r}. "
            "Use get(number=...) or get(name=...) to look tests up by number or name."
        )

    def get(self, number: int | None = None, name: str | None = None) -> Test:
        """
        Look up a single test by number or by name.

        Parameters
        ----------
        number : int, optional
            Test number.
        name : str, optional
            Test name (case-insensitive exact match). If several tests share
            the name, the first is returned.

        Returns
        -------
        Test
            The matching test.

        Raises
        ------
        ValueError
            If neither or both of ``number`` and ``name`` are given.
        KeyError
            If no test matches.
        """
        if (number is None) == (name is None):
            raise ValueError("Give exactly one of number= or name=.")
        if number is not None:
            n = _as_int(number, "number")
            for num, t in zip(self.numbers, self.tests):
                if num == n:
                    return t
            raise KeyError(f"No test with number {n} in the campaign.")
        key = str(name).lower()
        for t in self.tests:
            if (t.name or "").lower() == key:
                return t
        raise KeyError(f"No test named '{name}' in the campaign.")

    # ------------------------------------------------------------------ #
    # Metadata lists
    # ------------------------------------------------------------------ #

    @property
    def names(self) -> list[str]:
        """List of test names, in campaign order."""
        return [t.name for t in self.tests]

    @property
    def descriptions(self) -> list[str | None]:
        """List of test descriptions, in campaign order."""
        return [t.description for t in self.tests]

    @property
    def tags(self) -> list[set[str]]:
        """List of test tag sets (copies), in campaign order."""
        return [set(t.tags) for t in self.tests]

    def meta_values(self, key: str, default: Any = None) -> list[Any]:
        """
        Return one ``Test.meta`` entry for every test.

        Parameters
        ----------
        key : str
            Key in :attr:`Test.meta`.
        default : any, optional
            Value used for tests that do not have ``key``. Default ``None``.

        Returns
        -------
        list
            One value per test, in campaign order.
        """
        return [t.meta.get(key, default) for t in self.tests]

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def info(self, tablefmt: str = "github") -> str:
        """
        Return a summary table of the tests in the campaign.

        The table lists number, name, description, tags, duration and
        channel count for every test.

        The duration is that of the first channel after trimming, taken from
        its time axis and trim window only, so no channel data are processed
        (drift correction, filtering and the like are not run). If the
        channels of a test do not all have the same duration, the test shows
        ``nan`` and a note below the table names it. A test whose trim window
        is invalid shows ``-``.

        Parameters
        ----------
        tablefmt : str, optional
            Table format passed to :func:`tabulate.tabulate`.

        Returns
        -------
        str
            Multi-line string summary (not printed automatically).
        """
        title = self.name or "<unnamed>"
        header = f"Campaign: {title}"
        lines = [header, "-" * len(header), f"Tests: {len(self.tests)}"]
        if self.meta:
            lines.append("\nMetadata:")
            for k, v in self.meta.items():
                lines.append(f"  {k}: {v}")
        rows = []
        mixed: list[int] = []
        for num, t in zip(self.numbers, self.tests):
            duration = _test_duration(t)
            if duration is None:
                dur = "-"
            else:
                dur = f"{duration:.6g}"
                if np.isnan(duration):
                    mixed.append(num)
            rows.append(
                [
                    num,
                    t.name or "-",
                    t.description or "-",
                    ",".join(sorted(t.tags)) if t.tags else "-",
                    dur,
                    t.n_channels,
                ]
            )
        lines.append("")
        lines.append(
            tabulate(
                rows,
                headers=[
                    "number",
                    "name",
                    "description",
                    "tags",
                    "duration [s]",
                    "channels",
                ],
                tablefmt=tablefmt,
                numalign="right",
                stralign="left",
            )
        )
        if mixed:
            lines.append(
                "\nnan: the channels of test(s) "
                f"{', '.join(map(str, mixed))} do not all have the same duration."
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_tests(
        cls,
        tests: Iterable[Test],
        numbers: Sequence[int] | None = None,
        name: str = "",
    ) -> Campaign:
        """
        Construct a campaign from existing :class:`Test` objects.

        Parameters
        ----------
        tests : iterable of Test
            Tests in campaign order.
        numbers : sequence of int, optional
            Test numbers, one per test. Default ``1, 2, ..., len(tests)``.
        name : str, optional
            Name of the campaign.

        Returns
        -------
        Campaign
            New campaign holding ``tests``.
        """
        tests = list(tests)
        if numbers is not None:
            _check_length(numbers, len(tests), "numbers", "tests")
        return cls(tests=tests, numbers=list(numbers or []), name=name)

    @classmethod
    def from_files(
        cls,
        files: Iterable[str | os.PathLike],
        loader: _Loader = Test.from_sofsi_mat,
        names: Sequence[str] | None = None,
        descriptions: Sequence[str | None] | None = None,
        tags: Sequence[Iterable[str] | None] | None = None,
        meta: (
            Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any] | None] | None
        ) = None,
        numbers: Sequence[int] | None = None,
        name_template: str | None = None,
        number_from: str | None = None,
        name: str = "",
    ) -> Campaign:
        """
        Load a campaign from a list of files.

        Each file is loaded with ``loader`` and then updated with the
        optional per-test lists, which are matched to ``files`` in order.

        Parameters
        ----------
        files : iterable of str or path-like
            Data files, in campaign order.
        loader : callable, optional
            Any callable ``path -> Test``. Default :meth:`Test.from_sofsi_mat`.
            A full processing chain can be passed, for example
            ``lambda f: Test.from_sofsi_mat(f).drift_corrected()``.
        names : sequence of str, optional
            Test names. Cannot be combined with ``name_template``. If neither
            is given, the names set by ``loader`` are kept.
        descriptions : sequence of str, optional
            Test descriptions. ``None`` entries keep the loader's description.
        tags : sequence of iterables of str, optional
            Tags per test (for example ``[{"hammer"}, {"white-noise"}]``).
            They are added to the tags set by ``loader``.
        meta : mapping of lists, or sequence of mappings, optional
            Per-test metadata, merged into each :attr:`Test.meta` (given values
            win). The main form is columns, ``{"amp": [0.06, 0.1, ...]}``; a
            list of dicts, one per test, is also accepted. ``None`` values are
            skipped.
        numbers : sequence of int, optional
            Explicit test numbers. Cannot be combined with ``number_from``.
        name_template : str, optional
            Format string for test names using ``{number}`` and ``{stem}``
            (file name without extension), for example
            ``"Test {number:03d}"``.
        number_from : str, optional
            Glob-style pattern with one ``{number}`` placeholder, matched
            (case-insensitively) against each file name without its
            extension, for example ``"SH2601_{number}_*"``. ``{number}``
            matches one or more digits.
        name : str, optional
            Name of the campaign.

        Returns
        -------
        Campaign
            New campaign with one test per file.

        Raises
        ------
        ValueError
            If a per-test list does not match the number of files, if both
            ``names`` and ``name_template`` or both ``numbers`` and
            ``number_from`` are given, if a file does not match
            ``number_from``, or if test numbers are not unique.
        TypeError
            If ``loader`` does not return a :class:`Test`.

        Notes
        -----
        Test numbers come from ``number_from``, else ``numbers``, else
        ``1, 2, ..., len(files)``.
        """
        paths = [os.fspath(f) for f in files]
        n = len(paths)

        # Validate all inputs before loading anything
        if names is not None and name_template is not None:
            raise ValueError("Give names= or name_template=, not both.")
        if numbers is not None and number_from is not None:
            raise ValueError("Give numbers= or number_from=, not both.")
        for arg, label in (
            (names, "names"),
            (descriptions, "descriptions"),
            (tags, "tags"),
            (numbers, "numbers"),
        ):
            if arg is not None:
                _check_length(arg, n, label, "files")
        meta_rows = _meta_rows(meta, n)

        stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        if number_from is not None:
            nums = _numbers_from_pattern(stems, number_from)
        elif numbers is not None:
            nums = [_as_int(x, "numbers") for x in numbers]
        else:
            nums = list(range(1, n + 1))
        _check_unique(nums)

        if name_template is not None:
            try:
                new_names = [
                    name_template.format(number=num, stem=stem)
                    for num, stem in zip(nums, stems)
                ]
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"name_template {name_template!r} may only use "
                    "{number} and {stem}."
                ) from exc
        elif names is not None:
            new_names = list(names)
        else:
            new_names = [None] * n

        tests = []
        for i, path in enumerate(paths):
            test = loader(path)
            if not isinstance(test, Test):
                raise TypeError(
                    f"loader returned {type(test).__name__} for '{path}', "
                    "expected a Test."
                )
            updates: dict[str, Any] = {}
            if new_names[i] is not None:
                updates["name"] = new_names[i]
            if descriptions is not None and descriptions[i] is not None:
                updates["description"] = descriptions[i]
            if tags is not None and tags[i] is not None:
                updates["tags"] = set(test.tags) | _as_tag_set(tags[i])
            if meta_rows[i]:
                updates["meta"] = {**test.meta, **meta_rows[i]}
            tests.append(replace(test, **updates) if updates else test)
        return cls(tests=tests, numbers=nums, name=name)

    @classmethod
    def from_folder(
        cls,
        folder: str | os.PathLike,
        pattern: str = "*.mat",
        recursive: bool = False,
        loader: _Loader = Test.from_sofsi_mat,
        number_from: str | None = None,
        name_template: str | None = None,
        names: Sequence[str] | None = None,
        descriptions: Sequence[str | None] | None = None,
        tags: Sequence[Iterable[str] | None] | None = None,
        meta: (
            Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Any] | None] | None
        ) = None,
        numbers: Sequence[int] | None = None,
        name: str | None = None,
    ) -> Campaign:
        """
        Load every matching file in a folder as a campaign.

        Files are matched to ``pattern`` ignoring case (``*.mat`` matches
        ``.MAT`` and ``.mat``) and sorted in natural order, so ``Test_2``
        comes before ``Test_10``. The optional per-test lists are matched to
        the files in that sorted order.

        Parameters
        ----------
        folder : str or path-like
            Folder to search.
        pattern : str, optional
            Glob-style pattern matched against file names. Default ``"*.mat"``.
        recursive : bool, optional
            If ``True``, also search sub-folders. Files are then sorted by
            their path relative to ``folder``.
        loader, number_from, name_template, names, descriptions, tags, meta, numbers
            As in :meth:`from_files`.
        name : str, optional
            Name of the campaign. Default: the folder name.

        Returns
        -------
        Campaign
            New campaign with one test per matching file.

        Raises
        ------
        FileNotFoundError
            If ``folder`` does not exist.
        ValueError
            If no file matches ``pattern``, or as in :meth:`from_files`.
        """
        folder = os.fspath(folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder '{folder}' not found.")
        files = _find_files(folder, pattern=pattern, recursive=recursive)
        if not files:
            raise ValueError(f"No files matching '{pattern}' found in '{folder}'.")
        if name is None:
            name = os.path.basename(os.path.normpath(folder))
        return cls.from_files(
            files,
            loader=loader,
            names=names,
            descriptions=descriptions,
            tags=tags,
            meta=meta,
            numbers=numbers,
            name_template=name_template,
            number_from=number_from,
            name=name,
        )

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def select(
        self,
        indices: int | slice | Iterable[int] | None = None,
        numbers: int | Iterable[int] | None = None,
        names: str | Iterable[str] | None = None,
        tags: str | Iterable[str] | None = None,
        exclude_tags: str | Iterable[str] | None = None,
        where: Callable[[Test], bool] | None = None,
    ) -> Campaign:
        """
        Return a new campaign with the tests that meet all given criteria.

        Tests keep their numbers and their campaign order. Criteria that are
        not given are not applied; given criteria combine with AND.

        Parameters
        ----------
        indices : int, slice or iterable of int, optional
            0-based positions in this campaign.
        numbers : int or iterable of int, optional
            Test numbers, for example ``range(29, 40)``. Numbers not in the
            campaign are ignored, so ranges with gaps work.
        names : str or iterable of str, optional
            Test names (case-insensitive exact match).
        tags : str or iterable of str, optional
            Keep tests that have any of these tags.
        exclude_tags : str or iterable of str, optional
            Drop tests that have any of these tags.
        where : callable, optional
            Predicate ``Test -> bool``; tests for which it is true are kept.

        Returns
        -------
        Campaign
            New campaign with the selected tests. A warning is emitted if
            it is empty.

        Raises
        ------
        KeyError
            If a name in ``names`` matches no test.
        IndexError
            If an index is out of range.
        """
        n = len(self.tests)
        keep = [True] * n

        if indices is not None:
            if isinstance(indices, slice):
                chosen = set(range(n)[indices])
            else:
                if isinstance(indices, (int, np.integer)):
                    indices = [indices]
                chosen = set()
                for i in indices:
                    i = _as_int(i, "indices")
                    if not -n <= i < n:
                        raise IndexError(
                            f"Index {i} out of range for campaign of {n} tests."
                        )
                    chosen.add(i % n)
            keep = [k and i in chosen for i, k in enumerate(keep)]

        if numbers is not None:
            if isinstance(numbers, (int, np.integer)):
                numbers = [numbers]
            wanted = {_as_int(x, "numbers") for x in numbers}
            keep = [k and num in wanted for k, num in zip(keep, self.numbers)]

        if names is not None:
            if isinstance(names, str):
                names = [names]
            wanted_names = {str(nm).lower(): nm for nm in names}
            have = {(t.name or "").lower() for t in self.tests}
            missing = [nm for key, nm in wanted_names.items() if key not in have]
            if missing:
                raise KeyError(f"No test named {', '.join(map(repr, missing))}.")
            keep = [
                k and (t.name or "").lower() in wanted_names
                for k, t in zip(keep, self.tests)
            ]

        if tags is not None:
            tag_set = _as_tag_set(tags)
            keep = [k and bool(t.tags & tag_set) for k, t in zip(keep, self.tests)]

        if exclude_tags is not None:
            ex_set = _as_tag_set(exclude_tags)
            keep = [k and not (t.tags & ex_set) for k, t in zip(keep, self.tests)]

        if where is not None:
            keep = [k and bool(where(t)) for k, t in zip(keep, self.tests)]

        result = self._subset([i for i, k in enumerate(keep) if k])
        if not result.tests:
            warnings.warn("Selection is empty.", UserWarning, stacklevel=2)
        return result

    def _subset(self, positions: Iterable[int]) -> Campaign:
        """Return a new campaign with the tests at ``positions``."""
        positions = list(positions)
        return type(self)(
            tests=[self.tests[i] for i in positions],
            numbers=[self.numbers[i] for i in positions],
            name=self.name,
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Test-info CSV
    # ------------------------------------------------------------------ #

    def test_info_to_csv(self, filename: str, overwrite: bool = True) -> None:
        """
        Export test-level information to CSV (one row per test).

        Columns are ``number``, ``name``, ``description``, ``tags``,
        ``timestamp`` and ``source_file``, followed by one column per simple
        :attr:`Test.meta` entry (numbers, and text of at most 80 characters
        without line breaks). Bulky entries, such as the SoFSI file header,
        are skipped. ``number`` and ``name`` are always written; other
        columns are written if at least one test has a value. Tags are
        written comma-separated in one cell.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        overwrite : bool, optional
            If ``False`` and the file exists, a :class:`FileExistsError`
            is raised. Default is ``True``.

        See Also
        --------
        with_test_info_from_csv : Read the file back into a campaign.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists and overwrite=False."
            )
        rows: list[dict[str, Any]] = []
        meta_keys: list[str] = []
        reserved = {a for aliases in _CSV_ALIASES.values() for a in aliases}
        skipped: set[str] = set()
        for num, t in zip(self.numbers, self.tests):
            row: dict[str, Any] = {
                "number": num,
                "name": t.name,
                "description": t.description,
                "tags": ",".join(sorted(t.tags)) if t.tags else None,
                "timestamp": t.timestamp,
                "source_file": t.source_file,
            }
            for key, val in t.meta.items():
                if not _is_simple_meta(val):
                    continue
                if _norm_header(key) in reserved:
                    skipped.add(key)
                    continue
                if key not in meta_keys:
                    meta_keys.append(key)
                row[key] = val
            rows.append(row)
        if skipped:
            warnings.warn(
                "Meta keys clash with standard CSV columns and were not written: "
                f"{', '.join(sorted(skipped))}.",
                UserWarning,
                stacklevel=2,
            )

        columns = ["number", "name"]
        for col in [*_CSV_FIELDS[2:], *meta_keys]:
            if any(_csv_cell(r.get(col)) != "" for r in rows):
                columns.append(col)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for r in rows:
                writer.writerow([_csv_cell(r.get(col)) for col in columns])

    def with_test_info_from_csv(self, filename: str) -> Campaign:
        """
        Return a new campaign with test information updated from CSV.

        Rows are matched to tests by ``number`` if that column exists, and
        by ``name`` (case-insensitive) otherwise. Header matching is flexible
        (case, spaces and underscores are ignored) and common aliases are
        accepted, for example ``No`` or ``Test number`` for ``number``.

        * ``name``, ``description`` and ``timestamp`` replace the test's
          values. ``tags`` (comma-separated) replace the test's tags.
        * ``source_file`` is not imported.
        * Any other column goes into :attr:`Test.meta` under its header. A
          column becomes numbers if every filled cell parses as a number
          (integers if all are whole numbers written without a decimal
          point), and text otherwise.
        * Empty cells leave the test unchanged, and tests without a row
          are unchanged.

        Parameters
        ----------
        filename : str
            Path to the CSV file, for example one written by
            :meth:`test_info_to_csv`.

        Returns
        -------
        Campaign
            New campaign with updated tests. Numbers and order are kept.

        Raises
        ------
        FileNotFoundError
            If ``filename`` does not exist.
        ValueError
            If the CSV is empty, has neither a ``number`` nor a ``name``
            column, or a row cannot be matched to a test (or matches a test
            already matched by another row).
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        with open(filename, newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError("CSV file is empty.") from None
            rows = [r for r in reader if any(c.strip() for c in r)]
        if not any(h.strip() for h in header):
            raise ValueError("CSV has no header.")

        # Map columns to standard fields or meta keys
        col_field: dict[int, str] = {}
        col_meta: dict[int, str] = {}
        for i, h in enumerate(header):
            key = _norm_header(h)
            if not key:
                continue
            for fld, aliases in _CSV_ALIASES.items():
                if key in aliases:
                    if fld in col_field.values():
                        raise ValueError(f"Duplicate column for '{fld}' in CSV.")
                    col_field[i] = fld
                    break
            else:
                col_meta[i] = h.strip()
        field_col = {fld: i for i, fld in col_field.items()}
        if "number" not in field_col and "name" not in field_col:
            raise ValueError("CSV needs a 'number' or a 'name' column.")

        def cell(row: list[str], col: int) -> str:
            return row[col].strip() if col < len(row) else ""

        # Parse meta columns: numeric if every filled cell parses as a number
        meta_parse: dict[int, Callable[[str], Any]] = {}
        for col in col_meta:
            filled = [cell(r, col) for r in rows if cell(r, col)]
            meta_parse[col] = _column_parser(filled)

        # Match rows to tests
        pos_by_number = {num: i for i, num in enumerate(self.numbers)}
        pos_by_name: dict[str, list[int]] = {}
        for i, t in enumerate(self.tests):
            pos_by_name.setdefault((t.name or "").lower(), []).append(i)

        new_tests = list(self.tests)
        matched: set[int] = set()
        for row in rows:
            if "number" in field_col:
                raw = cell(row, field_col["number"])
                try:
                    num = _as_int(raw, "number")
                except ValueError:
                    raise ValueError(f"Row has no valid test number: {row}") from None
                pos = pos_by_number.get(num)
                if pos is None:
                    raise ValueError(f"Row could not match any test: {row}")
            else:
                raw = cell(row, field_col["name"])
                hits = pos_by_name.get(raw.lower(), [])
                if not hits:
                    raise ValueError(f"Row could not match any test: {row}")
                if len(hits) > 1:
                    raise ValueError(f"Row matches several tests named '{raw}': {row}")
                pos = hits[0]
            if pos in matched:
                raise ValueError(
                    f"Test number {self.numbers[pos]} matched by more than one row."
                )
            matched.add(pos)

            test = new_tests[pos]
            updates: dict[str, Any] = {}
            for fld in ("name", "description", "timestamp"):
                if fld in field_col:
                    val = cell(row, field_col[fld])
                    if val:
                        updates[fld] = val
            if "tags" in field_col:
                val = cell(row, field_col["tags"])
                if val:
                    updates["tags"] = _as_tag_set(_split_tags(val))
            new_meta = {}
            for col, key in col_meta.items():
                val = cell(row, col)
                if val:
                    new_meta[key] = meta_parse[col](val)
            if new_meta:
                updates["meta"] = {**test.meta, **new_meta}
            if updates:
                new_tests[pos] = replace(test, **updates)

        return type(self)(
            tests=new_tests,
            numbers=list(self.numbers),
            name=self.name,
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------ #
    # Channels
    # ------------------------------------------------------------------ #

    def with_channel_info_from_csv(
        self,
        filename: str,
        numbers: int | Iterable[int] | None = None,
    ) -> Campaign:
        """
        Return a new campaign with channel metadata updated from CSV.

        Applies :meth:`Test.with_channel_info_from_csv` to the selected
        tests. Applying different files to different ranges of tests handles
        a sensor change: give the replacement sensor the same ``name_user``
        in the second file, so analysis code keeps using one name.

        Parameters
        ----------
        filename : str
            Channel-info CSV, as read by :meth:`Test.with_channel_info_from_csv`.
            If channel positions differ between tests, match on
            ``name_input`` rather than ``idx``.
        numbers : int or iterable of int, optional
            Test numbers to update, for example ``range(1, 15)``. Numbers not
            in the campaign are ignored. Default: all tests.

        Returns
        -------
        Campaign
            New campaign; tests not selected are unchanged.
        """
        if numbers is None:
            wanted = set(self.numbers)
        else:
            if isinstance(numbers, (int, np.integer)):
                numbers = [numbers]
            wanted = {_as_int(x, "numbers") for x in numbers}
        if not wanted & set(self.numbers):
            warnings.warn(
                "No test in the campaign matches numbers=; nothing was updated.",
                UserWarning,
                stacklevel=2,
            )
        new_tests = [
            t.with_channel_info_from_csv(filename) if num in wanted else t
            for num, t in zip(self.numbers, self.tests)
        ]
        return type(self)(
            tests=new_tests,
            numbers=list(self.numbers),
            name=self.name,
            meta=dict(self.meta),
        )

    def missing_channels(self) -> dict[str, list[int]]:
        """
        Return the channels that are missing from some tests.

        Channels are identified by ``name_user`` (or ``name_input`` if no
        user name is set), compared case-insensitively.

        Returns
        -------
        dict
            Maps each channel name that is missing from at least one test to
            the list of test numbers it is missing from, in campaign order.
            Channels are listed in order of first appearance. Empty if every
            channel exists in every test.
        """
        present = [{k.lower() for k in _channel_keys(t)} for t in self.tests]
        order: dict[str, str] = {}
        for t in self.tests:
            for k in _channel_keys(t):
                order.setdefault(k.lower(), k)
        result: dict[str, list[int]] = {}
        for low, display in order.items():
            missing = [
                num for num, keys in zip(self.numbers, present) if low not in keys
            ]
            if missing:
                result[display] = missing
        return result

    def check_channels(self) -> str:
        """
        Return a short report of channels missing from some tests.

        For each channel missing from at least one test, the report gives
        the shorter of the tests it is missing from and the tests it is
        present in, for example ``Channel A12: missing in tests 14, 15`` or
        ``Channel A4b: present only in tests 14-35``. A range ``a-b`` means
        consecutive tests in campaign order. Tests with two channels of the
        same name are also reported, since lookups by name are then
        ambiguous.

        Returns
        -------
        str
            Multi-line report (not printed automatically).

        See Also
        --------
        missing_channels : The same information as a dict.
        """
        title = self.name or "<unnamed>"
        header = f"Channel check: {title}"
        lines = [header, "-" * len(header)]
        missing = self.missing_channels()
        pos_of = {num: i for i, num in enumerate(self.numbers)}
        for ch, nums in missing.items():
            miss_pos = [pos_of[n] for n in nums]
            pres_pos = sorted(set(range(len(self.tests))) - set(miss_pos))
            if len(miss_pos) <= len(pres_pos):
                lines.append(
                    f"Channel {ch}: missing in tests {self._format_runs(miss_pos)}"
                )
            else:
                lines.append(
                    f"Channel {ch}: present only in tests "
                    f"{self._format_runs(pres_pos)}"
                )
        duplicates = False
        for num, t in zip(self.numbers, self.tests):
            seen: dict[str, str] = {}
            dups: list[str] = []
            for k in _channel_keys(t):
                low = k.lower()
                if low in seen and seen[low] not in dups:
                    dups.append(seen[low])
                seen.setdefault(low, k)
            if dups:
                duplicates = True
                lines.append(f"Test {num}: duplicate channel name(s) {', '.join(dups)}")
        if not missing and not duplicates:
            n_ch = len({k.lower() for t in self.tests for k in _channel_keys(t)})
            lines.append(
                f"All {n_ch} channels are present in all {len(self.tests)} tests."
            )
        return "\n".join(lines)

    def _format_runs(self, positions: Sequence[int]) -> str:
        """Format sorted positions as test numbers, compressing runs to ``a-b``."""
        runs: list[list[int]] = []
        for p in positions:
            if runs and p == runs[-1][-1] + 1:
                runs[-1].append(p)
            else:
                runs.append([p])
        parts = []
        for run in runs:
            nums = [str(self.numbers[p]) for p in run]
            if len(run) >= 3:
                parts.append(f"{nums[0]}-{nums[-1]}")
            else:
                parts.extend(nums)
        return ", ".join(parts)

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #

    def metric(
        self,
        channel: str,
        what: str | _MetricFunction,
        processed: bool = True,
        strict: bool = False,
        label: str | None = None,
        units: str | None = None,
        **kwargs: Any,
    ) -> CampaignResult:
        """
        Reduce one channel in each test to one number.

        Parameters
        ----------
        channel : str
            Channel name (``name_user`` or ``name_input``), as used by
            ``test[channel]``. Use names rather than integer positions, which
            shift silently if the channel layout changes.
        what : str or callable
            Built-in metric name:

            * ``"max"``: largest value (signed).
            * ``"min"``: smallest value (signed).
            * ``"max_abs"``: the value of the sample with the largest
              magnitude, **keeping its sign**, exactly as
              ``Channel.max_abs()[1]``: a negative peak is returned as a
              negative number. Call ``.abs()`` on the result for the
              magnitude.
            * ``"rms"``: root-mean-square value.
            * ``"residual"``: mean of the final ``n_last`` samples, including
              the last one (signed), for example the residual displacement
              after a test. ``n_last`` is a number of samples, not seconds;
              default 50. Call ``.abs()`` on the result for the magnitude.

            Alternatively, any function that receives a :class:`Channel` and
            returns a number. Extra keyword arguments are passed to it.
        processed : bool, optional
            Use processed data for the built-in metrics (default ``True``).
            A custom function receives the channel and decides itself.
        strict : bool, optional
            If ``True``, a test without ``channel`` raises :class:`KeyError`.
            By default its value is NaN and a warning is emitted.
        label : str, optional
            Label of the result. Default: the channel name.
        units : str, optional
            Units of the result. Default: the channel units.
        **kwargs
            Passed to the metric, for example ``n_last=200`` for
            ``"residual"``.

        Returns
        -------
        CampaignResult
            One value per test, with test numbers, names, label and units.

        Raises
        ------
        ValueError
            If ``what`` is an unknown metric name.
        KeyError
            If ``strict=True`` and a test lacks ``channel``.
        TypeError
            If the metric does not return a number.

        Examples
        --------
        >>> a_max = camp.metric("A1", "max_abs")  # doctest: +SKIP
        >>> d_acc = camp.metric("L1", "residual").abs().cumsum()  # doctest: +SKIP
        """
        if isinstance(what, str):
            try:
                func = _METRICS[what]
            except KeyError:
                raise ValueError(
                    f"Unknown metric {what!r}; choose from "
                    f"{', '.join(map(repr, _METRICS))} or pass a function."
                ) from None
            what_name = what
            call_kwargs = {"processed": processed, **kwargs}
        elif callable(what):
            func = what
            what_name = getattr(what, "__name__", "")
            if what_name == "<lambda>":
                what_name = ""
            call_kwargs = dict(kwargs)
        else:
            raise TypeError("what must be a metric name or a function.")

        found, firsts = self._find_channels([channel], strict)
        first = firsts[0]
        values = np.full(len(self.tests), np.nan)
        for i, (num, chans) in enumerate(zip(self.numbers, found)):
            if chans is None:
                continue
            val = func(chans[0], **call_kwargs)
            try:
                values[i] = float(val)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Metric returned {type(val).__name__} for test {num}, "
                    "expected a number."
                ) from None
        if label is None:
            label = _channel_label(first, channel)
        if units is None and first is not None:
            units = first.units
        return CampaignResult(
            values=values,
            numbers=list(self.numbers),
            names=self.names,
            label=label,
            units=units,
            what=what_name,
            meta=[t.meta for t in self.tests],
        )

    def transfer_function(
        self,
        x: str,
        y: str,
        kind: str = "H1",
        tests: Iterable[int] | None = None,
        processed: bool = True,
        strict: bool = False,
        **kwargs: Any,
    ) -> CampaignResult:
        """
        Compute the transfer function between two channels in each test.

        Uses :meth:`Test.transfer_function` for every test and returns one
        complex curve per test.

        Parameters
        ----------
        x : str
            Input (excitation) channel name.
        y : str
            Output (response) channel name.
        kind : {"H1", "H2"}, optional
            Estimator, as in :meth:`Test.transfer_function`. Default ``"H1"``.
        tests : iterable of int, optional
            Test numbers to include, for a quick comparison such as
            ``tests=[3, 29]``. Default: all tests. To work on a group of
            tests (for example only white-noise tests), narrow the campaign
            with :meth:`select` first.
        processed : bool, optional
            Use processed data (default ``True``).
        strict : bool, optional
            If ``True``, a test without ``x`` or ``y`` raises
            :class:`KeyError`. By default it has no curve and a warning is
            emitted.
        **kwargs
            Passed to :func:`scipy.signal.csd`, for example ``nperseg=1024``.
            Use the same ``nperseg`` and sampling interval for all tests if
            you want a heatmap, which needs one shared frequency grid.

        Returns
        -------
        CampaignResult
            Curve result with complex values ``H(f)``, labelled
            ``"<y>/<x>"``. Plot it with ``.plot()`` (overlay) or
            ``.plot(kind="heatmap")``, and get natural frequencies with
            ``.peak_frequency(fmin=..., fmax=...)``.

        Raises
        ------
        KeyError
            If a number in ``tests`` is not in the campaign, or ``strict=True``
            and a test lacks a channel.
        """
        camp = self._for_tests(tests)
        found, (ch_x, ch_y) = camp._find_channels([x, y], strict)
        freqs: list[np.ndarray | None] = []
        curves: list[np.ndarray | None] = []
        for t, chans in zip(camp.tests, found):
            if chans is None:
                freqs.append(None)
                curves.append(None)
                continue
            f, h = t.transfer_function(
                x=chans[0], y=chans[1], kind=kind, processed=processed, **kwargs
            )
            freqs.append(f)
            curves.append(h)
        units = None
        if ch_x is not None and ch_y is not None and (ch_x.units or ch_y.units):
            units = _combine_units(ch_y.units, ch_x.units, "/")
        return CampaignResult(
            values=curves,
            freq=freqs,
            numbers=list(camp.numbers),
            names=camp.names,
            label=f"{_channel_label(ch_y, y)}/{_channel_label(ch_x, x)}",
            units=units,
            what=kind.upper(),
            meta=[t.meta for t in camp.tests],
        )

    def spectrum(
        self,
        channel: str,
        method: str = "welch",
        tests: Iterable[int] | None = None,
        processed: bool = True,
        strict: bool = False,
        **kwargs: Any,
    ) -> CampaignResult:
        """
        Compute the spectrum of one channel in each test.

        Parameters
        ----------
        channel : str
            Channel name.
        method : {"welch", "fourier"}, optional
            ``"welch"`` (default): power spectral density from
            :meth:`Channel.welch_psd`. ``"fourier"``: single-sided Fourier
            amplitude spectrum from :meth:`Channel.fourier`.
        tests : iterable of int, optional
            Test numbers to include. Default: all tests.
        processed : bool, optional
            Use processed data (default ``True``).
        strict : bool, optional
            If ``True``, a test without ``channel`` raises :class:`KeyError`.
            By default it has no curve and a warning is emitted.
        **kwargs
            ``"welch"`` only: passed to :func:`scipy.signal.welch`, for
            example ``nperseg=1024``.

        Returns
        -------
        CampaignResult
            Curve result, one spectrum per test, with the same plots and
            :meth:`CampaignResult.peak_frequency` as transfer functions.

        Notes
        -----
        The Fourier spectrum is zero-padded to the next power of two, so
        tests of different lengths have different frequency grids. That is
        fine for overlays, but a heatmap needs one shared grid: use
        ``"welch"`` with a fixed ``nperseg``, or trim the tests to equal
        length.

        Raises
        ------
        ValueError
            If ``method`` is unknown.
        KeyError
            If a number in ``tests`` is not in the campaign, or ``strict=True``
            and a test lacks the channel.
        """
        if method not in ("welch", "fourier"):
            raise ValueError(f"method must be 'welch' or 'fourier', got {method!r}.")
        if method == "fourier" and kwargs:
            raise TypeError(
                "Extra keyword arguments are only accepted with method='welch'."
            )
        camp = self._for_tests(tests)
        found, (first,) = camp._find_channels([channel], strict)
        freqs: list[np.ndarray | None] = []
        curves: list[np.ndarray | None] = []
        for chans in found:
            if chans is None:
                freqs.append(None)
                curves.append(None)
            elif method == "welch":
                spec = chans[0].welch_psd(processed=processed, **kwargs)
                freqs.append(spec.f)
                curves.append(spec.p)
            else:
                spec = chans[0].fourier(processed=processed)
                freqs.append(spec.f)
                curves.append(spec.s)
        units = first.units if first is not None else None
        if units and method == "welch":
            units = f"{units}²/Hz"
        return CampaignResult(
            values=curves,
            freq=freqs,
            numbers=list(camp.numbers),
            names=camp.names,
            label=_channel_label(first, channel),
            units=units,
            what="psd" if method == "welch" else "fourier",
            meta=[t.meta for t in camp.tests],
        )

    def _for_tests(self, tests: Iterable[int] | None) -> Campaign:
        """Return the campaign restricted to test numbers; unknown ones raise."""
        if tests is None:
            return self
        if isinstance(tests, (int, np.integer)):
            tests = [tests]
        wanted = [_as_int(x, "tests") for x in tests]
        unknown = [n for n in wanted if n not in self.numbers]
        if unknown:
            raise KeyError(
                "Test number(s) not in the campaign: "
                f"{', '.join(map(str, unknown))}."
            )
        return self._subset(i for i, n in enumerate(self.numbers) if n in wanted)

    def _find_channels(
        self, keys: Sequence[str], strict: bool
    ) -> tuple[list[list[Channel] | None], list[Channel | None]]:
        """
        Look up channels by name in every test.

        Returns, per test, the list of channels (``None`` if any is missing)
        and, per key, the first channel found (for labels and units). Missing
        channels raise ``KeyError`` if ``strict``; otherwise one warning per
        channel names the tests that lack it.
        """
        found: list[list[Channel] | None] = []
        firsts: list[Channel | None] = [None] * len(keys)
        missing: dict[str, list[int]] = {}
        for num, t in zip(self.numbers, self.tests):
            chans: list[Channel] = []
            for k, key in enumerate(keys):
                try:
                    ch = t[key]
                except KeyError:
                    if strict:
                        raise KeyError(
                            f"Channel '{key}' not found in test {num}."
                        ) from None
                    missing.setdefault(key, []).append(num)
                    continue
                chans.append(ch)
                if firsts[k] is None:
                    firsts[k] = ch
            found.append(chans if len(chans) == len(keys) else None)
        for key, nums in missing.items():
            warnings.warn(
                f"Channel '{key}' missing in tests "
                f"{', '.join(map(str, nums))}; values set to NaN.",
                UserWarning,
                stacklevel=3,
            )
        return found, firsts


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #


def _natural_sort_key(s: str) -> tuple:
    """
    Key for natural sorting, so ``"Test_2"`` sorts before ``"Test_10"``.

    Parameters
    ----------
    s : str
        String to sort.

    Returns
    -------
    tuple
        Sort key; digit runs compare as integers, text case-insensitively.
    """
    parts = re.split(r"(\d+)", s)
    key = tuple(int(p) if i % 2 else p.lower() for i, p in enumerate(parts))
    return (key, s)


def _find_files(
    folder: str | os.PathLike, pattern: str = "*", recursive: bool = False
) -> list[str]:
    """
    List files in ``folder`` whose names match ``pattern``, ignoring case.

    Parameters
    ----------
    folder : str or path-like
        Folder to search.
    pattern : str, optional
        Glob-style pattern matched against file names (not paths).
    recursive : bool, optional
        If ``True``, also search sub-folders.

    Returns
    -------
    list of str
        Matching file paths, in natural order of their path relative to
        ``folder``.
    """
    folder = os.fspath(folder)
    regex = re.compile(_glob_to_regex(pattern), re.IGNORECASE)
    found: list[tuple[str, str]] = []
    if recursive:
        for root, _dirs, fnames in os.walk(folder):
            for fn in fnames:
                if regex.fullmatch(fn):
                    path = os.path.join(root, fn)
                    found.append((os.path.relpath(path, folder), path))
    else:
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            if os.path.isfile(path) and regex.fullmatch(fn):
                found.append((fn, path))
    found.sort(key=lambda x: _natural_sort_key(x[0].replace(os.sep, "/")))
    return [path for _rel, path in found]


def _glob_to_regex(pattern: str) -> str:
    """Translate a glob pattern (``*``, ``?``, ``[...]``) to a regex string."""
    out = []
    i, n = 0, len(pattern)
    while i < n:
        c = pattern[i]
        i += 1
        if c == "*":
            out.append(".*")
        elif c == "?":
            out.append(".")
        elif c == "[":
            j = pattern.find("]", i + 1 if i < n and pattern[i] in "!]" else i)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j + 1
        else:
            out.append(re.escape(c))
    return "".join(out)


def _numbers_from_pattern(stems: Sequence[str], pattern: str) -> list[int]:
    """Extract test numbers from file stems with a ``{number}`` pattern."""
    if pattern.count("{number}") != 1:
        raise ValueError(
            f"number_from pattern {pattern!r} must contain '{{number}}' exactly once."
        )
    before, after = pattern.split("{number}")
    regex = re.compile(
        _glob_to_regex(before) + r"(\d+)" + _glob_to_regex(after), re.IGNORECASE
    )
    nums = []
    for stem in stems:
        m = regex.fullmatch(stem)
        if m is None:
            raise ValueError(
                f"File name '{stem}' does not match number_from pattern {pattern!r}."
            )
        nums.append(int(m.group(1)))
    return nums


def _meta_rows(meta: Any, n: int) -> list[dict[str, Any]]:
    """Normalise per-test ``meta`` (columns or rows) to one dict per test."""
    if meta is None:
        return [{} for _ in range(n)]
    if isinstance(meta, Mapping):
        rows: list[dict[str, Any]] = [{} for _ in range(n)]
        for key, values in meta.items():
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(
                    f"meta[{key!r}] must be a list with one value per file."
                )
            _check_length(values, n, f"meta[{key!r}]", "files")
            for row, val in zip(rows, values):
                if val is not None:
                    row[key] = val
        return rows
    if isinstance(meta, (str, bytes)) or not isinstance(meta, Sequence):
        raise TypeError(
            "meta must be a dict of lists (columns) or a list of dicts (rows)."
        )
    _check_length(meta, n, "meta", "files")
    rows = []
    for i, row in enumerate(meta):
        if row is None:
            rows.append({})
        elif isinstance(row, Mapping):
            rows.append({k: v for k, v in row.items() if v is not None})
        else:
            raise TypeError(f"meta[{i}] must be a dict or None.")
    return rows


def _check_length(values: Any, n: int, label: str, what: str) -> None:
    """Raise ``ValueError`` naming ``label`` if ``len(values) != n``."""
    if len(values) != n:
        raise ValueError(f"{label} has {len(values)} entries but there are {n} {what}.")


def _check_unique(numbers: Sequence[int]) -> None:
    """Raise ``ValueError`` if ``numbers`` contains duplicates."""
    seen: set[int] = set()
    dups = []
    for x in numbers:
        if x in seen and x not in dups:
            dups.append(x)
        seen.add(x)
    if dups:
        raise ValueError(
            f"Test numbers must be unique; duplicated: {', '.join(map(str, dups))}."
        )


def _as_int(value: Any, label: str) -> int:
    """Convert a whole-number value (int, float or text) to ``int``."""
    if isinstance(value, bool):
        raise TypeError(f"{label} must contain integers, got {value!r}.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must contain integers, got {value!r}.") from None
    if not f.is_integer():
        raise ValueError(f"{label} must contain integers, got {value!r}.")
    return int(f)


def _as_tag_set(tags: str | Iterable[str]) -> set[str]:
    """Return ``tags`` as a set; a single string is one tag."""
    if isinstance(tags, str):
        return {tags}
    return {str(t) for t in tags}


def _split_tags(cell: str) -> list[str]:
    """Split a CSV tags cell on commas or semicolons."""
    return [t.strip() for t in cell.replace(";", ",").split(",") if t.strip()]


def _norm_header(s: str) -> str:
    """Normalise a CSV header: lower case, without spaces, underscores, hyphens."""
    return re.sub(r"[\s_\-]+", "", s.strip().lower())


def _is_simple_meta(val: Any) -> bool:
    """True for numbers and short single-line text."""
    if isinstance(val, (bool, np.bool_)):
        return True
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    if isinstance(val, str):
        return len(val) <= _MAX_TEXT_LEN and "\n" not in val and "\r" not in val
    return False


def _csv_cell(val: Any) -> str:
    """Format a value for a CSV cell; ``None`` and NaN become empty."""
    if val is None:
        return ""
    if isinstance(val, (float, np.floating)) and math.isnan(val):
        return ""
    if isinstance(val, np.generic):
        val = val.item()
    return str(val)


def _column_parser(filled: Sequence[str]) -> Callable[[str], Any]:
    """Choose int, float or text parsing for a CSV column from its filled cells."""
    if not filled:
        return str
    try:
        for v in filled:
            float(v)
    except ValueError:
        return str
    if all(_INT_RE.fullmatch(v) for v in filled):
        return int
    return float


def _channel_duration(ch: Channel) -> float | None:
    """
    Duration of a channel after trimming, from its time axis only.

    Mirrors the trimming step of :meth:`Channel.processed` without
    processing the data (only trimming changes the time axis). Returns
    ``None`` if the trim window is invalid.
    """
    t = ch.time
    if t is None or len(t) == 0:
        return 0.0
    if ch.trim_params:
        t_start = ch.trim_params.get("t_start", float(t[0]))
        t_end = ch.trim_params.get("t_end", float(t[-1]))
        if t_end <= t_start:
            return None
        t = t[(t >= t_start) & (t <= t_end)]
        if len(t) == 0:
            return None
    return float(t[-1] - t[0])


def _test_duration(test: Test) -> float | None:
    """
    Duration of a test for :meth:`Campaign.info`, without processing data.

    The first channel's trimmed duration; NaN if the channels do not all
    have the same duration (same tolerance as :attr:`Test.duration`);
    ``None`` if a trim window is invalid.
    """
    if not test.channels:
        return 0.0
    durations = [_channel_duration(ch) for ch in test.channels]
    if any(d is None for d in durations):
        return None
    first = durations[0]
    if not all(np.isclose(d, first, rtol=1e-6, atol=1e-12) for d in durations[1:]):
        return float("nan")
    return first


def _channel_label(ch: Channel | None, key: str) -> str:
    """Label for a channel: its user or input name, else the key used."""
    if ch is not None:
        return ch.name_user or ch.name_input or str(key)
    return str(key)


def _channel_keys(test: Test) -> list[str]:
    """Channel names used for cross-test checks (``name_user`` or ``name_input``)."""
    keys = []
    for ch in test.channels:
        key = ch.name_user or ch.name_input
        if key:
            keys.append(key)
    return keys


# ---------------------------------------------------------------------- #
# Built-in metrics
# ---------------------------------------------------------------------- #


def _metric_max(ch: Channel, processed: bool = True) -> float:
    """Largest value."""
    return ch.max_value(processed=processed)[1]


def _metric_min(ch: Channel, processed: bool = True) -> float:
    """Smallest value."""
    return ch.min_value(processed=processed)[1]


def _metric_max_abs(ch: Channel, processed: bool = True) -> float:
    """Signed value of the sample with the largest magnitude (as Channel.max_abs)."""
    return ch.max_abs(processed=processed)[1]


def _metric_rms(ch: Channel, processed: bool = True) -> float:
    """Root-mean-square value."""
    return ch.rms(processed=processed)


def _metric_residual(ch: Channel, processed: bool = True, n_last: int = 50) -> float:
    """Signed mean of the final ``n_last`` samples (a count, not seconds)."""
    if n_last < 1:
        raise ValueError(f"n_last must be at least 1, got {n_last}.")
    _, y = ch.xy(processed=processed)
    if y.size == 0:
        raise ValueError("Cannot compute residual of empty signal")
    return float(np.mean(y[-n_last:]))


_METRICS: dict[str, _MetricFunction] = {
    "max": _metric_max,
    "min": _metric_min,
    "max_abs": _metric_max_abs,
    "rms": _metric_rms,
    "residual": _metric_residual,
}
