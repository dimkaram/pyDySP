# result.py
from __future__ import annotations

import csv
import numbers as _numbers
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate


@dataclass
class CampaignResult:
    """
    One value per test of a :class:`Campaign`, with the test numbers.

    A ``CampaignResult`` is returned by :meth:`Campaign.metric`. It holds one
    number per test, together with the test numbers (the default x-axis),
    the test names, a label and units, and a copy of each test's ``meta``
    (for plotting against a metadata value). Missing values are NaN.

    Results support ``abs()``, negation, :meth:`cumsum` and arithmetic with
    numbers or other results of the same tests, for example
    ``f_max / d_max``. Results combined with ``+`` or ``-`` must have the
    same units; ``*`` and ``/`` combine the units (``"kN/mm"``). All
    operations return new objects.

    Parameters
    ----------
    values : np.ndarray
        One value per test.
    numbers : list of int
        Test numbers, parallel to ``values``.
    names : list of str
        Test names, parallel to ``values``.
    label : str, optional
        Short label, by default the channel name (for example ``"A1"``).
    units : str, optional
        Units of ``values`` (for example ``"g"``).
    what : str, optional
        What was computed (for example ``"max_abs"``).
    meta : list of dict, optional
        Copy of each test's :attr:`Test.meta`, used by ``plot(x="key")``.
    """

    # One value per test.
    values: np.ndarray
    # Test numbers (default x-axis).
    numbers: list[int]
    # Test names.
    names: list[str]
    # Short label (default: channel name).
    label: str = ""
    # Units of the values.
    units: str | None = None
    # What was computed (metric name or function name).
    what: str = ""
    # Copy of each test's meta, for plotting against a meta key.
    meta: list[dict[str, Any]] = field(default_factory=list)

    __array_priority__ = 1000  # so that ndarray <op> result defers to us

    def __post_init__(self) -> None:
        """Normalise containers and check that all lists match ``values``."""
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 1:
            raise ValueError(
                f"values must be one-dimensional, got shape {self.values.shape!r}."
            )
        n = len(self.values)
        self.numbers = [int(x) for x in self.numbers]
        self.names = list(self.names)
        self.meta = (
            [dict(m) for m in self.meta] if self.meta else [{} for _ in range(n)]
        )
        for label, seq in (
            ("numbers", self.numbers),
            ("names", self.names),
            ("meta", self.meta),
        ):
            if len(seq) != n:
                raise ValueError(
                    f"{label} has {len(seq)} entries but there are {n} values."
                )

    def __repr__(self) -> str:
        return (
            f"CampaignResult(label={self.label!r}, what={self.what!r}, "
            f"units={self.units!r}, n_tests={len(self.values)})"
        )

    def __len__(self) -> int:
        """Return the number of tests."""
        return len(self.values)

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    def value(self, number: int) -> float:
        """
        Return the value for one test number.

        Parameters
        ----------
        number : int
            Test number.

        Returns
        -------
        float
            The value (NaN if it is missing).

        Raises
        ------
        KeyError
            If ``number`` is not in the result.
        """
        try:
            return float(self.values[self.numbers.index(int(number))])
        except ValueError:
            raise KeyError(f"No test with number {number} in the result.") from None

    @property
    def axis_label(self) -> str:
        """Default axis label: ``label`` with ``units`` in brackets."""
        label = self.label or self.what or "value"
        return f"{label} [{self.units}]" if self.units else label

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #

    def _with(self, values: np.ndarray, **changes: Any) -> CampaignResult:
        """Return a copy with new values (and optionally other fields)."""
        kwargs = {
            "values": values,
            "numbers": list(self.numbers),
            "names": list(self.names),
            "label": self.label,
            "units": self.units,
            "what": self.what,
            "meta": self.meta,
        }
        kwargs.update(changes)
        return type(self)(**kwargs)

    def abs(self) -> CampaignResult:
        """Return a new result with absolute values."""
        return self._with(np.abs(self.values))

    def __abs__(self) -> CampaignResult:
        return self.abs()

    def __neg__(self) -> CampaignResult:
        return self._with(-self.values)

    def __pos__(self) -> CampaignResult:
        return self._with(self.values.copy())

    def cumsum(self) -> CampaignResult:
        """
        Return a new result with the cumulative sum in test order.

        Missing (NaN) values add nothing to the running total and stay NaN
        in the output, so one missing test does not hide all later values.

        Returns
        -------
        CampaignResult
            Accumulated values, for example the accumulated residual
            displacement of an LVDT:
            ``camp.metric("L1", "residual").abs().cumsum()``.
        """
        missing = np.isnan(self.values)
        out = np.cumsum(np.where(missing, 0.0, self.values))
        out[missing] = np.nan
        return self._with(out)

    def _binary(
        self, other: Any, op: Callable[[Any, Any], Any], symbol: str, reverse: bool
    ) -> CampaignResult:
        """Apply ``op`` with a number or another result of the same tests."""
        if isinstance(other, CampaignResult):
            if other.numbers != self.numbers:
                raise ValueError(
                    "Results must have the same test numbers in the same order "
                    f"to be combined; got {self.numbers} and {other.numbers}."
                )
            a, b = (other, self) if reverse else (self, other)
            if symbol in "+-" and a.units != b.units:
                raise ValueError(
                    f"Cannot {'add' if symbol == '+' else 'subtract'} results with "
                    f"different units ({a.units!r} and {b.units!r})."
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                values = op(a.values, b.values)
            label = f"{a.label or a.what} {symbol} {b.label or b.what}"
            if symbol in "+-":
                units = a.units
            elif a.units and b.units:
                units = f"{a.units}{'·' if symbol == '*' else '/'}{b.units}"
            elif symbol == "/" and b.units:
                units = f"1/{b.units}"
            else:
                units = a.units or b.units
            return self._with(values, label=label, units=units, what="")
        if isinstance(other, _numbers.Number) and not isinstance(other, bool):
            with np.errstate(divide="ignore", invalid="ignore"):
                values = op(other, self.values) if reverse else op(self.values, other)
            return self._with(values)
        return NotImplemented

    def __add__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.add, "+", False)

    def __radd__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.add, "+", True)

    def __sub__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.subtract, "-", False)

    def __rsub__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.subtract, "-", True)

    def __mul__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.multiply, "*", False)

    def __rmul__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.multiply, "*", True)

    def __truediv__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.true_divide, "/", False)

    def __rtruediv__(self, other: Any) -> CampaignResult:
        return self._binary(other, np.true_divide, "/", True)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def to_array(self) -> np.ndarray:
        """Return a copy of the values as a NumPy array."""
        return self.values.copy()

    def to_table(self, tablefmt: str = "github", floatfmt: str = ".6g") -> str:
        """
        Return the values as a printable table (number, name, value).

        Parameters
        ----------
        tablefmt : str, optional
            Table format passed to :func:`tabulate.tabulate`.
        floatfmt : str, optional
            Number format for the values.

        Returns
        -------
        str
            Table string (not printed automatically). Missing values are
            shown as ``nan``.
        """
        rows = [
            [num, name, val]
            for num, name, val in zip(self.numbers, self.names, self.values)
        ]
        return tabulate(
            rows,
            headers=["number", "name", self.axis_label],
            tablefmt=tablefmt,
            floatfmt=floatfmt,
            numalign="right",
            stralign="left",
        )

    def to_csv(self, filename: str, overwrite: bool = True) -> None:
        """
        Write the values to CSV with columns number, name and value.

        Missing values are written as empty cells.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        overwrite : bool, optional
            If ``False`` and the file exists, a :class:`FileExistsError`
            is raised. Default is ``True``.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists and overwrite=False."
            )
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["number", "name", self.axis_label])
            for num, name, val in zip(self.numbers, self.names, self.values):
                writer.writerow([num, name, "" if np.isnan(val) else repr(float(val))])

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def plot(
        self,
        x: CampaignResult | str | None = None,
        ax: plt.Axes | None = None,
        highlight: int | Iterable[int] | None = None,
        origin: bool = False,
        xlabel: str | None = None,
        ylabel: str | None = None,
        label: str | None = None,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the values against test number, a metadata value or another result.

        Points are joined in test order. The line is broken only where a
        value is missing (NaN); skipped test numbers simply show as wider
        spacing on the x-axis. No figure size, font or colour is set, so the
        user's Matplotlib style applies.

        Like the other single-axes plot methods in pyDySP, this draws on one
        set of axes and returns those axes; the figure is ``ax.figure``.

        Parameters
        ----------
        x : CampaignResult or str, optional
            What to plot on the x-axis:

            * ``None`` (default): test number.
            * str: a :attr:`Test.meta` key, for example ``"amp"``. Tests
              without the key, or with a non-numeric value, are not drawn.
            * CampaignResult: another result of the same tests, for example
              ``f_max.plot(x=d_max)`` for force against displacement.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If ``None``, a new figure and axes are created.
        highlight : int or iterable of int, optional
            Test number(s) to mark with a square marker.
        origin : bool, optional
            If ``True``, start the line at ``(0, 0)``. Default ``False``.
        xlabel, ylabel : str, optional
            Axis labels. By default the y label is the result's label and
            units, and the x label is ``"Test number"``, the meta key, or the
            other result's label and units.
        label : str, optional
            Legend label for the line. Default: the result's label.
        **plot_kwargs
            Passed to :meth:`matplotlib.axes.Axes.plot`. The default marker is
            ``"o"``, so isolated points stay visible.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on (``ax`` if given, otherwise new axes). The
            figure is ``ax.figure``.

        Raises
        ------
        KeyError
            If a ``highlight`` number is not in the result.
        ValueError
            If ``x`` is a result with different test numbers.
        """
        y = self.values
        if x is None:
            xv = np.asarray(self.numbers, dtype=float)
            default_xlabel = "Test number"
        elif isinstance(x, str):
            xv = np.array([_to_float(m.get(x)) for m in self.meta])
            default_xlabel = x
        elif isinstance(x, CampaignResult):
            if x.numbers != self.numbers:
                raise ValueError(
                    "x must have the same test numbers in the same order; "
                    f"got {x.numbers} and {self.numbers}."
                )
            xv = x.values
            default_xlabel = x.axis_label
        else:
            raise TypeError("x must be None, a meta key (str) or a CampaignResult.")

        hl_idx: list[int] = []
        if highlight is not None:
            hl = [highlight] if isinstance(highlight, _numbers.Integral) else highlight
            for num in hl:
                if int(num) not in self.numbers:
                    raise KeyError(f"No test with number {num} in the result.")
                hl_idx.append(self.numbers.index(int(num)))

        line_x, line_y = xv, y
        if origin:
            line_x = np.concatenate([[0.0], line_x])
            line_y = np.concatenate([[0.0], line_y])

        if ax is None:
            _, ax = plt.subplots()
        plot_kwargs.setdefault("marker", "o")
        (line,) = ax.plot(
            line_x, line_y, label=self.label if label is None else label, **plot_kwargs
        )
        if hl_idx:
            ax.plot(
                xv[hl_idx],
                y[hl_idx],
                linestyle="none",
                marker="s",
                markersize=1.6 * line.get_markersize(),
                markerfacecolor="none",
                markeredgecolor=line.get_color(),
                markeredgewidth=1.5,
                label="_nolegend_",
            )
        ax.set_xlabel(default_xlabel if xlabel is None else xlabel)
        ax.set_ylabel(self.axis_label if ylabel is None else ylabel)
        if x is None:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return ax


def _to_float(value: Any) -> float:
    """Convert a meta value to float; missing or non-numeric values become NaN."""
    if value is None or isinstance(value, bool):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
