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

# Ways to turn complex curve values into real numbers for plots and CSV.
_PARTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "abs": np.abs,
    "real": np.real,
    "imag": np.imag,
    "phase": lambda v: np.degrees(np.angle(v)),
}


@dataclass
class CampaignResult:
    """
    One value or one curve per test of a :class:`Campaign`.

    A ``CampaignResult`` is returned by :meth:`Campaign.metric` (a *scalar*
    result: one number per test) and by :meth:`Campaign.transfer_function`
    and :meth:`Campaign.spectrum` (a *curve* result: one frequency curve per
    test). Both kinds hold the test numbers, the test names, a label and
    units, and a copy of each test's ``meta``; they share selection and
    export logic, and only plotting differs.

    Scalar results support ``abs()``, negation, :meth:`cumsum` and arithmetic
    with numbers or other scalar results of the same tests, for example
    ``f_max / d_max``. Results combined with ``+`` or ``-`` must have the
    same units; ``*`` and ``/`` combine the units (``"kN/mm"``). Curve
    results support ``abs()``, negation, arithmetic with numbers and
    :meth:`peak_frequency`, which returns a scalar result. All operations
    return new objects.

    Parameters
    ----------
    values : np.ndarray or list of np.ndarray
        Scalar result: one value per test (missing values are NaN).
        Curve result: one 1D array per test, or ``None`` for a missing test.
        Curve values may be complex (transfer functions).
    numbers : list of int
        Test numbers, parallel to ``values``.
    names : list of str
        Test names, parallel to ``values``.
    label : str, optional
        Short label, by default the channel name (for example ``"A1"``).
    units : str, optional
        Units of ``values`` (for example ``"g"``).
    what : str, optional
        What was computed (for example ``"max_abs"`` or ``"H1"``).
    meta : list of dict, optional
        Copy of each test's :attr:`Test.meta`, used by ``plot(x="key")``.
    freq : list of np.ndarray, optional
        Curve results only: the frequency array (Hz) of each test, or
        ``None`` for a missing test. ``None`` for scalar results.
    """

    # Scalar: one value per test. Curve: one array (or None) per test.
    values: Any
    # Test numbers (default x-axis).
    numbers: list[int]
    # Test names.
    names: list[str]
    # Short label (default: channel name).
    label: str = ""
    # Units of the values.
    units: str | None = None
    # What was computed (metric name, function name, estimator...).
    what: str = ""
    # Copy of each test's meta, for plotting against a meta key.
    meta: list[dict[str, Any]] = field(default_factory=list)
    # Curve results: frequency array per test (None for scalar results).
    freq: list[np.ndarray | None] | None = None

    __array_priority__ = 1000  # so that ndarray <op> result defers to us

    def __post_init__(self) -> None:
        """Normalise containers and check that all lists match ``values``."""
        if self.freq is None:
            self.values = np.asarray(self.values, dtype=float)
            if self.values.ndim != 1:
                raise ValueError(
                    "values must be one-dimensional for a scalar result, got "
                    f"shape {self.values.shape!r}."
                )
        else:
            self.values, self.freq = _check_curves(self.values, self.freq)
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
        kind = "curve" if self.is_curve else "scalar"
        return (
            f"CampaignResult({kind}, label={self.label!r}, what={self.what!r}, "
            f"units={self.units!r}, n_tests={len(self.values)})"
        )

    def __len__(self) -> int:
        """Return the number of tests."""
        return len(self.values)

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    @property
    def is_curve(self) -> bool:
        """``True`` for a curve result (one curve per test)."""
        return self.freq is not None

    def _position(self, number: int) -> int:
        """Return the position of a test number, or raise ``KeyError``."""
        try:
            return self.numbers.index(int(number))
        except ValueError:
            raise KeyError(f"No test with number {number} in the result.") from None

    def value(self, number: int) -> float:
        """
        Return the value for one test number (scalar results).

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
        TypeError
            If this is a curve result; use :meth:`curve`.
        """
        self._require_scalar("value()", "use curve()")
        return float(self.values[self._position(number)])

    def curve(self, number: int) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return the frequency and values of one test (curve results).

        Parameters
        ----------
        number : int
            Test number.

        Returns
        -------
        tuple of np.ndarray or None
            Copies of ``(freq, values)``, or ``None`` if the test has no
            curve (missing channel).

        Raises
        ------
        KeyError
            If ``number`` is not in the result.
        TypeError
            If this is a scalar result; use :meth:`value`.
        """
        self._require_curve("curve()", "use value()")
        i = self._position(number)
        if self.values[i] is None:
            return None
        return self.freq[i].copy(), self.values[i].copy()

    @property
    def axis_label(self) -> str:
        """Default axis label: ``label`` with ``units`` in brackets."""
        label = self.label or self.what or "value"
        return f"{label} [{self.units}]" if self.units else label

    def _require_scalar(self, op: str, hint: str = "") -> None:
        if self.is_curve:
            raise TypeError(
                f"{op} is only available for scalar results"
                + (f"; {hint}." if hint else ".")
            )

    def _require_curve(self, op: str, hint: str = "") -> None:
        if not self.is_curve:
            raise TypeError(
                f"{op} is only available for curve results"
                + (f"; {hint}." if hint else ".")
            )

    # ------------------------------------------------------------------ #
    # Transforms
    # ------------------------------------------------------------------ #

    def _with(self, values: Any, **changes: Any) -> CampaignResult:
        """Return a copy with new values (and optionally other fields)."""
        kwargs = {
            "values": values,
            "numbers": list(self.numbers),
            "names": list(self.names),
            "label": self.label,
            "units": self.units,
            "what": self.what,
            "meta": self.meta,
            "freq": self.freq,
        }
        kwargs.update(changes)
        return type(self)(**kwargs)

    def _map(self, func: Callable[[np.ndarray], np.ndarray]) -> CampaignResult:
        """Apply ``func`` to the values (to each curve for curve results)."""
        if self.is_curve:
            return self._with([None if v is None else func(v) for v in self.values])
        return self._with(func(self.values))

    def abs(self) -> CampaignResult:
        """Return a new result with absolute values (magnitude for complex curves)."""
        return self._map(np.abs)

    def __abs__(self) -> CampaignResult:
        return self.abs()

    def __neg__(self) -> CampaignResult:
        return self._map(np.negative)

    def __pos__(self) -> CampaignResult:
        return self._map(np.copy)

    def cumsum(self) -> CampaignResult:
        """
        Return a new result with the cumulative sum in test order.

        Missing (NaN) values add nothing to the running total and stay NaN
        in the output, so one missing test does not hide all later values.
        Scalar results only.

        Returns
        -------
        CampaignResult
            Accumulated values, for example the accumulated residual
            displacement of an LVDT:
            ``camp.metric("L1", "residual").abs().cumsum()``.
        """
        self._require_scalar("cumsum()")
        missing = np.isnan(self.values)
        out = np.cumsum(np.where(missing, 0.0, self.values))
        out[missing] = np.nan
        return self._with(out)

    def _binary(
        self, other: Any, op: Callable[[Any, Any], Any], symbol: str, reverse: bool
    ) -> CampaignResult:
        """Apply ``op`` with a number or another result of the same tests."""
        if isinstance(other, CampaignResult):
            if self.is_curve or other.is_curve:
                raise TypeError(
                    "Arithmetic between results is only available for scalar "
                    "results; curve results can be combined with numbers."
                )
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
            else:
                units = _combine_units(a.units, b.units, symbol)
            return self._with(values, label=label, units=units, what="")
        if isinstance(other, _numbers.Number) and not isinstance(other, bool):

            def apply(v: np.ndarray) -> np.ndarray:
                with np.errstate(divide="ignore", invalid="ignore"):
                    return op(other, v) if reverse else op(v, other)

            return self._map(apply)
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
    # Curves to scalars
    # ------------------------------------------------------------------ #

    def peak_frequency(
        self, fmin: float | None = None, fmax: float | None = None
    ) -> CampaignResult:
        """
        Return the frequency of the largest magnitude of each curve.

        Parameters
        ----------
        fmin, fmax : float, optional
            Frequency band (Hz, inclusive) to search. Default: the whole curve.

        Returns
        -------
        CampaignResult
            Scalar result in Hz, one value per test (NaN for a missing
            curve), which can be plotted against test number like any
            metric.

        Raises
        ------
        TypeError
            If this is a scalar result.
        ValueError
            If a curve has no frequencies inside ``[fmin, fmax]``.
        """
        self._require_curve("peak_frequency()")
        out = np.full(len(self.values), np.nan)
        for i, (f, v) in enumerate(zip(self.freq, self.values)):
            if v is None:
                continue
            mask = _band(f, fmin, fmax)
            if not mask.any():
                raise ValueError(
                    f"Test {self.numbers[i]} has no frequencies between "
                    f"{fmin} and {fmax} Hz."
                )
            mag = np.abs(v[mask])
            out[i] = f[mask][int(np.argmax(mag))]
        label = f"Peak frequency {self.label}".strip()
        return type(self)(
            values=out,
            numbers=list(self.numbers),
            names=list(self.names),
            label=label,
            units="Hz",
            what="peak_frequency",
            meta=self.meta,
        )

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def common_freq(self) -> np.ndarray:
        """
        Return the frequency grid shared by all curves.

        Returns
        -------
        np.ndarray
            Copy of the common frequency array.

        Raises
        ------
        TypeError
            If this is a scalar result.
        ValueError
            If there are no curves, or the tests do not share one frequency
            grid. Curves are never interpolated.
        """
        self._require_curve("common_freq()")
        grids = [(n, f) for n, f in zip(self.numbers, self.freq) if f is not None]
        if not grids:
            raise ValueError("The result has no curves.")
        ref_num, ref = grids[0]
        bad = [
            (n, f)
            for n, f in grids[1:]
            if f.shape != ref.shape or not np.allclose(f, ref, rtol=1e-9, atol=0.0)
        ]
        if bad:

            def describe(n: int, f: np.ndarray) -> str:
                return f"test {n}: {len(f)} points, df={_df(f):.6g} Hz"

            raise ValueError(
                "Tests do not share one frequency grid ("
                + "; ".join(describe(n, f) for n, f in [(ref_num, ref), *bad[:3]])
                + (", ..." if len(bad) > 3 else "")
                + "). Use the same sampling interval and segment length "
                "(nperseg) for all tests, or select tests with matching grids."
            )
        return ref.copy()

    def to_array(self) -> np.ndarray:
        """
        Return the values as a NumPy array (a copy).

        Returns
        -------
        np.ndarray
            Scalar result: shape ``(n_tests,)``. Curve result: shape
            ``(n_tests, n_freq)`` on the common frequency grid (see
            :meth:`common_freq`), with NaN rows for missing tests.

        Raises
        ------
        ValueError
            For a curve result whose tests do not share one frequency grid.
        """
        if not self.is_curve:
            return self.values.copy()
        f = self.common_freq()
        dtype = complex if any(np.iscomplexobj(v) for v in self.values) else float
        out = np.full((len(self.values), len(f)), np.nan, dtype=dtype)
        for i, v in enumerate(self.values):
            if v is not None:
                out[i] = v
        return out

    def to_table(self, tablefmt: str = "github", floatfmt: str = ".6g") -> str:
        """
        Return the values as a printable table (number, name, value).

        Scalar results only; write curve results with :meth:`to_csv`.

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
        self._require_scalar("to_table()", "write curves with to_csv()")
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

    def to_csv(
        self, filename: str, overwrite: bool = True, part: str | None = None
    ) -> None:
        """
        Write the result to CSV.

        * Scalar result: columns number, name and value, one row per test.
        * Curve result: frequency down the rows and one column per test
          (headed ``"Test <number>"``). All curves must share one frequency
          grid (see :meth:`common_freq`).

        Missing values are written as empty cells.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        overwrite : bool, optional
            If ``False`` and the file exists, a :class:`FileExistsError`
            is raised. Default is ``True``.
        part : {"abs", "real", "imag", "phase"}, optional
            Curve results only: what to write for each value. ``"phase"`` is
            in degrees. Default: ``"abs"`` (magnitude) for complex curves
            such as transfer functions, the values themselves otherwise.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File '{filename}' already exists and overwrite=False."
            )
        if not self.is_curve:
            if part is not None:
                raise TypeError("part= is only available for curve results.")
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["number", "name", self.axis_label])
                for num, name, val in zip(self.numbers, self.names, self.values):
                    writer.writerow([num, name, _cell(val)])
            return
        freq = self.common_freq()
        data = self._real_part(self.to_array(), part)
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frequency [Hz]", *(f"Test {n}" for n in self.numbers)])
            for j, fj in enumerate(freq):
                writer.writerow([repr(float(fj)), *(_cell(v) for v in data[:, j])])

    def _real_part(self, values: Any, part: str | None) -> Any:
        """Convert (possibly complex) curve values to real numbers."""
        if part is None:
            part = "abs" if np.iscomplexobj(values) else None
        if part is None:
            return values
        try:
            func = _PARTS[part]
        except KeyError:
            raise ValueError(
                f"part must be one of {', '.join(map(repr, _PARTS))}, got {part!r}."
            ) from None
        return func(values)

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
        kind: str | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        part: str | None = None,
        legend: bool = True,
        colorbar: bool = True,
        cmap: Any = None,
        norm: Any = None,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the result on one set of axes and return the axes.

        **Scalar results** are plotted against test number (default), a
        metadata value or another result. Points are joined in test order.
        The line is broken only where a value is missing (NaN); skipped test
        numbers simply show as wider spacing on the x-axis.

        **Curve results** are plotted as an overlay (default, one line per
        test labelled ``"Test <number>"``) or as a heatmap
        (``kind="heatmap"``, like a spectrogram: test number along the
        horizontal axis, increasing to the right, frequency along the
        vertical axis, increasing upward, colour showing amplitude). Each
        heatmap column is centred on its true test number and extends to the
        midpoints between neighbouring numbers, so a skipped number gives a
        wider column, as in the scalar plots. Because both use test number as
        x, a scalar result such as :meth:`peak_frequency` can be drawn on top
        of the heatmap with ``ax=``. Missing tests are blank columns.

        No figure size, font or colour scheme is set, so the user's
        Matplotlib style applies. Like the other single-axes plot methods in
        pyDySP, this draws on one set of axes and returns those axes; the
        figure is ``ax.figure``.

        Parameters
        ----------
        x : CampaignResult or str, optional
            Scalar results only. What to plot on the x-axis:

            * ``None`` (default): test number.
            * str: a :attr:`Test.meta` key, for example ``"amp"``. Tests
              without the key, or with a non-numeric value, are not drawn.
            * CampaignResult: another scalar result of the same tests, for
              example ``f_max.plot(x=d_max)`` for force against displacement.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. If ``None``, a new figure and axes are created.
        highlight : int or iterable of int, optional
            Scalar results only. Test number(s) to mark with a square marker.
        origin : bool, optional
            Scalar results only. If ``True``, start the line at ``(0, 0)``.
        xlabel, ylabel : str, optional
            Axis labels. Defaults come from the result's label and units.
        label : str, optional
            Scalar results only. Legend label for the line. Default: the
            result's label.
        kind : {"overlay", "heatmap"}, optional
            Curve results only. Default ``"overlay"``.
        fmin, fmax : float, optional
            Curve results only. Frequency band (Hz) to draw.
        part : {"abs", "real", "imag", "phase"}, optional
            Curve results only. What to draw for each value; ``"phase"`` is
            in degrees. Default: magnitude for complex curves (transfer
            functions), the values themselves otherwise.
        legend : bool, optional
            Overlay only. Add a legend with the test labels (default ``True``).
        colorbar : bool, optional
            Heatmap only. Add a colour bar next to ``ax`` (default ``True``).
        cmap : str or matplotlib.colors.Colormap, optional
            Heatmap only. Colour map, for example ``"magma"``. Default: the
            Matplotlib style's colour map.
        norm : matplotlib.colors.Normalize, optional
            Heatmap only. Colour scaling, for example
            ``matplotlib.colors.LogNorm()`` for a logarithmic colour scale,
            or ``Normalize(vmin=0, vmax=5)`` for fixed limits.
        **plot_kwargs
            Passed to :meth:`matplotlib.axes.Axes.plot` (scalar and overlay)
            or :meth:`matplotlib.axes.Axes.pcolormesh` (heatmap). For scalar
            results the default marker is ``"o"``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on (``ax`` if given, otherwise new axes). The
            figure is ``ax.figure``; for a heatmap the colour bar is
            ``ax.collections[0].colorbar``.

        Raises
        ------
        KeyError
            If a ``highlight`` number is not in the result.
        ValueError
            If ``x`` is a result with different test numbers, ``kind`` is
            unknown, or a heatmap is requested for tests that do not share
            one frequency grid.
        TypeError
            If an option is given that does not apply to this kind of result.
        """
        if self.is_curve:
            if x is not None or highlight is not None or origin or label is not None:
                raise TypeError(
                    "x=, highlight=, origin= and label= apply to scalar results only."
                )
            kind = "overlay" if kind is None else kind
            if kind == "heatmap":
                if cmap is not None:
                    plot_kwargs["cmap"] = cmap
                if norm is not None:
                    plot_kwargs["norm"] = norm
                return self._plot_heatmap(
                    ax, fmin, fmax, part, xlabel, ylabel, colorbar, plot_kwargs
                )
            if cmap is not None or norm is not None:
                raise TypeError("cmap= and norm= apply to kind='heatmap' only.")
            if kind == "overlay":
                return self._plot_overlay(
                    ax, fmin, fmax, part, xlabel, ylabel, legend, plot_kwargs
                )
            raise ValueError(f"kind must be 'overlay' or 'heatmap', got {kind!r}.")
        if (
            kind is not None
            or fmin is not None
            or fmax is not None
            or part
            or cmap is not None
            or norm is not None
        ):
            raise TypeError(
                "kind=, fmin=, fmax=, part=, cmap= and norm= apply to curve "
                "results only."
            )
        return self._plot_scalar(
            x, ax, highlight, origin, xlabel, ylabel, label, plot_kwargs
        )

    def _plot_scalar(
        self,
        x: CampaignResult | str | None,
        ax: plt.Axes | None,
        highlight: int | Iterable[int] | None,
        origin: bool,
        xlabel: str | None,
        ylabel: str | None,
        label: str | None,
        plot_kwargs: dict[str, Any],
    ) -> plt.Axes:
        """Scalar plot: values against test number, a meta key or a result."""
        y = self.values
        if x is None:
            xv = np.asarray(self.numbers, dtype=float)
            default_xlabel = "Test number"
        elif isinstance(x, str):
            xv = np.array([_to_float(m.get(x)) for m in self.meta])
            default_xlabel = x
        elif isinstance(x, CampaignResult):
            if x.is_curve:
                raise TypeError("x must be a scalar result.")
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
            hl_idx = [self._position(num) for num in hl]

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

    def _curve_ylabel(self, part: str | None, complex_values: bool) -> str:
        """Default value-axis label for curve plots."""
        if part is None and complex_values:
            part = "abs"
        label = self.label or self.what or "value"
        if part == "phase":
            return f"Phase {label} [deg]"
        if part == "abs":
            label = f"|{label}|"
        elif part in ("real", "imag"):
            label = f"{'Re' if part == 'real' else 'Im'}({label})"
        return f"{label} [{self.units}]" if self.units else label

    def _plot_overlay(
        self,
        ax: plt.Axes | None,
        fmin: float | None,
        fmax: float | None,
        part: str | None,
        xlabel: str | None,
        ylabel: str | None,
        legend: bool,
        plot_kwargs: dict[str, Any],
    ) -> plt.Axes:
        """Curve plot: one line per test."""
        if ax is None:
            _, ax = plt.subplots()
        is_complex = False
        for num, f, v in zip(self.numbers, self.freq, self.values):
            if v is None:
                continue
            is_complex = is_complex or np.iscomplexobj(v)
            mask = _band(f, fmin, fmax)
            ax.plot(
                f[mask],
                self._real_part(v[mask], part),
                label=f"Test {num}",
                **plot_kwargs,
            )
        ax.set_xlabel("Frequency [Hz]" if xlabel is None else xlabel)
        ax.set_ylabel(
            self._curve_ylabel(part, is_complex) if ylabel is None else ylabel
        )
        if legend and ax.lines:
            ax.legend()
        return ax

    def _plot_heatmap(
        self,
        ax: plt.Axes | None,
        fmin: float | None,
        fmax: float | None,
        part: str | None,
        xlabel: str | None,
        ylabel: str | None,
        colorbar: bool,
        plot_kwargs: dict[str, Any],
    ) -> plt.Axes:
        """Curve plot: test number along x, frequency along y (spectrogram-like)."""
        freq = self.common_freq()
        data = self.to_array()
        is_complex = np.iscomplexobj(data)
        data = np.asarray(self._real_part(data, part), dtype=float)
        mask = _band(freq, fmin, fmax)
        if not mask.any():
            raise ValueError(f"No frequencies between {fmin} and {fmax} Hz.")
        # Columns centred on the true test numbers, edges at the midpoints
        x_edges = _cell_edges(np.asarray(self.numbers, dtype=float))
        y_edges = _cell_edges(freq[mask])
        if ax is None:
            _, ax = plt.subplots()
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(data[:, mask].T),
            shading="flat",
            **plot_kwargs,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Test number" if xlabel is None else xlabel)
        ax.set_ylabel("Frequency [Hz]" if ylabel is None else ylabel)
        if colorbar:
            ax.figure.colorbar(mesh, ax=ax, label=self._curve_ylabel(part, is_complex))
        return ax


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #


def _check_curves(
    values: Any, freq: Any
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Validate per-test curves and frequency arrays."""
    values = list(values)
    freq = list(freq)
    if len(freq) != len(values):
        raise ValueError(
            f"freq has {len(freq)} entries but there are {len(values)} curves."
        )
    out_v: list[np.ndarray | None] = []
    out_f: list[np.ndarray | None] = []
    for i, (f, v) in enumerate(zip(freq, values)):
        if v is None or f is None:
            if (v is None) != (f is None):
                raise ValueError(
                    f"Curve {i}: values and freq must both be given or both None."
                )
            out_v.append(None)
            out_f.append(None)
            continue
        v = np.asarray(v)
        f = np.asarray(f, dtype=float)
        if v.ndim != 1 or f.shape != v.shape:
            raise ValueError(
                f"Curve {i}: values and freq must be 1D arrays of the same length, "
                f"got shapes {v.shape!r} and {f.shape!r}."
            )
        out_v.append(v)
        out_f.append(f)
    return out_v, out_f


def _combine_units(a: str | None, b: str | None, symbol: str) -> str | None:
    """Units of ``a * b`` or ``a / b`` as a compound string."""
    if a and b:
        return f"{a}{'·' if symbol == '*' else '/'}{b}"
    if symbol == "/" and b:
        return f"1/{b}"
    return a or b


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges at the midpoints between centres; outer cells mirror their neighbour."""
    if len(centres) == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    mid = (centres[:-1] + centres[1:]) / 2.0
    first = centres[0] - (mid[0] - centres[0])
    last = centres[-1] + (centres[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _band(f: np.ndarray, fmin: float | None, fmax: float | None) -> np.ndarray:
    """Boolean mask of ``fmin <= f <= fmax`` (either bound optional)."""
    mask = np.ones(f.shape, dtype=bool)
    if fmin is not None:
        mask &= f >= fmin
    if fmax is not None:
        mask &= f <= fmax
    return mask


def _df(f: np.ndarray) -> float:
    """Frequency step of a grid (NaN if fewer than two points)."""
    return float(f[1] - f[0]) if len(f) > 1 else float("nan")


def _cell(value: Any) -> str:
    """Format a number for CSV; NaN becomes an empty cell."""
    value = float(value)
    return "" if np.isnan(value) else repr(value)


def _to_float(value: Any) -> float:
    """Convert a meta value to float; missing or non-numeric values become NaN."""
    if value is None or isinstance(value, bool):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
