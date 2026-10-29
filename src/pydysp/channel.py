# channel.py
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, detrend, filtfilt, welch

from .arias import AriasResult
from .response import ResponseSpectrum, sdof_newmark_response
from .spectra import FourierSpectrum, WelchSpectrum


@dataclass(eq=False)
class Channel:
    """
    Single time-history channel with metadata and processing parameters.

    This class represents a single 1D time-history signal together with
    timing information, physical metadata (quantity, units), and a set of
    processing parameters (drift, filter, baseline, trim). Processing is
    non-destructive: methods update parameters and return new ``Channel``
    instances, while the raw data remain unchanged.

    Parameters
    ----------
    data : np.ndarray
        One-dimensional array of raw signal values.
    dt : float, optional
        Sampling interval in seconds. If not provided, an explicit ``time``
        array must be supplied.
    t0 : float, optional
        Start time in seconds. Used to build the time vector when ``dt`` is
        provided but ``time`` is not. Default is ``0.0``.
    time : np.ndarray, optional
        Explicit time array matching ``data`` in shape. If provided, it is
        used to infer ``dt`` and ``t0`` if they are not set.
    name_input : str, optional
        Original name as found in the input file or data source.
    name_user : str, optional
        User-defined short name (for example ``"Acc1"``).
    label_axis : str, optional
        Axis label for plotting, typically including units (for example
        ``"Acc1 [g]"``).
    label_legend : str, optional
        Legend label for plotting (for example ``"Acc1: Footing"``).
    description_long : str, optional
        Long human-readable description of the channel.
    quantity : str, optional
        Physical quantity type (for example ``"displacement"``, ``"force"``,
        ``"acceleration"``).
    units : str, optional
        Engineering units for the physical quantity (for example ``"m"``,
        ``"kN"``, ``"g"``).
    raw_units : str, optional
        Raw data acquisition units (for example ``"V"``).
    calibration_factor : float, optional
        Multiplicative factor used to convert raw data to physical units.
        Default is ``1.0`` (no scaling).
    drift_params : dict, optional
        Parameters controlling drift correction. By default an empty dict;
        when set, merged with ``DRIFT_DEFAULTS``.
    filter_params : dict, optional
        Parameters controlling Butterworth filtering. By default an empty
        dict; when set, merged with ``FILTER_DEFAULTS``.
    baseline_params : dict, optional
        Parameters controlling baseline correction via ``scipy.signal.detrend``.
    trim_params : dict, optional
        Parameters defining a time window for trimming (typically containing
        ``"t_start"`` and ``"t_end"`` in seconds).
    processing_notes : list of str, optional
        Free-form notes describing processing steps applied to the channel.
    tags : set of str, optional
        Tags used for grouping, such as ``{"q:acceleration"}``.
    meta : dict, optional
        Free-form metadata dictionary (for example sensor type, location).

    Notes
    -----
    Processed data are obtained through :meth:`processed` or :meth:`xy`,
    which apply drift correction, filtering, baseline correction, trimming,
    and calibration in a fixed order. Results can be cached for efficiency.
    """

    # Default processing parameters
    DRIFT_DEFAULTS: ClassVar[dict[str, Any]] = {"points": 50}
    FILTER_DEFAULTS: ClassVar[dict[str, Any]] = {
        "btype": "lowpass",
        "fc": 50.0,
        "order": 2,
    }
    BASELINE_DEFAULTS: ClassVar[dict[str, Any]] = {"type": "linear"}

    # Time-history data and time axis
    data: np.ndarray
    dt: float | None = None
    t0: float = 0.0
    time: np.ndarray | None = None

    # Naming, labels and physical metadata
    name_input: str | None = None
    name_user: str | None = None
    label_axis: str | None = None
    label_legend: str | None = None
    description_long: str | None = None
    quantity: str | None = None
    units: str | None = None
    raw_units: str | None = None
    calibration_factor: float = 1.0

    # Processing configuration and notes
    drift_params: dict[str, Any] = field(default_factory=dict)
    filter_params: dict[str, Any] = field(default_factory=dict)
    baseline_params: dict[str, Any] = field(default_factory=dict)
    trim_params: dict[str, Any] = field(default_factory=dict)
    processing_notes: list[str] = field(default_factory=list)

    # Grouping and free-form metadata
    tags: set[str] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)

    # Internal cache for processed data (not part of the public API)
    _cache_processed_time: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cache_processed_data: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cache_signature: tuple | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """
        Finalise initialisation after dataclass construction.

        This method normalises arrays, infers missing timing information, and
        sets sensible defaults for names, labels and tags. It is executed
        automatically by the dataclass after ``__init__``.

        Notes
        -----
        * ``data`` is converted to a 1D NumPy array and validated.
        * If ``time`` is provided, it is checked against ``data`` and used to
          infer ``dt`` and ``t0`` when missing.
        * If only ``dt`` is provided, an equally-spaced time vector is
          constructed starting at ``t0``.
        * Tags and labels are updated based on ``quantity``, ``units`` and
          names when available.
        """
        # Ensure data is a 1D NumPy array
        self.data = np.asarray(self.data)
        if self.data.ndim != 1:
            raise ValueError(
                f"Channel.data must be a 1D array (single time history), got shape {self.data.shape!r}"
            )

        # Handle time / dt relationship
        if self.time is not None:
            self.time = np.asarray(self.time)
            if self.time.shape != self.data.shape:
                raise ValueError(
                    f"Time and data must have the same shape; got time.shape={self.time.shape!r}, data.shape={self.data.shape!r}"
                )
            if self.dt is None and len(self.time) > 1:
                self.dt = float(self.time[1] - self.time[0])
            if self.t0 == 0.0 and len(self.time) > 0:
                self.t0 = float(self.time[0])
        else:
            if self.dt is None:
                raise ValueError("Either 'time' or 'dt' must be provided")
            if self.dt <= 0:
                raise ValueError(f"dt must be positive, got {self.dt!r}")
            n = self.data.shape[0]
            self.time = self.t0 + self.dt * np.arange(n)

        # Normalise and auto-add tags
        self.tags = set(self.tags)
        if self.quantity is not None:
            self.tags.add(f"q:{self.quantity}")

        # Sensible fallbacks for names and labels
        if self.name_user is None and self.name_input is not None:
            self.name_user = self.name_input
        if self.label_legend is None and self.name_user is not None:
            self.label_legend = self.name_user
        if self.label_axis is None and self.units is not None:
            base = self.label_legend or self.name_user or ""
            units = f"[{self.units}]"
            self.label_axis = f"{base} {units}".strip()

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def duration(self) -> float:
        """
        Total duration of the channel in seconds.

        Returns
        -------
        float
            Duration computed from the first and last entries of the time
            vector. Returns ``0.0`` for an empty channel.
        """
        if self.data.size == 0:
            return 0.0
        t, _ = self.processed()
        return float(t[-1] - t[0])

    # ------------------------------------------------------------------ #
    # Internal cache management
    # ------------------------------------------------------------------ #

    def _clear_cache(self) -> None:
        """
        Clear cached processed data.

        This method invalidates any cached processed time and data as well
        as the stored processing signature. It is called whenever processing
        parameters are updated on a new ``Channel`` instance.
        """
        self._cache_processed_time = None
        self._cache_processed_data = None
        self._cache_signature = None

    # ------------------------------------------------------------------ #
    # Info
    # ------------------------------------------------------------------ #

    def info(self) -> str:
        """
        Return a human-readable summary of channel metadata and processing.

        The summary includes basic signal characteristics, timing,
        physical meaning and calibration, naming and labels, tags,
        processing parameters and notes, and any additional metadata.

        Returns
        -------
        str
            Multi-line summary string suitable for printing.
        """
        lines = []
        # Header
        title = self.name_user or self.name_input or "<unnamed>"
        lines.append(f"Channel: {title}")
        lines.append("-" * (len(title) + 9))
        # Basic signal info
        lines.append(f"Length: {len(self.data)} samples")
        if self.dt is not None and self.dt > 0:
            lines.append(f"Sampling frequency: fs = {1.0 / self.dt:g} Hz")
            lines.append(f"Timestep: dt = {self.dt:g} s")
        else:
            lines.append("Sampling frequency: fs = (unknown)")
            lines.append("Timestep: dt = (unknown)")
        lines.append(f"Start time: t0 = {self.t0} s")
        # Physical meaning & calibration
        if any(
            [self.quantity, self.units, self.raw_units, self.calibration_factor != 1.0]
        ):
            lines.append("\nPhysical meaning & calibration:")
            if self.quantity:
                lines.append(f"  Quantity: {self.quantity}")
            if self.units:
                lines.append(f"  Units: {self.units}")
            if self.raw_units:
                lines.append(f"  Raw units: {self.raw_units}")
            if self.calibration_factor != 1.0:
                lines.append(
                    f"  Calibration factor: {self.calibration_factor}"
                    + f" {self.units or 'units'}/{self.raw_units or 'raw'}"
                )
        # Naming / labels
        if any(
            [
                self.name_input,
                self.name_user,
                self.label_legend,
                self.label_axis,
                self.description_long,
            ]
        ):
            lines.append("\nNaming / labels:")
            if self.name_input:
                lines.append(f"  Input name: {self.name_input}")
            if self.name_user:
                lines.append(f"  User name: {self.name_user}")
            if self.label_legend:
                lines.append(f"  Legend label: {self.label_legend}")
            if self.label_axis:
                lines.append(f"  Axis label: {self.label_axis}")
            if self.description_long:
                lines.append(f"  Description: {self.description_long}")
        # Tags
        if self.tags:
            lines.append("\nTags:")
            taglist = "\n  ".join(sorted(self.tags))
            lines.append(f"  {taglist}")
        # Processing
        if (
            self.drift_params
            or self.filter_params
            or self.baseline_params
            or self.trim_params
            or self.processing_notes
        ):
            lines.append("\nProcessing:")
            if self.drift_params:
                lines.append(f"  Drift params: {self.drift_params}")
            if self.filter_params:
                lines.append(f"  Filter params: {self.filter_params}")
            if self.baseline_params:
                lines.append(f"  Baseline params: {self.baseline_params}")
            if self.trim_params:
                lines.append(f"  Trim params: {self.trim_params}")
            if self.processing_notes:
                lines.append("  Notes:")
                for note in self.processing_notes:
                    lines.append(f"    - {note}")
        # Free-form metadata
        if self.meta:
            lines.append("\nMetadata:")
            for k, v in self.meta.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Processing methods that create new Channels
    # ------------------------------------------------------------------ #

    def drift_corrected(self, **override: Any) -> "Channel":
        """
        Return a new channel with updated drift correction parameters.

        Drift correction is applied lazily in :meth:`processed`. This method
        only updates the stored parameters, clears the processing cache and
        appends a processing note.

        Parameters
        ----------
        **override
            Keyword arguments that override or extend the current drift
            parameters. Merged with ``DRIFT_DEFAULTS`` and ``drift_params``.

        Returns
        -------
        Channel
            New channel instance with updated drift parameters and tags.
        """
        params = {**self.DRIFT_DEFAULTS, **self.drift_params, **override}
        new = replace(self, drift_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"drift_corrected"})
        new.processing_notes = [
            *self.processing_notes,
            f"Drift params set: {params}",
        ]
        return new

    def filtered(self, **override: Any) -> "Channel":
        """
        Return a new channel with updated Butterworth filter parameters.

        Filtering is applied lazily in :meth:`processed`. This method records
        the requested filter specification, clears the processing cache and
        appends a processing note.

        Parameters
        ----------
        **override
            Keyword arguments that override or extend the current filter
            parameters. Merged with ``FILTER_DEFAULTS`` and ``filter_params``.

        Returns
        -------
        Channel
            New channel instance with updated filter parameters and tags.
        """
        params = {**self.FILTER_DEFAULTS, **self.filter_params, **override}
        new = replace(self, filter_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"filtered"})
        new.processing_notes = [
            *self.processing_notes,
            f"Filter params set: {params}",
        ]
        return new

    def baseline_corrected(self, **override: Any) -> "Channel":
        """
        Return a new channel with updated baseline correction parameters.

        Baseline correction is applied lazily in :meth:`processed` using
        :func:`scipy.signal.detrend`. This method records the detrend
        parameters, clears the processing cache and appends a processing note.

        Parameters
        ----------
        **override
            Keyword arguments forwarded to :func:`scipy.signal.detrend`
            when baseline correction is applied.

        Returns
        -------
        Channel
            New channel instance with updated baseline parameters and tags.
        """
        params = {**self.BASELINE_DEFAULTS, **self.baseline_params, **override}
        new = replace(self, baseline_params=params)
        new._clear_cache()
        arg_str = ", ".join(f"{k}={v}" for k, v in params.items()) or ""
        new.tags = set(self.tags).union({"baseline_corrected"})
        new.processing_notes = [
            *self.processing_notes,
            f"Baseline params set: detrend({arg_str})",
        ]
        return new

    def trimmed(self, **override: Any) -> "Channel":
        """
        Return a new channel with updated trimming window parameters.

        Trimming is applied in :meth:`processed` by restricting the time and
        data arrays to a specified interval.

        Parameters
        ----------
        **override
            Keyword arguments that override or extend the current trimming
            parameters. Typical keys are ``"t_start"`` and ``"t_end"`` in
            seconds.

        Returns
        -------
        Channel
            New channel instance with updated trimming parameters and tags.
        """
        defaults = {
            "t_start": float(self.time[0]),
            "t_end": float(self.time[-1]),
        }
        params = {**defaults, **self.trim_params, **override}
        t_start = params["t_start"]
        t_end = params["t_end"]
        new = replace(self, trim_params=params)
        new._clear_cache()
        new.tags = set(self.tags).union({"trimmed"})
        new.processing_notes = [
            *self.processing_notes,
            f"Trim params set: {t_start}–{t_end} s",
        ]
        return new

    # ------------------------------------------------------------------ #
    # Trimmers
    # ------------------------------------------------------------------ #

    def trim_by_threshold(
        self,
        threshold: float = 0.01,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Channel":
        """
        Trim the channel to where the signal exceeds a given threshold.

        The trimming window is defined from the first to the last sample where
        the signal magnitude exceeds ``threshold``, optionally in absolute
        value, with configurable buffers before and after this window.

        Parameters
        ----------
        threshold : float, optional
            Amplitude threshold used to detect the start and end of the
            significant portion of the record. Default is ``0.01``.
        use_abs : bool, optional
            If ``True`` (default), the threshold is applied to the absolute
            value of the signal. If ``False``, it is applied to the raw
            signal.
        buffer_before : float, optional
            Time (seconds) to extend the trimming window before the first
            threshold-crossing sample. Default is ``0.0``.
        buffer_after : float, optional
            Time (seconds) to extend the trimming window after the last
            threshold-crossing sample. Default is ``0.0``.
        processed : bool, optional
            If ``True`` (default), trimming is based on the processed signal
            returned by :meth:`xy`. If ``False``, the raw data are used.
        use_cache : bool, optional
            If ``True`` (default), use the processing cache when obtaining
            the signal.

        Returns
        -------
        Channel
            New channel instance with updated trimming parameters.

        Raises
        ------
        ValueError
            If the channel is empty, no samples exceed the threshold, or
            the resulting window is empty after buffering.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot trim empty signal: channel has zero samples")
        if use_abs:
            mask = np.abs(y) >= threshold
        else:
            mask = y >= threshold
        if not np.any(mask):
            raise ValueError(f"No samples exceed the specified threshold ({threshold})")

        # Indices of first and last samples above the threshold
        i_start = int(np.argmax(mask))
        i_end = int(len(mask) - 1 - np.argmax(mask[::-1]))

        t_start = float(t[i_start]) - buffer_before
        t_end = float(t[i_end]) + buffer_after
        # Clamp to original time range
        t_start = max(t_start, float(t[0]))
        t_end = min(t_end, float(t[-1]))
        if t_end <= t_start:
            raise ValueError(
                f"Computed trim window is empty after buffering: t_start={t_start}, t_end={t_end}"
            )
        return self.trimmed(t_start=t_start, t_end=t_end)

    def trim_by_fraction_of_peak(
        self,
        fraction: float = 0.05,
        *,
        use_abs: bool = True,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Channel":
        """
        Trim the channel to where the signal exceeds a fraction of its peak.

        The trimming window is defined using a threshold equal to
        ``fraction`` times the signal peak amplitude, optionally in absolute
        value, with configurable buffers before and after.

        Parameters
        ----------
        fraction : float, optional
            Fraction of the peak amplitude used to define the threshold.
            Must be in the range ``(0, 1]``. Default is ``0.05`` (5% of peak).
        use_abs : bool, optional
            If ``True`` (default), the peak and threshold are computed using
            the absolute value of the signal. If ``False``, they are computed
            from the raw signal.
        buffer_before : float, optional
            Time (seconds) to extend the trimming window before the first
            threshold-crossing sample. Default is ``0.0``.
        buffer_after : float, optional
            Time (seconds) to extend the trimming window after the last
            threshold-crossing sample. Default is ``0.0``.
        processed : bool, optional
            If ``True`` (default), use the processed signal; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default), use the processing cache when obtaining
            the signal.

        Returns
        -------
        Channel
            New channel instance with updated trimming parameters.

        Raises
        ------
        ValueError
            If ``fraction`` is out of range, the channel is empty, the peak
            amplitude is non-positive, or the resulting window is empty.
        """
        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        _, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot trim empty signal: channel has zero samples")
        if use_abs:
            peak = float(np.max(np.abs(y)))
        else:
            peak = float(np.max(y))
        if peak <= 0.0:
            raise ValueError(
                f"Signal peak is non-positive ({peak}); cannot define threshold from peak"
            )
        threshold = fraction * peak
        return self.trim_by_threshold(
            threshold=threshold,
            use_abs=use_abs,
            buffer_before=buffer_before,
            buffer_after=buffer_after,
            processed=processed,
            use_cache=use_cache,
        )

    def trim_by_arias(
        self,
        lower: float = 0.05,
        upper: float = 0.95,
        g: float = 9.81,
        buffer_before: float = 0.0,
        buffer_after: float = 0.0,
        processed: bool = True,
        use_cache: bool = True,
    ) -> "Channel":
        """
        Trim the channel using an Arias-intensity-based significant duration.

        The trimming window is defined between the times when the cumulative
        Arias intensity reaches fractions ``lower`` and ``upper`` of its final
        value, optionally extended with time buffers.

        Parameters
        ----------
        lower : float, optional
            Lower fraction of final Arias intensity (for example ``0.05``
            for 5%). Must satisfy ``0 <= lower < upper <= 1``. Default is
            ``0.05``.
        upper : float, optional
            Upper fraction of final Arias intensity (for example ``0.95``
            for 95%). Default is ``0.95``.
        g : float, optional
            Gravity acceleration used to convert acceleration in g to m/s².
            Default is ``9.81``.
        buffer_before : float, optional
            Time (seconds) to extend the trimming window before the lower
            Arias intensity crossing. Default is ``0.0``.
        buffer_after : float, optional
            Time (seconds) to extend the trimming window after the upper
            Arias intensity crossing. Default is ``0.0``.
        processed : bool, optional
            If ``True`` (default), use the processed acceleration signal
            when computing Arias intensity.
        use_cache : bool, optional
            If ``True`` (default), use the processing cache when obtaining
            the signal.

        Returns
        -------
        Channel
            New channel instance with updated trimming parameters.

        Raises
        ------
        ValueError
            If the channel is empty, the final Arias intensity is non-positive,
            the ``lower``/``upper`` values are invalid, or the resulting
            window is empty after buffering.

        Notes
        -----
        The channel data are assumed to represent acceleration in units of g.
        """
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError(
                f"Require 0 <= lower < upper <= 1, got lower={lower!r}, upper={upper!r}"
            )
        res = self.arias_intensity(g=g, processed=processed, use_cache=use_cache)
        t = res.t
        Ia = res.Ia
        if Ia.size == 0:
            raise ValueError("Cannot trim by Arias intensity for empty signal")
        Ia_final = float(Ia[-1])
        if Ia_final <= 0.0:
            raise ValueError(f"Final Arias intensity is non-positive ({Ia_final})")

        # Indices where cumulative Arias intensity crosses the requested fractions
        idx_lower = int(np.argmax(Ia >= lower * Ia_final))
        idx_upper = int(np.argmax(Ia >= upper * Ia_final))

        t_lower = float(t[idx_lower]) - buffer_before
        t_upper = float(t[idx_upper]) + buffer_after
        # Clamp to original time range
        t_lower = max(t_lower, float(t[0]))
        t_upper = min(t_upper, float(t[-1]))
        if t_upper <= t_lower:
            raise ValueError(
                f"Computed Arias trim window is empty after buffering: t_start={t_lower}, t_end={t_upper}"
            )
        return self.trimmed(t_start=t_lower, t_end=t_upper)

    # ------------------------------------------------------------------ #
    # Processed
    # ------------------------------------------------------------------ #

    def processed(self, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Return time and data after applying all processing steps.

        The following operations are applied in a fixed order:

        1. Drift correction (if ``drift_params`` are set).
        2. Butterworth filtering (if ``filter_params`` are set).
        3. Baseline detrend (if ``baseline_params`` are set).
        4. Trimming by time window (if ``trim_params`` are set).
        5. Calibration-factor scaling.

        Parameters
        ----------
        use_cache : bool, optional
            If ``True`` (default), cached processed results are returned
            when available and the processing signature has not changed.

        Returns
        -------
        t : np.ndarray
            Time array after trimming (if applied).
        y : np.ndarray
            Processed data array after all enabled steps.

        Raises
        ------
        ValueError
            If processing parameters are inconsistent (for example invalid
            filter cutoff, trimming window outside the time range, or missing
            ``dt`` for filtering or response-related calculations).
        """
        # Signature summarising the current processing configuration for caching
        signature = (
            tuple(sorted(self.drift_params.items())),
            tuple(sorted(self.filter_params.items())),
            tuple(sorted(self.baseline_params.items())),
            tuple(sorted(self.trim_params.items())),
            self.calibration_factor,
        )
        if (
            use_cache
            and self._cache_signature == signature
            and self._cache_processed_data is not None
            and self._cache_processed_time is not None
        ):
            return self._cache_processed_time, self._cache_processed_data

        # Start from raw data
        t = self.time
        y = self.data.astype(float, copy=False)

        # 1. Drift correction
        if self.drift_params:
            params = {**self.DRIFT_DEFAULTS, **self.drift_params}
            points = params["points"]
            if points > len(y):
                raise ValueError(
                    f"Number of points for drift correction ({points}) exceeds data length ({len(y)})"
                )
            drift_value = float(np.mean(y[:points]))
            y = y - drift_value

        # 2. Filtering
        if self.filter_params:
            if self.dt is None or self.dt <= 0:
                raise ValueError(
                    f"Filtering requires a positive dt, got dt={self.dt!r}"
                )
            fs = 1.0 / self.dt
            params = {**self.FILTER_DEFAULTS, **self.filter_params}
            btype = params["btype"]
            order = params["order"]
            if btype in ("lowpass", "highpass"):
                fc = params["fc"]
                Wn = 2 * fc / fs
            elif btype in ("bandpass", "bandstop"):
                f1 = params.get("f1")
                f2 = params.get("f2")
                if f1 is None or f2 is None:
                    raise ValueError(
                        f"Band filters require f1 and f2, got f1={f1!r}, f2={f2!r}"
                    )
                Wn = [2 * f1 / fs, 2 * f2 / fs]
            else:
                raise ValueError(f"Unsupported filter mode: {btype!r}")
            if isinstance(Wn, (list, tuple)):
                if not (0 < Wn[0] < Wn[1] < 1):
                    raise ValueError(
                        f"Normalized band edges must satisfy 0 < f1 < f2 < fs/2; got Wn={Wn} with fs={fs}"
                    )
            else:
                if not (0 < Wn < 1):
                    raise ValueError(
                        f"Normalized cutoff must satisfy 0 < fc < fs/2; got Wn={Wn} with fs={fs}"
                    )
            b, a = butter(order, Wn, btype=btype)
            y = filtfilt(b, a, y)

        # 3. Baseline correction
        if self.baseline_params:
            params = {**self.BASELINE_DEFAULTS, **self.baseline_params}
            y = detrend(y, **params)

        # 4. Trimming
        if self.trim_params:
            defaults = {"t_start": float(t[0]), "t_end": float(t[-1])}
            params = {**defaults, **self.trim_params}
            t_start = params["t_start"]
            t_end = params["t_end"]
            if t_end <= t_start:
                raise ValueError(
                    f"t_end must be greater than t_start: t_start={t_start}, t_end={t_end}"
                )
            mask = (t >= t_start) & (t <= t_end)
            if not np.any(mask):
                raise ValueError(
                    f"Trim window [{t_start}, {t_end}] does not overlap channel time range [{float(t[0])}, {float(t[-1])}]"
                )
            t = t[mask]
            y = y[mask]

        # 5. Calibration
        if self.calibration_factor != 1.0:
            y = y * self.calibration_factor

        # Cache result
        if use_cache:
            self._cache_signature = signature
            self._cache_processed_time = t
            self._cache_processed_data = y
        return t, y

    def xy(
        self, processed: bool = True, use_cache: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return coordinate arrays for plotting.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), return the processed time and data arrays
            from :meth:`processed`. If ``False``, return the raw ``time`` and
            ``data`` attributes.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache when computing processed data.

        Returns
        -------
        x : np.ndarray
            Time array.
        y : np.ndarray
            Data array (raw or processed depending on ``processed``).
        """
        if processed:
            return self.processed(use_cache=use_cache)
        else:
            return self.time, self.data

    # ------------------------------------------------------------------ #
    # Simple time-domain peaks and RMS
    # ------------------------------------------------------------------ #

    def max_abs(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the time and value of the maximum absolute amplitude.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        t_peak : float
            Time at which the maximum absolute amplitude occurs.
        y_peak : float
            Data value at the maximum absolute amplitude.

        Raises
        ------
        ValueError
            If the channel is empty.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute max_abs of empty signal")
        idx = int(np.argmax(np.abs(y)))
        return float(t[idx]), float(y[idx])

    def max_value(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the time and value of the maximum (positive) data value.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        t_peak : float
            Time at which the maximum value occurs.
        y_peak : float
            Maximum data value.

        Raises
        ------
        ValueError
            If the channel is empty.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute max_value of empty signal")
        idx = int(np.argmax(y))
        return float(t[idx]), float(y[idx])

    def min_value(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the time and value of the minimum (negative) data value.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        t_peak : float
            Time at which the minimum value occurs.
        y_peak : float
            Minimum data value.

        Raises
        ------
        ValueError
            If the channel is empty.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute min_value of empty signal")
        idx = int(np.argmin(y))
        return float(t[idx]), float(y[idx])

    def rms(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> float:
        """
        Compute the root-mean-square (RMS) value of the channel.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), compute RMS of the processed data;
            otherwise use the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        float
            RMS value of the signal.

        Raises
        ------
        ValueError
            If the channel is empty.
        """
        _, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute RMS of empty signal")
        return float(np.sqrt(np.mean(y**2)))

    # ------------------------------------------------------------------ #
    # Fourier amplitude spectrum + peak
    # ------------------------------------------------------------------ #

    def fourier(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> FourierSpectrum:
        """
        Compute the (single-sided) Fourier amplitude spectrum of the channel.

        The spectrum is computed via zero-padded FFT to the next power-of-two
        length, and the single-sided amplitude spectrum is returned.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        FourierSpectrum
            Object containing frequency array and amplitude spectrum.

        Raises
        ------
        ValueError
            If the channel is empty or ``dt`` is missing/invalid.
        """
        _, y = self.xy(processed=processed, use_cache=use_cache)
        n = len(y)
        if n == 0:
            raise ValueError("Cannot compute Fourier spectrum of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Fourier transform requires a positive dt")
        n_fft = 1 << (n - 1).bit_length()
        f = np.fft.rfftfreq(n=n_fft, d=self.dt)
        s = np.abs(np.fft.rfft(y, n=n_fft))
        return FourierSpectrum(f=f, s=s)

    def fourier_peak(
        self,
        processed: bool = True,
        use_cache: bool = True,
    ) -> tuple[float, float]:
        """
        Return the dominant frequency and amplitude from the Fourier spectrum.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        f_peak : float
            Frequency at which the spectrum amplitude is maximum.
        s_peak : float
            Maximum amplitude of the Fourier spectrum.
        """
        spec = self.fourier(processed=processed, use_cache=use_cache)
        return spec.peak()

    # ------------------------------------------------------------------ #
    # Welch PSD + peak (MATLAB-ish defaults)
    # ------------------------------------------------------------------ #

    def welch_psd(
        self,
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> WelchSpectrum:
        """
        Compute the power spectral density (PSD) using Welch's method.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        **kwargs
            Additional keyword arguments forwarded to
            :func:`scipy.signal.welch`. If ``nperseg`` is not provided, a
            default of ``min(256, n)`` is used.

        Returns
        -------
        WelchSpectrum
            Object containing frequency array and PSD values.

        Raises
        ------
        ValueError
            If the channel is empty or ``dt`` is missing/invalid.
        """
        _, y = self.xy(processed=processed, use_cache=use_cache)
        n = len(y)
        if n == 0:
            raise ValueError("Cannot compute Welch PSD of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Welch PSD requires a positive dt")
        fs = 1.0 / self.dt
        if "nperseg" not in kwargs:
            kwargs["nperseg"] = min(256, n)
        f, p = welch(x=y, fs=fs, **kwargs)
        return WelchSpectrum(f=f, p=p)

    def welch_peak(
        self,
        processed: bool = True,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> tuple[float, float]:
        """
        Return the dominant frequency and PSD amplitude from the Welch spectrum.

        Parameters
        ----------
        processed : bool, optional
            If ``True`` (default), use the processed data; otherwise use
            the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        **kwargs
            Additional keyword arguments forwarded to
            :meth:`welch_psd` and then to :func:`scipy.signal.welch`.

        Returns
        -------
        f_peak : float
            Frequency at which the PSD is maximum.
        p_peak : float
            Maximum PSD value.
        """
        psd = self.welch_psd(processed=processed, use_cache=use_cache, **kwargs)
        return psd.peak()

    # ------------------------------------------------------------------ #
    # Arias intensity
    # ------------------------------------------------------------------ #

    def arias_intensity(
        self,
        g: float = 9.81,
        processed: bool = True,
        use_cache: bool = True,
    ) -> AriasResult:
        """
        Compute cumulative Arias intensity and significant-duration window.

        The channel is assumed to contain acceleration in units of g. The
        resulting Arias intensity is computed via numerical integration of
        squared acceleration, and the 5–95% significant-duration window is
        identified.

        Parameters
        ----------
        g : float, optional
            Gravity acceleration used to convert from g to m/s². Default is
            ``9.81``.
        processed : bool, optional
            If ``True`` (default), use the processed acceleration signal;
            otherwise use the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        AriasResult
            Object containing time array, cumulative Arias intensity, and
            start/end times of the 5–95% significant-duration window.

        Raises
        ------
        ValueError
            If the channel is empty or the final Arias intensity is
            non-positive.
        """
        t, y = self.xy(processed=processed, use_cache=use_cache)
        if y.size == 0:
            raise ValueError("Cannot compute Arias intensity of empty signal")
        a_mps2 = g * y
        coef = np.pi / (2.0 * g)
        Ia = cumulative_trapezoid(coef * a_mps2**2, t, initial=0.0)
        Ia_final = float(Ia[-1])
        if Ia_final <= 0:
            raise ValueError("Final Arias intensity is non-positive")
        idx_start = int(np.argmax(Ia >= 0.05 * Ia_final))
        idx_end = int(np.argmax(Ia >= 0.95 * Ia_final))
        t_start = float(t[idx_start])
        t_end = float(t[idx_end])
        return AriasResult(
            t=t,
            Ia=Ia,
            t_start=t_start,
            t_end=t_end,
        )

    # ------------------------------------------------------------------ #
    # Response spectrum (SDOF, Newmark-beta average acceleration)
    # ------------------------------------------------------------------ #

    def response_spectrum(
        self,
        # Public default; changing it would alter the signature.
        periods: np.ndarray = np.linspace(0.05, 5.0, 100),  # noqa: B008
        g: float = 9.81,
        ksi: float = 0.05,
        processed: bool = True,
        use_cache: bool = True,
    ) -> ResponseSpectrum:
        """
        Compute an elastic response spectrum for a grid of periods.

        The response spectrum is computed by subjecting a set of linear
        single-degree-of-freedom (SDOF) oscillators, with specified damping
        ratio, to the acceleration record using the Newmark-beta average
        acceleration method.

        Parameters
        ----------
        periods : np.ndarray, optional
            One-dimensional array of natural periods (in seconds) at which
            the spectrum is evaluated. Default is ``np.linspace(0.05, 5.0, 100)``.
        g : float, optional
            Gravity acceleration used to convert from g to m/s². Default is
            ``9.81``.
        ksi : float, optional
            Damping ratio of the SDOF systems (for example ``0.05`` for 5%
            damping). Default is ``0.05``.
        processed : bool, optional
            If ``True`` (default), use the processed acceleration signal;
            otherwise use the raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.

        Returns
        -------
        ResponseSpectrum
            Object containing periods, spectral displacement (``Sd``),
            velocity (``Sv``) and acceleration (``Sa`` in units of g), and
            the damping ratio.

        Raises
        ------
        ValueError
            If the channel is empty, ``dt`` is missing/invalid, ``periods``
            is not 1D, or contains non-positive values.
        """
        _, a = self.xy(processed=processed, use_cache=use_cache)
        if a.size == 0:
            raise ValueError("Cannot compute response spectrum of empty signal")
        if self.dt is None or self.dt <= 0:
            raise ValueError("Response spectrum requires a positive dt")
        a_mps2 = g * a  # convert g to m/s^2
        # Copy, so the result never shares the (default) periods array
        periods = np.array(periods, dtype=float)
        if periods.ndim != 1:
            raise ValueError("periods must be a 1D array")
        if np.any(periods <= 0.0):
            raise ValueError("All periods must be positive")

        # Preallocate spectral arrays and loop over SDOF periods
        Sd = np.zeros_like(periods, dtype=float)
        Sv = np.zeros_like(periods, dtype=float)
        Sa = np.zeros_like(periods, dtype=float)
        for i, T in enumerate(periods):
            omega = 2.0 * np.pi / T
            Sd[i], Sv[i], Sa[i] = sdof_newmark_response(a_mps2, self.dt, omega, ksi)
        Sa_g = Sa / g  # convert back to g
        return ResponseSpectrum(T=periods, Sd=Sd, Sv=Sv, Sa=Sa_g, ksi=ksi)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    def plot(
        self,
        ax: plt.Axes | None = None,
        processed: bool = True,
        use_cache: bool = True,
        include_label: bool = True,
        include_kind: bool = False,
        include_legend: bool = False,
        plot_type: str = "timehistory",
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the channel in one of several supported representations.

        The default behaviour (``plot_type='timehistory'``) is to plot the
        time history. Other values of ``plot_type`` delegate to helper
        methods to plot Fourier spectrum, power spectral density, Arias
        intensity or response spectrum.

        Supported ``plot_type`` values
        -------------------------------
        * ``"timehistory"``, ``"time"``, ``"time_history"``
        * ``"fourier"``, ``"fft"``
        * ``"psd"``, ``"welch"``, ``"power"``
        * ``"arias"``, ``"husid"``
        * ``"response"``, ``"response_spectrum"``, ``"rs"``

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created
            with :func:`matplotlib.pyplot.subplots`.
        processed : bool, optional
            If ``True`` (default), use processed data for time-domain
            plotting and for spectral quantities. If ``False``, use raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        include_label : bool, optional
            For time-history plots, if ``True`` (default) use ``label_axis``
            as the y-axis label when available.
        include_kind : bool, optional
            For time-history plots, if ``True``, build a generic y-label from
            ``quantity`` and ``units`` (e.g. ``"Acceleration [g]"``). This
            overrides any existing y-label if set.
        include_legend : bool, optional
            For time-history plots, if ``True``, add a legend using
            ``label_legend`` or ``name_user``.
        plot_type : str, optional
            Representation to plot. See list above. Default is
            ``"timehistory"``.
        **plot_kwargs
            Additional keyword arguments forwarded to the underlying plotting
            method (for example line style, color, ``fmax``, ``periods``).

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance containing the requested plot.

        Raises
        ------
        ValueError
            If an unknown ``plot_type`` is given.
        """
        if ax is None:
            _, ax = plt.subplots()

        pt = plot_type.lower()

        # Time history (original behaviour)
        if pt in ("time", "timehistory", "time_history"):
            t, y = self.xy(processed=processed, use_cache=use_cache)
            line_label = self.label_legend or self.name_user or self.name_input
            ax.plot(t, y, label=line_label, **plot_kwargs)
            ax.set_xlabel("Time [s]")

            ylabel = ax.get_ylabel() or ""
            if include_label and self.label_axis:
                ylabel = self.label_axis
            if include_kind and self.quantity:
                ylabel = self.quantity.capitalize() + (
                    f" [{self.units}]" if self.units else ""
                )
            ax.set_ylabel(ylabel)

            if include_legend and line_label:
                ax.legend()

            ax.grid(True)
            return ax

        # Delegated spectral / intensity / response plots
        if pt in ("fourier", "fft"):
            return self.plot_fourier(
                ax=ax,
                processed=processed,
                use_cache=use_cache,
                **plot_kwargs,
            )
        if pt in ("psd", "welch", "power"):
            return self.plot_psd(
                ax=ax,
                processed=processed,
                use_cache=use_cache,
                **plot_kwargs,
            )
        if pt in ("arias", "husid"):
            return self.plot_arias(
                ax=ax,
                processed=processed,
                use_cache=use_cache,
                **plot_kwargs,
            )
        if pt in ("response", "response_spectrum", "rs"):
            return self.plot_response_spectrum(
                ax=ax,
                processed=processed,
                use_cache=use_cache,
                **plot_kwargs,
            )

        raise ValueError(
            f"Unknown plot_type {plot_type!r}. "
            "Use 'timehistory', 'fourier', 'psd', 'arias', or 'response'."
        )

    def plot_fourier(
        self,
        ax: plt.Axes | None = None,
        processed: bool = True,
        use_cache: bool = True,
        fmax: float | None = 50.0,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Fourier amplitude spectrum of the channel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        processed : bool, optional
            If ``True`` (default), compute the spectrum from processed data;
            otherwise use raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        fmax : float or None, optional
            Maximum frequency shown on the plot (in Hz). If ``None``, the
            full frequency range is used. Default is ``50.0``.
        **plot_kwargs
            Additional keyword arguments forwarded to
            :meth:`FourierSpectrum.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance containing the Fourier spectrum plot.
        """
        spec = self.fourier(processed=processed, use_cache=use_cache)
        return spec.plot(ax=ax, fmax=fmax, **plot_kwargs)

    def plot_psd(
        self,
        ax: plt.Axes | None = None,
        processed: bool = True,
        use_cache: bool = True,
        fmax: float | None = 50.0,
        **welch_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the power spectral density (PSD) using Welch's method.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        processed : bool, optional
            If ``True`` (default), compute the PSD from processed data;
            otherwise use raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        fmax : float or None, optional
            Maximum frequency shown on the plot (in Hz). If ``None``, the
            full frequency range is used. Default is ``50.0``.
        **welch_kwargs
            Additional keyword arguments forwarded to :meth:`welch_psd` and
            then to :func:`scipy.signal.welch`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance containing the PSD plot.
        """
        spec = self.welch_psd(processed=processed, use_cache=use_cache, **welch_kwargs)
        return spec.plot(ax=ax, fmax=fmax)

    def plot_arias(
        self,
        ax: plt.Axes | None = None,
        g: float = 9.81,
        processed: bool = True,
        use_cache: bool = True,
        show_window: bool = True,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Arias intensity time history (Husid plot) for this channel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        g : float, optional
            Gravity acceleration used to convert acceleration from g to m/s².
            Default is ``9.81``.
        processed : bool, optional
            If ``True`` (default), compute Arias intensity from processed
            acceleration; otherwise use raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        show_window : bool, optional
            If ``True`` (default), plot vertical lines marking the 5–95%
            significant-duration window.
        **plot_kwargs
            Additional keyword arguments forwarded to
            :meth:`AriasResult.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance containing the Arias intensity plot.

        Notes
        -----
        The channel data are assumed to represent acceleration in units of g.
        """
        res = self.arias_intensity(g=g, processed=processed, use_cache=use_cache)
        return res.plot(ax=ax, show_window=show_window, **plot_kwargs)

    def plot_response_spectrum(
        self,
        ax: plt.Axes | None = None,
        # Public default; changing it would alter the signature.
        periods: np.ndarray = np.linspace(0.05, 5.0, 100),  # noqa: B008
        ksi: float = 0.05,
        processed: bool = True,
        use_cache: bool = True,
        y: str = "Sa",
        logx: bool = False,
        logy: bool = False,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the elastic response spectrum for this channel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        periods : np.ndarray, optional
            One-dimensional array of natural periods (in seconds) at which
            the spectrum is evaluated. Default is
            ``np.linspace(0.05, 5.0, 100)``.
        ksi : float, optional
            Damping ratio of the SDOF systems (for example ``0.05`` for 5%
            damping). Default is ``0.05``.
        processed : bool, optional
            If ``True`` (default), compute the spectrum from processed data;
            otherwise use raw data.
        use_cache : bool, optional
            If ``True`` (default) and ``processed`` is ``True``, use the
            processing cache.
        y : {"Sa", "Sv", "Sd"}, optional
            Response quantity to plot: spectral acceleration ``"Sa"``, velocity
            ``"Sv"`` or displacement ``"Sd"``. Default is ``"Sa"``.
        logx : bool, optional
            If ``True``, use a logarithmic scale on the x-axis (period).
            Default is ``False``.
        logy : bool, optional
            If ``True``, use a logarithmic scale on the y-axis (response).
            Default is ``False``.
        **plot_kwargs
            Additional keyword arguments forwarded to
            :meth:`ResponseSpectrum.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance containing the response spectrum plot.
        """
        rs = self.response_spectrum(
            periods=periods,
            ksi=ksi,
            processed=processed,
            use_cache=use_cache,
        )
        return rs.plot(
            ax=ax,
            y=y,
            logx=logx,
            logy=logy,
            **plot_kwargs,
        )
