# helpers.py
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import matplotlib.pyplot as plt

from .channel import Channel
from .response import ResponseSpectrum

PlotType = Literal["time", "timehistory", "fourier", "psd", "welch"]


def annotate_peak(
    channel: Channel,
    ax: plt.Axes,
    plot: PlotType = "time",
    *,
    peak: Literal["abs", "max", "min"] = "abs",
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    welch_kwargs: Mapping[str, Any] | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate a peak on a plot of a :class:`Channel`.

    This helper assumes the relevant quantity has already been plotted on
    ``ax`` using the corresponding ``Channel.plot*`` method (for example
    :meth:`Channel.plot`, :meth:`Channel.plot_fourier`,
    :meth:`Channel.plot_psd`). It then identifies the appropriate peak
    (time-domain, Fourier or PSD) and adds a marker and text annotation.

    Parameters
    ----------
    channel : Channel
        Channel whose peak will be annotated.
    ax : matplotlib.axes.Axes
        Matplotlib axes on which the data has already been plotted.
    plot : {"time", "timehistory", "fourier", "psd", "welch"}, optional
        Kind of plot that is currently shown on ``ax``:

        * ``"time"`` / ``"timehistory"`` : time-history peak in the time domain.
        * ``"fourier"``                  : Fourier amplitude spectrum peak.
        * ``"psd"`` / ``"welch"``        : Welch PSD peak.

        Default is ``"time"``.
    peak : {"abs", "max", "min"}, optional
        Which time-domain peak to use when ``plot`` is ``"time"`` or
        ``"timehistory"``:

        * ``"abs"`` : point where ``|y|`` is maximum (default).
        * ``"max"`` : maximum (most positive) value.
        * ``"min"`` : minimum (most negative) value.

        Ignored for Fourier / PSD plots.
    processed : bool, optional
        If ``True`` (default), use processed channel data when computing
        the peak. If ``False``, use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        channel's processed-data cache.
    fmt : str, optional
        Text format string for the annotation. If ``None`` (default),
        a sensible default is used depending on ``plot``:

        * time-domain: ``"{y:.3g} {units}"`` or ``"{y:.3g}"``.
        * Fourier   : ``"{f:.3g} Hz\\n{amp:.3g}"``.
        * PSD       : ``"{f:.3g} Hz\\n{p:.3g}"``.

        The format string is applied with :py:meth:`str.format` and can use:

        * time-domain: ``t``, ``y``.
        * Fourier   : ``f``, ``amp``.
        * PSD       : ``f``, ``p``.
    welch_kwargs : mapping, optional
        Extra keyword arguments forwarded to :meth:`Channel.welch_psd` when
        ``plot`` is ``"psd"`` or ``"welch"`` (for example ``nperseg``).
    **annotate_kwargs
        Additional keyword arguments forwarded to :meth:`matplotlib.axes.Axes.annotate`,
        e.g. ``fontsize``, ``arrowprops``, ``ha``, ``va``, etc.

    Returns
    -------
    None
        This function modifies the given axes in place.

    Raises
    ------
    ValueError
        If an unsupported ``plot`` type is given, or if the underlying
        peak computation fails (for example on an empty channel).
    """
    plot_norm = plot.lower()
    if plot_norm in {"time", "timehistory"}:
        _annotate_time_peak(
            channel=channel,
            ax=ax,
            peak=peak,
            processed=processed,
            use_cache=use_cache,
            fmt=fmt,
            **annotate_kwargs,
        )
    elif plot_norm == "fourier":
        _annotate_fourier_peak(
            channel=channel,
            ax=ax,
            processed=processed,
            use_cache=use_cache,
            fmt=fmt,
            **annotate_kwargs,
        )
    elif plot_norm in {"psd", "welch"}:
        _annotate_psd_peak(
            channel=channel,
            ax=ax,
            processed=processed,
            use_cache=use_cache,
            fmt=fmt,
            welch_kwargs=welch_kwargs,
            **annotate_kwargs,
        )
    else:
        raise ValueError(
            "Unsupported plot type for annotate_peak: "
            f"{plot!r}. Expected 'time', 'timehistory', 'fourier', 'psd' or 'welch'."
        )


def _annotate_time_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    peak: Literal["abs", "max", "min"] = "abs",
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate a time-domain peak on an existing time-history plot.

    Parameters
    ----------
    channel : Channel
        Channel providing the time-history data.
    ax : matplotlib.axes.Axes
        Axes on which the time history has already been plotted.
    peak : {"abs", "max", "min"}, optional
        Type of peak to annotate (see :func:`annotate_peak`). Default is
        ``"abs"``.
    processed : bool, optional
        If ``True`` (default), use processed data; otherwise use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        processing cache.
    fmt : str, optional
        Format string for the annotation text. Defaults to
        ``"{y:.3g} {units}"`` or ``"{y:.3g}"`` depending on the channel
        units.
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.

    Returns
    -------
    None
        The axes are modified in place.
    """
    if peak == "abs":
        t_peak, y_peak = channel.max_abs(processed=processed, use_cache=use_cache)
    elif peak == "max":
        t_peak, y_peak = channel.max_value(processed=processed, use_cache=use_cache)
    elif peak == "min":
        t_peak, y_peak = channel.min_value(processed=processed, use_cache=use_cache)
    else:
        raise ValueError("peak must be one of 'abs', 'max', or 'min'")

    if fmt is None:
        if channel.units:
            fmt = "{y:.3g} " + channel.units
        else:
            fmt = "{y:.3g}"

    text = fmt.format(t=t_peak, y=y_peak)

    ax.plot([t_peak], [y_peak], "o")
    # only set xytext/textcoords defaults if caller didn't provide them
    local_annotate_kwargs = dict(annotate_kwargs)
    if "xytext" not in local_annotate_kwargs:
        local_annotate_kwargs["xytext"] = (5, 5)
    if "textcoords" not in local_annotate_kwargs:
        local_annotate_kwargs["textcoords"] = "offset points"
    ax.annotate(
        text,
        xy=(t_peak, y_peak),
        ha="left",
        va="bottom",
        **local_annotate_kwargs,
    )


def _annotate_fourier_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate the dominant Fourier amplitude on a spectrum plot.

    Parameters
    ----------
    channel : Channel
        Channel providing the data for the Fourier spectrum.
    ax : matplotlib.axes.Axes
        Axes on which the Fourier spectrum has already been plotted.
    processed : bool, optional
        If ``True`` (default), use processed data; otherwise use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        processing cache.
    fmt : str, optional
        Format string for the annotation text. Defaults to
        ``"{f:.3g} Hz\\n{amp:.3g}"``.
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.

    Returns
    -------
    None
        The axes are modified in place.
    """
    spec = channel.fourier(processed=processed, use_cache=use_cache)
    f_peak, s_peak = spec.peak()

    if fmt is None:
        fmt = "{f:.3g} Hz\n{amp:.3g}"

    text = fmt.format(f=f_peak, amp=s_peak)

    ax.plot([f_peak], [s_peak], "o")
    # only set xytext/textcoords defaults if caller didn't provide them
    local_annotate_kwargs = dict(annotate_kwargs)
    if "xytext" not in local_annotate_kwargs:
        local_annotate_kwargs["xytext"] = (5, 5)
    if "textcoords" not in local_annotate_kwargs:
        local_annotate_kwargs["textcoords"] = "offset points"
    ax.annotate(
        text,
        xy=(f_peak, s_peak),
        ha="left",
        va="bottom",
        **local_annotate_kwargs,
    )


def _annotate_psd_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    welch_kwargs: Mapping[str, Any] | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate the dominant PSD value on a Welch spectrum plot.

    Parameters
    ----------
    channel : Channel
        Channel providing the data for the PSD.
    ax : matplotlib.axes.Axes
        Axes on which the PSD (Welch) has already been plotted.
    processed : bool, optional
        If ``True`` (default), use processed data; otherwise use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        processing cache.
    fmt : str, optional
        Format string for the annotation text. Defaults to
        ``"{f:.3g} Hz\\n{p:.3g}"``.
    welch_kwargs : mapping, optional
        Additional keyword arguments forwarded to :meth:`Channel.welch_psd`
        when computing the PSD (for example ``nperseg``).
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.

    Returns
    -------
    None
        The axes are modified in place.
    """
    psd = channel.welch_psd(
        processed=processed,
        use_cache=use_cache,
        **({} if welch_kwargs is None else dict(welch_kwargs)),
    )
    f_peak, p_peak = psd.peak()

    if fmt is None:
        fmt = "{f:.3g} Hz\n{p:.3g}"

    text = fmt.format(f=f_peak, p=p_peak)

    ax.plot([f_peak], [p_peak], "o")
    # only set xytext/textcoords defaults if caller didn't provide them
    local_annotate_kwargs = dict(annotate_kwargs)
    if "xytext" not in local_annotate_kwargs:
        local_annotate_kwargs["xytext"] = (5, 5)
    if "textcoords" not in local_annotate_kwargs:
        local_annotate_kwargs["textcoords"] = "offset points"
    ax.annotate(
        text,
        xy=(f_peak, p_peak),
        ha="left",
        va="bottom",
        **local_annotate_kwargs,
    )


def annotate_time_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    peak: Literal["abs", "max", "min"] = "abs",
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Convenience wrapper to annotate a time-history peak.

    Parameters
    ----------
    channel : Channel
        Channel whose time-domain peak will be annotated.
    ax : matplotlib.axes.Axes
        Axes on which the time history has already been plotted.
    peak : {"abs", "max", "min"}, optional
        Type of time-domain peak to annotate. See :func:`annotate_peak`.
        Default is ``"abs"``.
    processed : bool, optional
        If ``True`` (default), use processed data; otherwise use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        processing cache.
    fmt : str, optional
        Format string for the annotation text. If ``None``, a default based
        on channel units is used.
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.
    """
    annotate_peak(
        channel=channel,
        ax=ax,
        plot="time",
        peak=peak,
        processed=processed,
        use_cache=use_cache,
        fmt=fmt,
        **annotate_kwargs,
    )


def annotate_fourier_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Convenience wrapper to annotate the dominant Fourier amplitude.

    Parameters
    ----------
    channel : Channel
        Channel whose Fourier peak will be annotated.
    ax : matplotlib.axes.Axes
        Axes on which the Fourier spectrum has already been plotted.
    processed : bool, optional
        If ``True`` (default), use processed data; otherwise use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        processing cache.
    fmt : str, optional
        Format string for the annotation text. If ``None``, the default
        ``"{f:.3g} Hz\\n{amp:.3g}"`` is used.
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.
    """
    annotate_peak(
        channel=channel,
        ax=ax,
        plot="fourier",
        processed=processed,
        use_cache=use_cache,
        fmt=fmt,
        **annotate_kwargs,
    )


def annotate_psd_peak(
    channel: Channel,
    ax: plt.Axes,
    *,
    processed: bool = True,
    use_cache: bool = True,
    fmt: str | None = None,
    welch_kwargs: Mapping[str, Any] | None = None,
    **annotate_kwargs: Any,
) -> None:
    """
    Convenience wrapper to annotate the dominant PSD (Welch) peak.

    Parameters
    ----------
    channel : Channel
        Channel whose PSD peak will be annotated.
    ax : matplotlib.axes.Axes
        Axes on which the PSD (Welch) has already been plotted.
    processed : bool, optional
        If ``True`` (default), use processed data; otherwise use raw data.
    use_cache : bool, optional
        If ``True`` (default) and ``processed`` is ``True``, use the
        processing cache.
    fmt : str, optional
        Format string for the annotation text. If ``None``, the default
        ``"{f:.3g} Hz\\n{p:.3g}"`` is used.
    welch_kwargs : mapping, optional
        Additional keyword arguments forwarded to :meth:`Channel.welch_psd`
        when computing the PSD.
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.
    """
    annotate_peak(
        channel=channel,
        ax=ax,
        plot="psd",
        peak="abs",  # ignored by PSD branch
        processed=processed,
        use_cache=use_cache,
        fmt=fmt,
        welch_kwargs=welch_kwargs,
        **annotate_kwargs,
    )


def annotate_response_peak(
    rs: ResponseSpectrum,
    ax: plt.Axes,
    *,
    fmt: str = "{T:.3g} s\\n{Sa:.3g} g",
    **annotate_kwargs: Any,
) -> None:
    """
    Annotate the peak spectral acceleration on a response spectrum plot.

    This helper assumes a :class:`ResponseSpectrum` has already been
    plotted on ``ax`` via :meth:`ResponseSpectrum.plot`. It finds the
    period at which spectral acceleration is maximum, marks the point
    and draws a text annotation.

    Parameters
    ----------
    rs : ResponseSpectrum
        Response spectrum object whose peak is to be annotated.
    ax : matplotlib.axes.Axes
        Axes on which the response spectrum has already been plotted.
    fmt : str, optional
        Format string for the annotation text. It is applied with
        :py:meth:`str.format` and can use ``T`` (period in seconds) and
        ``Sa`` (spectral acceleration in g). Default is
        ``"{T:.3g} s\\n{Sa:.3g} g"``.
    **annotate_kwargs
        Additional keyword arguments forwarded to ``ax.annotate``.

    Returns
    -------
    None
        The axes are modified in place.
    """
    T_peak, Sa_peak = rs.peak()

    # ensure default includes a real newline (if user left default)
    if fmt == "{T:.3g} s\\n{Sa:.3g} g":
        fmt = "{T:.3g} s\n{Sa:.3g} g"
    text = fmt.format(T=T_peak, Sa=Sa_peak)

    ax.plot([T_peak], [Sa_peak], "o")
    # only set xytext/textcoords defaults if caller didn't provide them
    local_annotate_kwargs = dict(annotate_kwargs)
    if "xytext" not in local_annotate_kwargs:
        local_annotate_kwargs["xytext"] = (5, 5)
    if "textcoords" not in local_annotate_kwargs:
        local_annotate_kwargs["textcoords"] = "offset points"
    ax.annotate(
        text,
        xy=(T_peak, Sa_peak),
        ha="left",
        va="bottom",
        **local_annotate_kwargs,
    )
