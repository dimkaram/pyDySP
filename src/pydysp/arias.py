# arias.py
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AriasResult:
    """
    Container for Arias intensity time history and significant-duration window.

    This dataclass stores the Husid-style cumulative Arias intensity and the
    corresponding significant-duration window, typically defined between two
    Arias intensity percentiles (e.g. 5–95%).

    Parameters
    ----------
    t : np.ndarray
        One-dimensional array of time values (in seconds) corresponding to the
        input acceleration record.
    Ia : np.ndarray
        One-dimensional array of cumulative Arias intensity values evaluated at
        times ``t``.
    t_start : float
        Start time of the significant-duration window (e.g. the time at which
        a given lower percentile of Arias intensity is reached).
    t_end : float
        End time of the significant-duration window (e.g. the time at which
        a given upper percentile of Arias intensity is reached).

    Notes
    -----
    The ``AriasResult`` class is typically created by higher-level routines
    that compute Arias intensity from acceleration time series. It is mainly
    used to:
    * Visualise the cumulative Arias intensity (Husid plot).
    * Provide a convenient representation of the significant-duration window
      for trimming and further analysis.
    """

    t: np.ndarray
    Ia: np.ndarray
    t_start: float
    t_end: float

    def plot(
        self,
        ax: plt.Axes | None = None,
        show_window: bool = True,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Arias intensity time history.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If ``None``, a new figure and axes are
            created with ``plt.subplots()``.
        show_window : bool, optional
            If ``True``, vertical dashed lines are drawn at ``t_start`` and
            ``t_end`` to indicate the significant-duration window. The default
            is ``True``.
        **plot_kwargs
            Additional keyword arguments passed directly to ``ax.plot`` (for
            example ``color``, ``linewidth``, etc.).

        Returns
        -------
        matplotlib.axes.Axes
            The axes instance containing the Arias intensity plot.

        Notes
        -----
        The x-axis is labelled as time in seconds, and the y-axis as Arias
        intensity in consistent units with the underlying acceleration record.
        A grid is enabled by default to facilitate reading the Husid plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.t, self.Ia, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Arias intensity")
        if show_window:
            ax.axvline(self.t_start, linestyle="--", c="gray")
            ax.axvline(self.t_end, linestyle="--", c="gray")
        ax.grid(True)
        return ax
