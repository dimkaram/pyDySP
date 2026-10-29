# spectra.py
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FourierSpectrum:
    """
    Single-sided Fourier amplitude spectrum.

    This dataclass stores the frequency axis and corresponding single-sided
    amplitude spectrum (magnitude of the FFT) for a real-valued time series.

    Parameters
    ----------
    f : np.ndarray
        One-dimensional array of frequencies in Hz.
    s : np.ndarray
        One-dimensional array of amplitude values corresponding to ``f``.
    """

    f: np.ndarray
    s: np.ndarray

    def peak(self) -> tuple[float, float]:
        """
        Return the frequency and amplitude at the maximum spectral peak.

        Returns
        -------
        f_peak : float
            Frequency at which the amplitude spectrum is maximum.
        s_peak : float
            Maximum amplitude value.

        Raises
        ------
        ValueError
            If the spectrum array ``s`` is empty.
        """
        if self.s.size == 0:
            raise ValueError(
                "Spectrum is empty; cannot determine peak frequency and amplitude"
            )
        idx = int(np.argmax(self.s))
        return float(self.f[idx]), float(self.s[idx])

    def plot(
        self,
        ax: plt.Axes | None = None,
        fmax: float | None = 50.0,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Fourier amplitude spectrum.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        fmax : float or None, optional
            Upper x-limit for the frequency axis in Hz. If ``None``, the full
            frequency range is shown. Default is ``50.0``.
        **plot_kwargs
            Additional keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.plot` (for example ``color``,
            ``linewidth``).

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance with the plotted spectrum.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.f, self.s, **plot_kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Fourier amplitude")
        if fmax is not None:
            ax.set_xlim(0.0, fmax)
        ax.grid(True)
        return ax


@dataclass
class WelchSpectrum:
    """
    Power spectral density (PSD) result from Welch's method.

    This dataclass stores the frequency axis and corresponding power
    spectral density values obtained using Welch's method.

    Parameters
    ----------
    f : np.ndarray
        One-dimensional array of frequencies in Hz.
    p : np.ndarray
        One-dimensional array of PSD values corresponding to ``f``.
    """

    f: np.ndarray
    p: np.ndarray

    def peak(self) -> tuple[float, float]:
        """
        Return the frequency and PSD value at the maximum spectral peak.

        Returns
        -------
        f_peak : float
            Frequency at which the PSD is maximum.
        p_peak : float
            Maximum PSD value.

        Raises
        ------
        ValueError
            If the PSD array ``p`` is empty.
        """
        if self.p.size == 0:
            raise ValueError("PSD is empty; cannot determine peak frequency and value")
        idx = int(np.argmax(self.p))
        return float(self.f[idx]), float(self.p[idx])

    def plot(
        self,
        ax: plt.Axes | None = None,
        fmax: float | None = 50.0,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot the Welch power spectral density.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        fmax : float or None, optional
            Upper x-limit for the frequency axis in Hz. If ``None``, the full
            frequency range is shown. Default is ``50.0``.
        **plot_kwargs
            Additional keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance with the plotted PSD.
        """
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.f, self.p, **plot_kwargs)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD")
        if fmax is not None:
            ax.set_xlim(0.0, fmax)
        ax.grid(True)
        return ax
