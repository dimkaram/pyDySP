# response.py
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ResponseSpectrum:
    """
    Elastic response spectrum for a family of SDOF oscillators.

    This dataclass stores spectral displacement, velocity and acceleration
    for a set of single-degree-of-freedom (SDOF) oscillators with the same
    damping ratio, evaluated over a grid of natural periods.

    Parameters
    ----------
    T : np.ndarray
        One-dimensional array of natural periods in seconds.
    Sd : np.ndarray
        Spectral displacement values in metres (same shape as ``T``).
    Sv : np.ndarray
        Spectral velocity values in metres per second (same shape as ``T``).
    Sa : np.ndarray
        Spectral pseudo-acceleration values in g (same shape as ``T``).
    ksi : float
        Damping ratio used to compute the spectrum (e.g. 0.05 for 5%).
    """

    T: np.ndarray
    Sd: np.ndarray
    Sv: np.ndarray
    Sa: np.ndarray
    ksi: float

    def peak(self) -> tuple[float, float]:
        """
        Return the dominant period and peak spectral acceleration.

        Returns
        -------
        T_peak : float
            Period at which ``Sa`` attains its maximum value.
        Sa_peak : float
            Maximum spectral acceleration value in g.

        Raises
        ------
        ValueError
            If the spectral acceleration array ``Sa`` is empty.
        """
        if self.Sa.size == 0:
            raise ValueError(
                "Response spectrum is empty; cannot determine peak period and acceleration"
            )
        idx = int(np.argmax(self.Sa))
        return float(self.T[idx]), float(self.Sa[idx])

    def plot(
        self,
        ax: plt.Axes | None = None,
        y: str = "Sa",
        logx: bool = False,
        logy: bool = False,
        **plot_kwargs: Any,
    ) -> plt.Axes:
        """
        Plot one of the response spectra (Sa, Sv or Sd).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        y : {"Sa", "Sv", "Sd"}, optional
            Which spectrum to plot: spectral acceleration ``"Sa"`` (default),
            velocity ``"Sv"``, or displacement ``"Sd"``.
        logx : bool, optional
            If ``True``, use a logarithmic scale on the x-axis. Default is
            ``False``.
        logy : bool, optional
            If ``True``, use a logarithmic scale on the y-axis. Default is
            ``False``.
        **plot_kwargs
            Additional keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot`
            (for example ``color``, ``linewidth``, etc.).

        Returns
        -------
        matplotlib.axes.Axes
            Axes instance containing the plotted spectrum.

        Raises
        ------
        ValueError
            If ``y`` is not one of ``"Sa"``, ``"Sv"``, or ``"Sd"``.
        """
        if ax is None:
            _, ax = plt.subplots()
        match y:
            case "Sa":
                y_vals = self.Sa
                ylabel = "Spectral acceleration [g]"
            case "Sv":
                y_vals = self.Sv
                ylabel = "Spectral velocity [m/s]"
            case "Sd":
                y_vals = self.Sd
                ylabel = "Spectral displacement [m]"
            case _:
                raise ValueError("y must be one of 'Sa', 'Sv', 'Sd'")
        ax.plot(self.T, y_vals, **plot_kwargs)
        ax.set_xlabel("Period [s]")
        ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="both")
        return ax


def sdof_newmark_response(
    acc: np.ndarray,
    dt: float,
    omega: float,
    ksi: float,
) -> tuple[float, float, float]:
    """
    Newmark-beta (average-acceleration) SDOF response to base acceleration.

    This function integrates the response of a linear single-degree-of-freedom
    (SDOF) oscillator subjected to ground acceleration using the Newmark-beta
    average-acceleration scheme, and returns peak relative displacement,
    relative velocity and absolute acceleration.

    Parameters
    ----------
    acc : np.ndarray
        Ground acceleration time history :math:`a_g(t)` in m/s².
    dt : float
        Time step in seconds.
    omega : float
        Circular frequency of the oscillator in rad/s.
    ksi : float
        Damping ratio (e.g. 0.05 for 5% damping).

    Returns
    -------
    Sd : float
        Peak relative displacement in metres (maximum absolute value of
        relative displacement time history).
    Sv : float
        Peak relative velocity in metres per second.
    Sa : float
        Peak absolute acceleration in m/s² (sum of relative and ground
        acceleration).

    Notes
    -----
    * A unit mass :math:`m = 1` is assumed, with stiffness ``k = m * omega**2``
      and viscous damping ``c = 2 * ksi * omega * m``.
    * If the input acceleration array is empty, all returned values are 0.0.
    """
    n = len(acc)
    if n == 0:
        return 0.0, 0.0, 0.0

    # SDOF properties for unit mass
    m = 1.0
    k = m * omega**2
    c = 2.0 * ksi * omega * m

    # Newmark average-acceleration parameters (γ, β)
    gamma = 0.5
    beta = 0.25

    # State vectors for relative displacement, velocity, and acceleration
    u = np.zeros(n)
    v = np.zeros(n)
    a_rel = np.zeros(n)

    # Initial relative acceleration from equilibrium at t = 0
    a_rel[0] = (-acc[0] - c * v[0] - k * u[0]) / m

    # Precompute Newmark effective-stiffness coefficients
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    k_eff = k + a0 * m + a1 * c

    # Effective load due to base acceleration
    p = -m * acc

    # Time-stepping loop (Newmark average-acceleration scheme)
    for i in range(n - 1):
        dp = (
            p[i + 1]
            + m * (a0 * u[i] + a2 * v[i] + a3 * a_rel[i])
            + c * (a1 * u[i] + a4 * v[i] + a5 * a_rel[i])
        )
        u[i + 1] = dp / k_eff
        a_rel[i + 1] = a0 * (u[i + 1] - u[i]) - a2 * v[i] - a3 * a_rel[i]
        v[i + 1] = v[i] + dt * ((1.0 - gamma) * a_rel[i] + gamma * a_rel[i + 1])

    Sd = float(np.max(np.abs(u)))
    Sv = float(np.max(np.abs(v)))
    a_abs = a_rel + acc
    Sa = float(np.max(np.abs(a_abs)))
    return Sd, Sv, Sa
