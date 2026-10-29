import matplotlib.pyplot as plt
import numpy as np
import pytest

from pydysp.response import ResponseSpectrum, sdof_newmark_response


def test_sdof_newmark_response_zero_acceleration():
    """Zero ground acceleration should produce zero SDOF response."""
    dt = 0.01
    t = np.arange(0.0, 5.0, dt)
    acc = np.zeros_like(t)
    omega = 2.0 * np.pi / 1.0  # natural period = 1 s
    ksi = 0.05

    Sd, Sv, Sa = sdof_newmark_response(acc=acc, dt=dt, omega=omega, ksi=ksi)

    assert Sd == pytest.approx(0.0)
    assert Sv == pytest.approx(0.0)
    assert Sa == pytest.approx(0.0)


def test_response_spectrum_zero_input_is_zero():
    """With zero spectral values, peak detection and plotting should behave correctly."""
    periods = np.array([0.5, 1.0, 2.0])
    Sd = np.zeros_like(periods)
    Sv = np.zeros_like(periods)
    Sa = np.zeros_like(periods)
    ksi = 0.05

    rs = ResponseSpectrum(T=periods, Sd=Sd, Sv=Sv, Sa=Sa, ksi=ksi)

    # Peak should occur at the first period (tie → pick first)
    T_peak, Sa_peak = rs.peak()
    assert T_peak == pytest.approx(periods[0])
    assert Sa_peak == pytest.approx(0.0)

    # Smoke test for plotting all spectral components
    fig, ax = plt.subplots()
    rs.plot(ax=ax, y="Sa")
    rs.plot(ax=ax, y="Sv")
    rs.plot(ax=ax, y="Sd")
    plt.close(fig)
