import matplotlib.pyplot as plt
import numpy as np

from pydysp.channel import Channel
from pydysp.helpers import (
    annotate_fourier_peak,
    annotate_peak,
    annotate_psd_peak,
    annotate_response_peak,
    annotate_time_peak,
)
from pydysp.response import ResponseSpectrum

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def make_sine_channel(
    f0: float = 5.0,
    fs: float = 100.0,
    duration: float = 2.0,
    units: str | None = "m/s^2",
) -> Channel:
    """Create a simple sine-wave Channel for peak-annotation tests."""
    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)
    y = np.sin(2.0 * np.pi * f0 * t)
    return Channel(data=y, dt=dt, units=units)


# ----------------------------------------------------------------------
# Generic dispatcher tests
# ----------------------------------------------------------------------


def test_annotate_peak_time_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot(ax=ax)

    n_text_before = len(ax.texts)
    annotate_peak(ch, ax, plot="time", peak="abs")
    n_text_after = len(ax.texts)

    # At least one new text annotation should have been added
    assert n_text_after >= n_text_before + 1

    plt.close(fig)


def test_annotate_peak_fourier_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot_fourier(ax=ax, fmax=20.0)

    n_text_before = len(ax.texts)
    annotate_peak(ch, ax, plot="fourier")
    n_text_after = len(ax.texts)

    assert n_text_after >= n_text_before + 1

    plt.close(fig)


def test_annotate_peak_psd_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot_psd(ax=ax, fmax=20.0, nperseg=128)

    n_text_before = len(ax.texts)
    annotate_peak(
        ch,
        ax,
        plot="psd",
        welch_kwargs={"nperseg": 128},
    )
    n_text_after = len(ax.texts)

    assert n_text_after >= n_text_before + 1

    plt.close(fig)


def test_annotate_peak_timehistory_alias_and_welch_alias():
    """Check that alternative plot names 'timehistory' and 'welch' work."""
    ch = make_sine_channel()

    # timehistory alias
    fig1, ax1 = plt.subplots()
    ch.plot(ax=ax1)
    annotate_peak(ch, ax1, plot="timehistory", peak="max")
    plt.close(fig1)

    # welch alias
    fig2, ax2 = plt.subplots()
    ch.plot_psd(ax=ax2, fmax=20.0)
    annotate_peak(ch, ax2, plot="welch")
    plt.close(fig2)


# ----------------------------------------------------------------------
# Convenience wrapper tests
# ----------------------------------------------------------------------


def test_annotate_time_peak_wrapper_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot(ax=ax)

    n_text_before = len(ax.texts)
    annotate_time_peak(ch, ax, peak="min")
    n_text_after = len(ax.texts)

    assert n_text_after >= n_text_before + 1

    plt.close(fig)


def test_annotate_fourier_peak_wrapper_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot_fourier(ax=ax, fmax=20.0)

    n_text_before = len(ax.texts)
    annotate_fourier_peak(ch, ax)
    n_text_after = len(ax.texts)

    assert n_text_after >= n_text_before + 1

    plt.close(fig)


def test_annotate_psd_peak_wrapper_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot_psd(ax=ax, fmax=20.0, nperseg=64)

    n_text_before = len(ax.texts)
    annotate_psd_peak(ch, ax, welch_kwargs={"nperseg": 64})
    n_text_after = len(ax.texts)

    assert n_text_after >= n_text_before + 1

    plt.close(fig)


def test_annotate_response_peak_smoke():
    ch = make_sine_channel()
    periods = np.array([0.2, 0.5, 1.0, 2.0])

    rs: ResponseSpectrum = ch.response_spectrum(periods=periods)

    fig, ax = plt.subplots()
    rs.plot(ax=ax, y="Sa")

    n_text_before = len(ax.texts)
    annotate_response_peak(rs, ax)
    n_text_after = len(ax.texts)

    assert n_text_after >= n_text_before + 1

    plt.close(fig)
