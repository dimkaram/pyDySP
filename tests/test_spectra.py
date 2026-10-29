import matplotlib.pyplot as plt
import numpy as np
import pytest

from pydysp.spectra import FourierSpectrum, WelchSpectrum


def test_fourier_spectrum_peak_single_tone():
    """Single-tone signal should produce a Fourier peak at the input frequency."""
    fs = 100.0  # Hz
    dt = 1.0 / fs
    t = np.arange(0.0, 2.0, dt)
    f0 = 5.0  # Hz
    y = np.sin(2.0 * np.pi * f0 * t)

    # Zero-pad FFT, mirroring Channel.fourier behaviour
    n = len(y)
    n_fft = 1 << (n - 1).bit_length()
    f = np.fft.rfftfreq(n=n_fft, d=dt)
    s = np.abs(np.fft.rfft(y, n=n_fft))
    spec = FourierSpectrum(f=f, s=s)

    f_peak, _ = spec.peak()
    assert f_peak == pytest.approx(f0, rel=0.02)


def test_welch_spectrum_peak_single_tone():
    """Single-tone signal should produce a Welch peak at the input frequency."""
    from scipy.signal import welch

    fs = 100.0
    dt = 1.0 / fs
    t = np.arange(0.0, 2.0, dt)
    f0 = 5.0
    y = np.sin(2.0 * np.pi * f0 * t)

    f, p = welch(y, fs=fs, nperseg=min(256, len(y)))
    psd = WelchSpectrum(f=f, p=p)

    f_peak, _ = psd.peak()
    assert f_peak == pytest.approx(f0, rel=0.05)


def test_fourier_plot_smoke():
    """Smoke test for FourierSpectrum.plot."""
    fs = 50.0
    dt = 1.0 / fs
    t = np.arange(0.0, 1.0, dt)
    y = np.sin(2.0 * np.pi * 3.0 * t)

    n_fft = 1 << (len(y) - 1).bit_length()
    f = np.fft.rfftfreq(n=n_fft, d=dt)
    s = np.abs(np.fft.rfft(y, n=n_fft))
    spec = FourierSpectrum(f=f, s=s)

    fig, ax = plt.subplots()
    spec.plot(ax=ax, fmax=20.0)
    plt.close(fig)


def test_welch_plot_smoke():
    """Smoke test for WelchSpectrum.plot."""
    from scipy.signal import welch

    fs = 50.0
    dt = 1.0 / fs
    t = np.arange(0.0, 1.0, dt)
    y = np.sin(2.0 * np.pi * 3.0 * t)

    f, p = welch(y, fs=fs, nperseg=min(128, len(y)))
    psd = WelchSpectrum(f=f, p=p)

    fig, ax = plt.subplots()
    psd.plot(ax=ax, fmax=20.0)
    plt.close(fig)
