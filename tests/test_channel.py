import numpy as np
import matplotlib.pyplot as plt
import pytest

from pydysp.channel import Channel

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def make_sine_channel(
    f0: float = 5.0,
    fs: float = 100.0,
    duration: float = 2.0,
    amplitude: float = 1.0,
) -> Channel:
    """Convenience helper to create a simple sine-wave Channel."""
    dt = 1.0 / fs
    t = np.arange(0.0, duration, dt)
    y = amplitude * np.sin(2.0 * np.pi * f0 * t)
    return Channel(
        data=y,
        dt=dt,
        quantity="acceleration",
        units="g",
        name_input="Acc1",
    )


# ----------------------------------------------------------------------
# Metadata / init
# ----------------------------------------------------------------------


def test_channel_init_time_and_labels():
    dt = 0.01
    data = np.arange(100)
    ch = Channel(
        data=data,
        dt=dt,
        name_input="RawName",
        name_user=None,
        units="g",
    )
    assert ch.time.shape == ch.data.shape
    assert ch.dt == pytest.approx(dt)
    assert ch.name_user == "RawName"
    assert ch.label_legend == "RawName"
    assert "g" in ch.label_axis


def test_channel_info_contains_basic_fields():
    ch = make_sine_channel()
    info = ch.info()
    assert "Channel:" in info
    assert "Length:" in info
    assert "Sampling frequency:" in info
    assert "Quantity:" in info or "Units:" in info


def test_channel_duration_computes_correctly():
    dt = 0.01
    t = np.arange(0, 5, dt)
    y = np.sin(2 * np.pi * t)

    ch = Channel(data=y, dt=dt)

    # Same definition as Test.duration (last minus first sample)
    expected_duration = t[-1] - t[0]
    assert ch.duration == pytest.approx(expected_duration)


def test_channel_duration_empty_signal():
    ch = Channel(data=np.array([]), dt=0.01)
    assert ch.duration == 0.0


# ----------------------------------------------------------------------
# Processing steps
# ----------------------------------------------------------------------


def test_drift_corrected_removes_initial_mean():
    fs = 100.0
    dt = 1.0 / fs
    t = np.arange(0.0, 2.0, dt)
    offset = 2.0
    y = offset + 0.1 * np.sin(2.0 * np.pi * 2.0 * t)
    ch = Channel(data=y, dt=dt)

    ch_dc = ch.drift_corrected(points=100)
    _, y_dc = ch_dc.processed(use_cache=False)
    # Mean of first 100 samples should be ~0 after drift correction
    assert np.mean(y_dc[:100]) == pytest.approx(0.0, abs=1e-3)


def test_filtered_band_limited_sine():
    fs = 200.0
    dt = 1.0 / fs
    t = np.arange(0.0, 2.0, dt)
    y = np.sin(2.0 * np.pi * 10.0 * t) + 0.5 * np.sin(2.0 * np.pi * 60.0 * t)
    ch = Channel(data=y, dt=dt)

    # Lowpass at 20 Hz should suppress the 60 Hz component
    ch_filt = ch.filtered(btype="lowpass", fc=20.0, order=4)
    _, y_filt = ch_filt.processed(use_cache=False)

    # Rough check: high-frequency energy is reduced
    assert np.std(y_filt) < np.std(y)


def test_trim_by_threshold():
    fs = 50.0
    dt = 1.0 / fs
    t = np.arange(0.0, 4.0, dt)
    y = np.zeros_like(t)
    # Pulse between 1 s and 3 s at amplitude 1.0
    y[(t >= 1.0) & (t <= 3.0)] = 1.0
    ch = Channel(data=y, dt=dt)

    ch_trim = ch.trim_by_threshold(threshold=0.5)
    t_trim, _ = ch_trim.processed(use_cache=False)

    assert t_trim[0] == pytest.approx(1.0, rel=0.01)
    assert t_trim[-1] == pytest.approx(3.0, rel=0.01)


def test_trim_by_fraction_of_peak():
    ch = make_sine_channel()
    ch_trim = ch.trim_by_fraction_of_peak(fraction=0.5)
    t_trim, y_trim = ch_trim.processed(use_cache=False)

    # At least some samples should be above ~0.5 of peak
    assert np.max(np.abs(y_trim)) >= 0.5 * np.max(np.abs(ch.data))


# ----------------------------------------------------------------------
# Time-domain metrics
# ----------------------------------------------------------------------


def test_max_min_rms_simple_sine():
    ch = make_sine_channel(f0=2.0, fs=100.0, duration=1.0, amplitude=2.0)
    t_max, y_max = ch.max_value()
    t_min, y_min = ch.min_value()
    rms = ch.rms()

    assert y_max == pytest.approx(2.0, rel=0.05)
    assert y_min == pytest.approx(-2.0, rel=0.05)
    # RMS of sine with amplitude A is A/sqrt(2)
    assert rms == pytest.approx(2.0 / np.sqrt(2), rel=0.05)

    # max_abs should correspond to either y_max or y_min
    t_abs, y_abs = ch.max_abs()
    assert abs(y_abs) == pytest.approx(2.0, rel=0.05)
    assert t_abs >= 0.0


# ----------------------------------------------------------------------
# Spectra from Channel
# ----------------------------------------------------------------------


def test_channel_fourier_peak_matches_frequency():
    ch = make_sine_channel(f0=5.0, fs=200.0, duration=2.0)
    f_peak, _ = ch.fourier_peak()
    assert f_peak == pytest.approx(5.0, rel=0.02)


def test_channel_welch_peak_matches_frequency():
    ch = make_sine_channel(f0=3.0, fs=200.0, duration=2.0)
    f_peak, _ = ch.welch_peak()
    assert f_peak == pytest.approx(3.0, rel=0.05)


# ----------------------------------------------------------------------
# Arias intensity via Channel
# ----------------------------------------------------------------------


def test_arias_intensity_monotonic_and_positive():
    ch = make_sine_channel(f0=5.0, fs=200.0, duration=5.0, amplitude=0.1)
    res = ch.arias_intensity(g=9.81)

    # Ia should be non-decreasing
    diff = np.diff(res.Ia)
    assert np.all(diff >= -1e-10)
    assert res.Ia[-1] > 0.0
    assert res.t_start < res.t_end


def test_trim_by_arias_reduces_duration():
    ch = make_sine_channel(f0=5.0, fs=200.0, duration=10.0, amplitude=0.2)
    ch_trim = ch.trim_by_arias(lower=0.05, upper=0.95, g=9.81)
    t_full, _ = ch.xy(processed=False)
    t_trim, _ = ch_trim.processed(use_cache=False)

    # Duration after Arias-based trimming should be shorter than full record
    assert t_trim[-1] - t_trim[0] < t_full[-1] - t_full[0]


# ----------------------------------------------------------------------
# Response spectrum via Channel
# ----------------------------------------------------------------------


def test_response_spectrum_zero_signal_all_zero():
    fs = 100.0
    dt = 1.0 / fs
    t = np.arange(0.0, 5.0, dt)
    y = np.zeros_like(t)
    ch = Channel(data=y, dt=dt, quantity="acceleration", units="g")

    periods = np.array([0.5, 1.0, 2.0])
    rs = ch.response_spectrum(periods=periods, g=9.81, ksi=0.05)

    assert np.allclose(rs.Sd, 0.0)
    assert np.allclose(rs.Sv, 0.0)
    assert np.allclose(rs.Sa, 0.0)


def test_response_spectrum_smoke_nonzero_signal():
    ch = make_sine_channel(f0=2.0, fs=50.0, duration=10.0, amplitude=0.05)
    periods = np.array([0.5, 1.0, 2.0])
    rs = ch.response_spectrum(periods=periods, g=9.81, ksi=0.05)

    # Basic sanity checks on shapes and values
    assert rs.T.shape == periods.shape
    assert rs.Sd.shape == periods.shape
    assert rs.Sv.shape == periods.shape
    assert rs.Sa.shape == periods.shape
    # Spectral acceleration should be non-negative
    assert np.all(rs.Sa >= 0.0)


# ----------------------------------------------------------------------
# Plotting smoke tests
# ----------------------------------------------------------------------


def test_channel_plot_smoke():
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    ch.plot(ax=ax, include_label=True, include_legend=True)
    plt.close(fig)


def test_channel_spectral_plots_smoke():
    ch = make_sine_channel()

    fig1, ax1 = plt.subplots()
    ch.plot_fourier(ax=ax1, fmax=20.0)

    fig2, ax2 = plt.subplots()
    ch.plot_psd(ax=ax2, fmax=20.0)

    fig3, ax3 = plt.subplots()
    ch.plot_arias(ax=ax3, g=9.81, show_window=True)

    fig4, ax4 = plt.subplots()
    ch.plot_response_spectrum(ax=ax4, periods=np.array([0.5, 1.0, 2.0]))

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


def test_channel_plot_type_dispatch_smoke():
    """Channel.plot should delegate correctly for supported plot_type values."""
    ch = make_sine_channel()

    # Time history
    fig1, ax1 = plt.subplots()
    ch.plot(ax=ax1, plot_type="timehistory", include_label=True, include_legend=True)
    plt.close(fig1)

    # Fourier spectrum
    fig2, ax2 = plt.subplots()
    ch.plot(ax=ax2, plot_type="fourier", fmax=20.0)
    plt.close(fig2)

    # PSD / Welch
    fig3, ax3 = plt.subplots()
    ch.plot(ax=ax3, plot_type="psd", fmax=20.0)
    plt.close(fig3)

    # Arias intensity
    fig4, ax4 = plt.subplots()
    ch.plot(ax=ax4, plot_type="arias", g=9.81, show_window=True)
    plt.close(fig4)

    # Response spectrum
    fig5, ax5 = plt.subplots()
    ch.plot(
        ax=ax5,
        plot_type="response",
        periods=np.array([0.5, 1.0, 2.0]),
    )
    plt.close(fig5)


def test_channel_plot_type_invalid_raises():
    """Channel.plot should raise for an unsupported plot_type."""
    ch = make_sine_channel()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Unknown plot_type"):
        ch.plot(ax=ax, plot_type="not_a_plot_type")
    plt.close(fig)
