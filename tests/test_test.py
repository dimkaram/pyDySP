import csv
import numpy as np
import pytest
import matplotlib.pyplot as plt

from pydysp.channel import Channel
from pydysp.test import Test

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def make_channel(
    n: int = 1000,
    dt: float = 0.01,
    freq: float = 1.0,
    amp: float = 1.0,
    phase: float = 0.0,
    name_user: str | None = None,
    name_input: str | None = None,
    quantity: str | None = None,
    units: str | None = None,
    tags: set[str] | None = None,
) -> Channel:
    """Helper to build a simple sinusoidal Channel."""
    t = np.arange(n) * dt
    data = amp * np.sin(2.0 * np.pi * freq * t + phase)
    return Channel(
        data=data,
        dt=dt,
        name_user=name_user,
        name_input=name_input,
        quantity=quantity,
        units=units,
        tags=tags or set(),
    )


def make_ch(data, dt: float = 0.01, name: str = "Ch") -> Channel:
    """Helper for channel_health tests with explicit data arrays."""
    return Channel(
        data=np.asarray(data),
        dt=dt,
        name_user=name,
        quantity="acceleration",
        units="g",
    )


# ----------------------------------------------------------------------
# Construction / basic properties
# ----------------------------------------------------------------------


def test_from_channels_and_basic_properties():
    n = 500
    dt = 0.02
    ch1 = make_channel(n=n, dt=dt, freq=1.0, name_user="Acc1")
    ch2 = make_channel(n=n, dt=dt, freq=2.0, name_user="Acc2")

    test = Test.from_channels(
        name="MyTest",
        channels=[ch1, ch2],
        description="Example test",
    )

    assert isinstance(test, Test)
    assert test.name == "MyTest"
    assert test.description == "Example test"

    assert len(test) == 2
    assert test.n_channels == 2
    assert test.n_timesteps == n
    assert test.shape == (2, n)

    assert test.dt == pytest.approx(dt)
    expected_duration = (n - 1) * dt
    assert test.duration == pytest.approx(expected_duration)


def test_channel_index_and_name_lookup():
    ch1 = make_channel(name_user="Acc1", name_input="IN1")
    ch2 = make_channel(name_user=None, name_input="Acc2_IN")

    test = Test.from_channels(name="LookupTest", channels=[ch1, ch2])

    # Integer indexing
    assert test[0] is ch1
    assert test[1] is ch2

    # String lookup by name_user
    assert test["Acc1"] is ch1

    # String lookup by name_input
    assert test["Acc2_IN"] is ch2

    # channel_names helper
    names = test.channel_names()
    assert "Acc1" in names
    assert "Acc2_IN" in names


def test_iter_channels_and_tags_selection():
    ch1 = make_channel(name_user="AccDeck", tags={"acc", "deck"})
    ch2 = make_channel(name_user="AccAbut", tags={"acc", "abutment"})
    ch3 = make_channel(
        name_user="DispDeck", quantity="displacement", tags={"disp", "deck"}
    )

    test = Test.from_channels(name="TagTest", channels=[ch1, ch2, ch3])

    # Default selector -> all channels
    all_ch = list(test.iter_channels())
    assert all_ch == [ch1, ch2, ch3]

    # Index selector
    sel_idx = list(test.iter_channels(selector=1))
    assert sel_idx == [ch2]

    # Slice selector
    sel_slice = list(test.iter_channels(selector=slice(1, 3)))
    assert sel_slice == [ch2, ch3]

    # Tag-based selection (any of the tags)
    acc_ch = list(test.iter_channels(tags={"acc"}))
    assert acc_ch == [ch1, ch2]

    # Tag-based selection (require_all_tags=True)
    deck_ch = list(test.iter_channels(tags={"deck"}, require_all_tags=True))
    assert deck_ch == [ch1, ch3]


def test_dt_inconsistent_raises():
    ch1 = make_channel(dt=0.01)
    ch2 = make_channel(dt=0.02)
    test = Test.from_channels(name="BadDt", channels=[ch1, ch2])

    with pytest.raises(ValueError):
        _ = test.dt


def test_n_timesteps_and_duration_inconsistent_raises():
    dt = 0.01
    ch1 = make_channel(n=100, dt=dt)
    ch2 = make_channel(n=120, dt=dt)
    test = Test.from_channels(name="BadLen", channels=[ch1, ch2])

    with pytest.raises(ValueError):
        _ = test.n_timesteps

    with pytest.raises(ValueError):
        _ = test.duration


def test_info_returns_reasonable_string():
    ch1 = make_channel(name_user="Acc1", quantity="acceleration", units="g")
    ch2 = make_channel(name_user="Acc2", quantity="acceleration", units="g")
    test = Test.from_channels(name="InfoTest", channels=[ch1, ch2], description="Desc")

    info = test.info()
    assert isinstance(info, str)
    assert "Test: InfoTest" in info
    assert "Description" in info
    assert "Channels" in info
    # Table header sanity check
    assert "idx" in info
    assert "name" in info


# ----------------------------------------------------------------------
# CSV I/O
# ----------------------------------------------------------------------


def test_to_csv_and_from_csv_roundtrip(tmp_path):
    dt = 0.01
    ch1 = make_channel(n=200, dt=dt, name_user="Acc1")
    ch2 = make_channel(n=200, dt=dt, name_user="Acc2")

    test = Test.from_channels(name="CsvTest", channels=[ch1, ch2])

    csv_file = tmp_path / "data.csv"
    test.to_csv(str(csv_file))

    test2 = Test.from_csv(str(csv_file), name="CsvTestReloaded")

    assert test2.n_channels == 2
    assert test2.n_timesteps == 200
    assert test2.dt == pytest.approx(dt)

    # Check data content roughly matches
    t1, y1 = test[0].xy()
    t2, y2 = test2[0].xy()
    assert np.allclose(t1, t2)
    assert np.allclose(y1, y2)


# ----------------------------------------------------------------------
# Spectral analyses and transfer functions
# ----------------------------------------------------------------------


def test_cross_spectrum_shapes():
    dt = 0.01
    n = 1024
    ch_x = make_channel(n=n, dt=dt, freq=5.0, name_user="In")
    # Output as scaled + noise
    t = np.arange(n) * dt
    y = 2.0 * np.sin(2.0 * np.pi * 5.0 * t) + 0.1 * np.random.randn(n)
    ch_y = Channel(data=y, dt=dt, name_user="Out")

    test = Test.from_channels(name="SpecTest", channels=[ch_x, ch_y])

    f, Pxy = test.cross_spectrum("In", "Out")
    assert f.ndim == 1
    assert Pxy.shape == f.shape
    assert Pxy.dtype == np.complex128 or np.iscomplexobj(Pxy)


def test_transfer_function_shapes_and_kind_switch():
    dt = 0.01
    n = 1024
    ch_x = make_channel(n=n, dt=dt, freq=3.0, name_user="In")
    t = np.arange(n) * dt
    y = 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    ch_y = Channel(data=y, dt=dt, name_user="Out")

    test = Test.from_channels(name="HTest", channels=[ch_x, ch_y])

    f_h1, H1 = test.transfer_function("In", "Out", kind="H1")
    f_h2, H2 = test.transfer_function("In", "Out", kind="H2")

    assert f_h1.shape == H1.shape
    assert f_h2.shape == H2.shape
    assert np.allclose(f_h1, f_h2)

    with pytest.raises(ValueError):
        test.transfer_function("In", "Out", kind="invalid")


def test_plot_transfer_function_smoke():
    """Smoke tests for magnitude-only and magnitude+phase transfer function plots."""
    plt.switch_backend("Agg")

    dt = 0.01
    n = 1024
    ch_x = make_channel(n=n, dt=dt, freq=3.0, name_user="In")
    t = np.arange(n) * dt
    y = 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    ch_y = Channel(data=y, dt=dt, name_user="Out")

    test = Test.from_channels(name="HPlotTest", channels=[ch_x, ch_y])

    # Magnitude-only plot
    ax_mag = test.plot_transfer_function("In", "Out")
    assert isinstance(ax_mag, plt.Axes)
    assert len(ax_mag.lines) >= 1

    # Magnitude + phase plot: expect a twinned second axis
    ax_mag_phase = test.plot_transfer_function("In", "Out", phase=True)
    fig = ax_mag_phase.figure
    # One main axis + one twin axis
    assert len(fig.axes) == 2

    plt.close(fig)


def test_time_delay_estimation_sign_and_magnitude():
    dt = 0.01
    n = 1024
    t = np.arange(n) * dt
    x = np.sin(2.0 * np.pi * 2.0 * t)
    # Create a delayed version: shift by k samples
    k = 5
    y = np.roll(x, k)

    ch_x = Channel(data=x, dt=dt, name_user="In")
    ch_y = Channel(data=y, dt=dt, name_user="Out")

    test = Test.from_channels(name="DelayTest", channels=[ch_x, ch_y])

    tau = test.time_delay("In", "Out")
    # Only check magnitude; sign depends on convention
    assert abs(tau) == pytest.approx(k * dt, rel=1e-1)


# ----------------------------------------------------------------------
# Arias-based trimming
# ----------------------------------------------------------------------


def test_trimmed_by_arias_reduces_duration():
    dt = 0.01
    n = 2000
    # Strong burst between t≈2s and t≈4s
    t = np.arange(n) * dt
    data = np.zeros_like(t)
    mask = (t >= 2.0) & (t <= 4.0)
    data[mask] = np.sin(2.0 * np.pi * 5.0 * t[mask])

    ch1 = Channel(
        data=data, dt=dt, name_user="Acc1", quantity="acceleration", units="g"
    )
    ch2 = make_channel(
        n=n, dt=dt, freq=1.0, name_user="Acc2", quantity="acceleration", units="g"
    )

    test = Test.from_channels(name="AriasTest", channels=[ch1, ch2])

    original_duration = test.duration
    trimmed = test.trimmed_by_arias(ref="Acc1", lower=0.05, upper=0.95)

    assert isinstance(trimmed, Test)
    assert trimmed.n_channels == test.n_channels
    assert trimmed.duration < original_duration


# ----------------------------------------------------------------------
# Plotting helpers on Test
# ----------------------------------------------------------------------


def test_plot_channels_and_grid_smoke():
    """Smoke tests for Test.plot_channels and Test.plot_grid."""
    plt.switch_backend("Agg")

    ch1 = make_channel(name_user="Acc1")
    ch2 = make_channel(name_user="Acc2")

    test = Test.from_channels(name="PlotTest", channels=[ch1, ch2])

    # plot_channels smoke
    fig1, axes1 = test.plot_channels()
    assert fig1 is not None
    assert axes1 is not None

    # plot_grid smoke
    fig2, axes2 = test.plot_grid()
    assert fig2 is not None
    assert axes2 is not None

    plt.close(fig1)
    plt.close(fig2)


# ----------------------------------------------------------------------
# Channel health report
# ----------------------------------------------------------------------


def test_channel_health_basic():
    """Basic sanity checks for the channel_health report."""
    # Dead channel (all zeros)
    dead = make_ch(np.zeros(1000), name="Dead")

    # Noise channel: small random values (no clear burst)
    rng = np.random.default_rng(0)
    noise = make_ch(0.001 * rng.standard_normal(2000), name="Noise")

    # Good channel: clear event from t=2–4 s
    dt = 0.01
    t = np.arange(2000) * dt
    y = 0.001 * rng.standard_normal(2000)
    mask = (t >= 2.0) & (t <= 4.0)
    y[mask] += np.sin(2 * np.pi * 5 * t[mask])
    good = make_ch(y, name="Good")

    test = Test.from_channels(name="HealthTest", channels=[dead, noise, good])

    report = test.channel_health()

    # Basic checks
    assert isinstance(report, str)
    assert "Dead" in report
    assert "Noise" in report
    assert "Good" in report

    # Status keywords
    lower = report.lower()
    assert "dead" in lower
    assert "no event" in lower or "noise" in lower
    assert "ok" in lower or "weak" in lower or "response" in lower


# ----------------------------------------------------------------------
# Channel metadata CSV import/export
# ----------------------------------------------------------------------


def test_channel_info_to_csv_removes_redundant(tmp_path):
    ch1 = make_channel(n=20, dt=0.01, name_user="A", name_input="A", units="g")
    ch2 = make_channel(n=20, dt=0.01, name_user="A", name_input="ChA", units="g")

    test = Test.from_channels(name="T", channels=[ch1, ch2])

    csv_path = tmp_path / "meta.csv"
    test.channel_info_to_csv(str(csv_path))

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row1 = next(reader)
        row2 = next(reader)

    # Only the appropriate columns should remain after removing redundant fields
    assert header == [
        "idx",
        "name_user",
        "name_input",
        "units",
        "label_axis",
        "calibration_factor",
    ]
    assert row1 == ["0", "", "A", "g", "A [g]", "1.0"]
    assert row2 == ["1", "A", "ChA", "g", "A [g]", "1.0"]


def test_import_csv_raises_on_unmatched_row(tmp_path):
    ch1 = make_channel(n=20, dt=0.01, name_user="Acc1")
    test = Test.from_channels(name="X", channels=[ch1])

    csv_path = tmp_path / "bad.csv"

    # CSV row referencing unknown channel
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name_user", "units"])
        writer.writerow(["NonexistentChannel", "g"])  # must trigger error

    with pytest.raises(ValueError):
        test.with_channel_info_from_csv(str(csv_path))
