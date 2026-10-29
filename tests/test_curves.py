import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.signal import iirpeak, lfilter

from pydysp import CampaignResult
from pydysp.campaign import Campaign
from pydysp.channel import Channel
from pydysp.test import Test

FS = 200.0
N = 8192

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def resonant_test(name, fn, seed, n=N, fs=FS, channels=("In", "Out")):
    """White-noise input and a response with a resonance at ``fn`` Hz."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    b, a = iirpeak(fn, Q=8.0, fs=fs)
    y = lfilter(b, a, x) * 3.0
    data = {"In": x, "Out": y}
    chans = [
        Channel(data=data[c], dt=1.0 / fs, name_user=c, units="g") for c in channels
    ]
    return Test(name=name, channels=chans, meta={"fn": fn})


@pytest.fixture
def camp():
    # Natural frequency drops through the campaign (damage); test 4 skipped
    fns = {1: 20.0, 2: 19.0, 3: 17.5, 5: 15.0, 6: 14.0}
    tests = [resonant_test(f"T{n}", fn, seed=n) for n, fn in fns.items()]
    return Campaign.from_tests(tests, numbers=list(fns), name="C")


# ----------------------------------------------------------------------
# transfer_function / spectrum
# ----------------------------------------------------------------------


def test_transfer_function_curves(camp):
    tf = camp.transfer_function("In", "Out", nperseg=1024)
    assert tf.is_curve
    assert tf.numbers == [1, 2, 3, 5, 6]
    assert tf.label == "Out/In"
    assert tf.units == "g/g"
    assert tf.what == "H1"
    f, h = tf.curve(3)
    assert np.iscomplexobj(h)
    # Matches Test.transfer_function for the same test
    f_ref, h_ref = camp.get(number=3).transfer_function("In", "Out", nperseg=1024)
    np.testing.assert_array_equal(f, f_ref)
    np.testing.assert_array_equal(h, h_ref)
    assert "curve" in repr(tf)


def test_transfer_function_tests_argument(camp):
    tf = camp.transfer_function("In", "Out", tests=[6, 1], nperseg=512)
    assert tf.numbers == [1, 6]  # campaign order
    with pytest.raises(KeyError, match="4"):
        camp.transfer_function("In", "Out", tests=[1, 4])


def test_transfer_function_h2_and_missing_channel():
    tests = [
        resonant_test("a", 20.0, 1),
        resonant_test("b", 18.0, 2, channels=("In",)),
        resonant_test("c", 16.0, 3),
    ]
    camp = Campaign.from_tests(tests)
    with pytest.warns(UserWarning, match="'Out' missing in tests 2"):
        tf = camp.transfer_function("In", "Out", kind="H2", nperseg=512)
    assert tf.what == "H2"
    assert tf.curve(2) is None
    assert tf.curve(1) is not None
    with pytest.raises(KeyError, match="test 2"):
        camp.transfer_function("In", "Out", strict=True)
    fp = tf.peak_frequency()
    assert np.isnan(fp.value(2)) and not np.isnan(fp.value(1))


def test_spectrum_welch_and_fourier(camp):
    psd = camp.spectrum("Out", nperseg=1024)
    assert psd.is_curve and psd.what == "psd"
    assert psd.units == "g²/Hz"
    assert psd.label == "Out"
    _f, p = psd.curve(1)
    ref = camp[0]["Out"].welch_psd(nperseg=1024)
    np.testing.assert_array_equal(p, ref.p)
    four = camp.spectrum("Out", method="fourier", tests=[1, 2])
    assert four.what == "fourier" and four.units == "g"
    np.testing.assert_array_equal(four.curve(2)[1], camp[1]["Out"].fourier().s)
    with pytest.raises(ValueError, match="method"):
        camp.spectrum("Out", method="fft")
    with pytest.raises(TypeError):
        camp.spectrum("Out", method="fourier", nperseg=10)


# ----------------------------------------------------------------------
# peak_frequency
# ----------------------------------------------------------------------


def test_peak_frequency_tracks_resonance(camp):
    tf = camp.transfer_function("In", "Out", nperseg=2048)
    fp = tf.peak_frequency(fmin=5, fmax=40)
    assert not fp.is_curve
    assert fp.units == "Hz" and fp.what == "peak_frequency"
    assert fp.numbers == [1, 2, 3, 5, 6]
    df = FS / 2048
    np.testing.assert_allclose(fp.values, camp.meta_values("fn"), atol=3 * df)
    # A scalar result like any other: plot against test number
    ax = fp.plot()
    assert ax.get_ylabel() == "Peak frequency Out/In [Hz]"
    plt.close("all")


def test_peak_frequency_band_and_errors(camp):
    psd = camp.spectrum("Out", nperseg=1024)
    # Band that excludes the resonance picks the band edge nearest to it
    fp = psd.peak_frequency(fmin=30, fmax=60)
    assert np.all(fp.values >= 30) and np.all(fp.values <= 60)
    with pytest.raises(ValueError, match="no frequencies"):
        psd.peak_frequency(fmin=500, fmax=600)
    with pytest.raises(TypeError, match="curve results"):
        camp.metric("Out", "max").peak_frequency()


# ----------------------------------------------------------------------
# Transforms and access on curves
# ----------------------------------------------------------------------


def test_curve_transforms(camp):
    tf = camp.transfer_function("In", "Out", tests=[1, 2], nperseg=512)
    mag = tf.abs()
    assert not np.iscomplexobj(mag.curve(1)[1])
    np.testing.assert_allclose(mag.curve(1)[1], np.abs(tf.curve(1)[1]))
    np.testing.assert_allclose((tf * 2).curve(2)[1], 2 * tf.curve(2)[1])
    np.testing.assert_allclose((-tf).curve(2)[1], -tf.curve(2)[1])
    with pytest.raises(TypeError, match="scalar results"):
        tf + tf
    with pytest.raises(TypeError, match="scalar results"):
        tf.cumsum()
    with pytest.raises(TypeError, match="curve"):
        tf.value(1)
    with pytest.raises(TypeError, match="use value"):
        camp.metric("Out", "max").curve(1)
    with pytest.raises(TypeError, match="to_csv"):
        tf.to_table()


def test_curve_validation():
    with pytest.raises(ValueError, match="same length"):
        CampaignResult(values=[np.ones(3)], freq=[np.ones(4)], numbers=[1], names=["a"])
    with pytest.raises(ValueError, match="both"):
        CampaignResult(values=[None], freq=[np.ones(4)], numbers=[1], names=["a"])
    with pytest.raises(ValueError, match="freq has"):
        CampaignResult(values=[np.ones(3)], freq=[], numbers=[1], names=["a"])


# ----------------------------------------------------------------------
# Common grid, to_array and to_csv
# ----------------------------------------------------------------------


def test_common_grid_and_to_array(camp):
    tf = camp.transfer_function("In", "Out", nperseg=1024)
    f = tf.common_freq()
    arr = tf.to_array()
    assert arr.shape == (5, len(f))
    assert np.iscomplexobj(arr)


def test_different_grids_raise_clear_error():
    tests = [resonant_test("a", 20.0, 1), resonant_test("b", 20.0, 2, n=3000)]
    camp = Campaign.from_tests(tests)
    four = camp.spectrum("Out", method="fourier")
    # Overlay works with different grids
    ax = four.plot()
    assert len(ax.lines) == 2
    plt.close("all")
    with pytest.raises(ValueError, match="do not share one frequency grid"):
        four.plot(kind="heatmap")
    with pytest.raises(ValueError, match="nperseg"):
        four.to_csv("unused.csv")
    with pytest.raises(ValueError, match="frequency grid"):
        four.to_array()


def test_curve_to_csv(tmp_path, camp):
    tf = camp.transfer_function("In", "Out", tests=[1, 3], nperseg=256)
    out = tmp_path / "tf.csv"
    tf.to_csv(str(out))
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["frequency [Hz]", "Test 1", "Test 3"]
    f_ref, h_ref = tf.curve(3)
    assert len(rows) == 1 + len(f_ref)
    assert float(rows[5][0]) == f_ref[4]
    assert float(rows[5][2]) == pytest.approx(abs(h_ref[4]))  # magnitude default
    tf.to_csv(str(out), part="phase")
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert float(rows[5][2]) == pytest.approx(np.degrees(np.angle(h_ref[4])))
    with pytest.raises(ValueError, match="part"):
        tf.to_csv(str(out), part="mag")
    with pytest.raises(FileExistsError):
        tf.to_csv(str(out), overwrite=False)


def test_curve_to_csv_missing_test_is_empty_column(tmp_path):
    tests = [resonant_test("a", 20.0, 1), resonant_test("b", 18, 2, channels=("In",))]
    camp = Campaign.from_tests(tests)
    with pytest.warns(UserWarning):
        psd = camp.spectrum("Out", nperseg=256)
    out = tmp_path / "psd.csv"
    psd.to_csv(str(out))
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["frequency [Hz]", "Test 1", "Test 2"]
    assert all(r[2] == "" for r in rows[1:])
    assert rows[3][1] != ""


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------


def test_overlay_plot(camp):
    tf = camp.transfer_function("In", "Out", nperseg=1024)
    ax = tf.plot(fmax=50)
    assert isinstance(ax, plt.Axes)
    labels = [ln.get_label() for ln in ax.lines]
    assert labels == ["Test 1", "Test 2", "Test 3", "Test 5", "Test 6"]
    assert ax.get_legend() is not None
    assert ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_ylabel() == "|Out/In| [g/g]"
    assert ax.lines[0].get_xdata().max() <= 50
    assert np.all(ax.lines[0].get_ydata() >= 0)  # magnitude
    plt.close("all")


def test_overlay_into_given_axes_and_options(camp):
    fig, axes = plt.subplots(1, 2)
    tf = camp.transfer_function("In", "Out", tests=[1, 2], nperseg=512)
    ax = tf.plot(ax=axes[1], part="phase", legend=False, fmin=1, color="k")
    assert ax is axes[1] and ax.figure is fig
    assert ax.get_legend() is None
    assert ax.get_ylabel() == "Phase Out/In [deg]"
    assert ax.lines[0].get_xdata().min() >= 1
    assert ax.lines[0].get_color() == "k"
    psd = camp.spectrum("Out", nperseg=512)
    ax2 = psd.plot(ax=axes[0])
    assert ax2.get_ylabel() == "Out [g²/Hz]"
    plt.close("all")


def test_heatmap(camp):
    tf = camp.transfer_function("In", "Out", nperseg=1024)
    fig, axes = plt.subplots(1, 2)
    ax = tf.plot(kind="heatmap", ax=axes[0], fmax=50)
    assert ax is axes[0]
    mesh = ax.collections[0]
    data = mesh.get_array()
    # One stripe per test (rows), frequencies up to fmax (columns)
    n_cols = int(np.sum(tf.common_freq() <= 50))
    assert data.shape == (5, n_cols)
    assert mesh.colorbar is not None
    assert mesh.colorbar.ax.get_ylabel() == "|Out/In| [g/g]"
    assert ax.get_ylabel() == "Test number"
    # Rows are labelled with real test numbers (4 is skipped, no empty row)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert set(labels) <= {"1", "2", "3", "5", "6"}
    assert "5" in labels or "6" in labels
    plt.close("all")
    ax = tf.plot(kind="heatmap", colorbar=False)
    assert ax.collections[0].colorbar is None
    plt.close("all")


def test_heatmap_missing_test_row_is_masked():
    tests = [
        resonant_test("a", 20.0, 1),
        resonant_test("b", 18, 2, channels=("In",)),
        resonant_test("c", 16, 3),
    ]
    with pytest.warns(UserWarning):
        psd = Campaign.from_tests(tests).spectrum("Out", nperseg=256)
    ax = psd.plot(kind="heatmap")
    data = ax.collections[0].get_array()
    assert data.mask[1].all() and not data.mask[0].any()
    plt.close("all")


def test_plot_option_errors(camp):
    tf = camp.transfer_function("In", "Out", tests=[1], nperseg=256)
    with pytest.raises(ValueError, match="kind"):
        tf.plot(kind="waterfall")
    with pytest.raises(TypeError, match="scalar results only"):
        tf.plot(highlight=1)
    scalar = camp.metric("Out", "max")
    with pytest.raises(TypeError, match="curve results only"):
        scalar.plot(kind="heatmap")
    with pytest.raises(TypeError):
        scalar.plot(x=tf)


def test_no_warnings_for_complete_campaign(camp):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tf = camp.transfer_function("In", "Out", nperseg=512)
        tf.plot()
        tf.plot(kind="heatmap")
        tf.peak_frequency()
    plt.close("all")
