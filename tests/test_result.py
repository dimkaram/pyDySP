import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pydysp import CampaignResult
from pydysp.campaign import Campaign
from pydysp.channel import Channel
from pydysp.test import Test

# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------


def make_test(name, amp=1.0, offset=0.0, channels=("A1", "L1"), meta=None):
    """Test with a sine of amplitude ``amp`` on A1 and a step to ``offset`` on L1."""
    n, dt = 400, 0.01
    t = np.arange(n) * dt
    data = {
        "A1": amp * np.sin(2 * np.pi * 2.0 * t),
        "L1": np.where(t > 2.0, offset, 0.0),
    }
    units = {"A1": "g", "L1": "mm"}
    chans = [
        Channel(data=data[c], dt=dt, name_user=c, units=units[c]) for c in channels
    ]
    return Test(name=name, channels=chans, meta=meta or {})


@pytest.fixture
def camp():
    tests = [
        make_test("T29", amp=1.0, offset=0.5, meta={"amp": 0.06}),
        make_test("T30", amp=2.0, offset=-0.2, meta={"amp": 0.1}),
        make_test("T32", amp=3.0, offset=0.3, meta={"amp": 0.2}),
        make_test("T33", amp=4.0, offset=0.1, meta={}),
    ]
    return Campaign.from_tests(tests, numbers=[29, 30, 32, 33], name="C")


def make_result(values, numbers=None, **kw):
    numbers = numbers or list(range(1, len(values) + 1))
    return CampaignResult(
        values=values, numbers=numbers, names=[f"T{n}" for n in numbers], **kw
    )


# ----------------------------------------------------------------------
# metric
# ----------------------------------------------------------------------


def test_metric_builtins(camp):
    for what, expected in [
        ("max", [1, 2, 3, 4]),
        ("min", [-1, -2, -3, -4]),
        ("rms", np.array([1, 2, 3, 4]) / np.sqrt(2)),
    ]:
        res = camp.metric("A1", what)
        np.testing.assert_allclose(res.values, expected, rtol=1e-2)
        assert res.what == what
    peak = camp.metric("A1", "max_abs")
    np.testing.assert_allclose(np.abs(peak.values), [1, 2, 3, 4], rtol=1e-2)
    # Same value Channel.max_abs reports, without the time
    assert peak.values[0] == camp[0]["A1"].max_abs()[1]


def test_metric_result_contents(camp):
    res = camp.metric("A1", "max")
    assert res.numbers == [29, 30, 32, 33]
    assert res.names == ["T29", "T30", "T32", "T33"]
    assert res.label == "A1"
    assert res.units == "g"
    assert res.axis_label == "A1 [g]"
    assert res.value(32) == pytest.approx(3.0, rel=1e-2)
    with pytest.raises(KeyError):
        res.value(31)
    assert len(res) == 4


def test_metric_residual(camp):
    res = camp.metric("L1", "residual")
    np.testing.assert_allclose(res.values, [0.5, -0.2, 0.3, 0.1])
    # Window reaching back before the step mixes in zeros
    res2 = camp.metric("L1", "residual", n_last=300)
    np.testing.assert_allclose(res2.values, np.array([0.5, -0.2, 0.3, 0.1]) * 199 / 300)
    with pytest.raises(ValueError, match="n_last"):
        camp.metric("L1", "residual", n_last=0)


def test_metric_custom_function_and_overrides(camp):
    def peak_to_peak(ch, scale=1.0):
        _, y = ch.processed()
        return scale * (y.max() - y.min())

    res = camp.metric("A1", peak_to_peak, scale=0.5, label="A1 p2p/2")
    np.testing.assert_allclose(res.values, [1, 2, 3, 4], rtol=1e-2)
    assert res.what == "peak_to_peak"
    assert res.label == "A1 p2p/2"
    lam = camp.metric("A1", lambda ch: 1, units="-")
    assert lam.what == "" and lam.units == "-"
    with pytest.raises(TypeError, match="expected a number"):
        camp.metric("A1", lambda ch: "x")


def test_metric_unknown_name(camp):
    with pytest.raises(ValueError, match="Unknown metric"):
        camp.metric("A1", "median")
    with pytest.raises(TypeError):
        camp.metric("A1", 3)


def test_metric_missing_channel_nan_and_strict():
    tests = [
        make_test("a"),
        make_test("b", channels=("L1",)),
        make_test("c", amp=3.0),
    ]
    camp = Campaign.from_tests(tests, numbers=[1, 2, 3])
    with pytest.warns(UserWarning, match="'A1' missing in tests 2"):
        res = camp.metric("A1", "max")
    assert np.isnan(res.values[1])
    assert res.values[2] == pytest.approx(3.0, rel=1e-2)
    with pytest.raises(KeyError, match="test 2"):
        camp.metric("A1", "max", strict=True)
    # Label falls back to the requested name if no test has the channel
    with pytest.warns(UserWarning):
        none = camp.select(numbers=[2]).metric("A1", "max")
    assert none.label == "A1" and none.units is None


def test_metric_no_warning_when_complete(camp):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        camp.metric("A1", "rms")


# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------


def test_abs_neg_and_original_unchanged():
    r = make_result([1.0, -2.0, np.nan], label="L1", units="mm")
    np.testing.assert_array_equal(r.abs().values, [1, 2, np.nan])
    np.testing.assert_array_equal(abs(r).values, [1, 2, np.nan])
    np.testing.assert_array_equal((-r).values, [-1, 2, np.nan])
    assert r.values[1] == -2.0
    assert (-r).label == "L1" and (-r).units == "mm"


def test_cumsum_skips_nan():
    r = make_result([1.0, np.nan, 2.0, 3.0])
    np.testing.assert_array_equal(r.cumsum().values, [1, np.nan, 3, 6])


def test_accumulated_residual_chain(camp):
    acc = camp.metric("L1", "residual").abs().cumsum()
    np.testing.assert_allclose(acc.values, [0.5, 0.7, 1.0, 1.1])
    assert acc.numbers == [29, 30, 32, 33]


def test_arithmetic_with_numbers():
    r = make_result([1.0, 2.0], units="g")
    np.testing.assert_array_equal((r * 9.81).values, [9.81, 19.62])
    np.testing.assert_array_equal((2 * r).values, [2, 4])
    np.testing.assert_array_equal((r + 1).values, [2, 3])
    np.testing.assert_array_equal((1 - r).values, [0, -1])
    np.testing.assert_array_equal((r / 2).values, [0.5, 1])
    np.testing.assert_array_equal((2 / r).values, [2, 1])
    assert (r * 2).units == "g"


def test_arithmetic_between_results():
    f = make_result([10.0, 20.0, np.nan], label="F1", units="kN")
    d = make_result([2.0, 0.0, 1.0], label="L1", units="mm")
    k = f / d
    np.testing.assert_array_equal(k.values, [5.0, np.inf, np.nan])
    assert k.label == "F1 / L1"
    assert k.units == "kN/mm"
    assert (f * d).units == "kN·mm"
    assert (f + f).units == "kN"
    assert (f - d).units is None
    other = make_result([1.0, 2.0, 3.0], numbers=[1, 2, 4])
    with pytest.raises(ValueError, match="same test numbers"):
        f + other
    with pytest.raises(TypeError):
        f + "x"


def test_result_validation():
    with pytest.raises(ValueError, match="numbers has"):
        CampaignResult(values=[1.0, 2.0], numbers=[1], names=["a", "b"])
    with pytest.raises(ValueError, match="one-dimensional"):
        CampaignResult(values=[[1.0]], numbers=[1], names=["a"])


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------


def test_to_array_is_copy():
    r = make_result([1.0, 2.0])
    arr = r.to_array()
    arr[0] = 99
    assert r.values[0] == 1.0


def test_to_table_and_csv(tmp_path):
    r = make_result([1.5, np.nan], numbers=[3, 4], label="A1", units="g")
    table = r.to_table()
    assert "A1 [g]" in table and "T3" in table and "nan" in table
    out = tmp_path / "a.csv"
    r.to_csv(str(out))
    with open(out, newline="") as f:
        rows = list(csv.reader(f))
    assert rows == [["number", "name", "A1 [g]"], ["3", "T3", "1.5"], ["4", "T4", ""]]
    with pytest.raises(FileExistsError):
        r.to_csv(str(out), overwrite=False)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------


def test_plot_default_returns_fig_ax_and_labels(camp):
    res = camp.metric("A1", "max")
    fig, ax = res.plot()
    assert fig is ax.figure
    assert ax.get_xlabel() == "Test number"
    assert ax.get_ylabel() == "A1 [g]"
    plt.close(fig)


def test_plot_gaps_in_numbers(camp):
    res = camp.metric("A1", "max")  # numbers 29, 30, 32, 33
    fig, ax = res.plot()
    x = ax.lines[0].get_xdata()
    assert np.isnan(x[2])  # break between 30 and 32
    np.testing.assert_array_equal(x[[0, 1, 3, 4]], [29, 30, 32, 33])
    plt.close(fig)
    fig, ax = res.plot(gaps=False)
    assert not np.isnan(ax.lines[0].get_xdata()).any()
    plt.close(fig)


def test_plot_into_given_axes_and_overrides(camp):
    fig, axes = plt.subplots(2, 2)
    res = camp.metric("A1", "max")
    fig2, ax = res.plot(ax=axes[0, 1], ylabel="Peak [g]", xlabel="Test", color="k")
    assert fig2 is fig and ax is axes[0, 1]
    assert ax.get_ylabel() == "Peak [g]" and ax.get_xlabel() == "Test"
    assert ax.lines[0].get_color() == "k"
    # No styling applied to the figure
    assert tuple(fig.get_size_inches()) == tuple(plt.rcParams["figure.figsize"])
    plt.close(fig)


def test_plot_highlight(camp):
    res = camp.metric("A1", "max")
    fig, ax = res.plot(highlight=32)
    assert len(ax.lines) == 2
    hl = ax.lines[1]
    assert hl.get_marker() == "s"
    np.testing.assert_array_equal(hl.get_xdata(), [32])
    plt.close(fig)
    fig, ax = res.plot(highlight=[29, 33])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [29, 33])
    plt.close(fig)
    with pytest.raises(KeyError):
        res.plot(highlight=31)


def test_plot_against_other_result_with_origin(camp):
    f = camp.metric("A1", "max")
    d = camp.metric("L1", "residual")
    fig, ax = f.plot(x=d, origin=True, highlight=30)
    x = ax.lines[0].get_xdata()
    y = ax.lines[0].get_ydata()
    assert x[0] == 0 and y[0] == 0
    np.testing.assert_allclose(x[1:], d.values)
    assert not np.isnan(x).any()  # no gap breaks against another result
    assert ax.get_xlabel() == "L1 [mm]"
    np.testing.assert_allclose(ax.lines[1].get_xdata(), [d.value(30)])
    plt.close(fig)
    other = camp.select(numbers=[29, 30]).metric("L1", "residual")
    with pytest.raises(ValueError, match="same test numbers"):
        f.plot(x=other)


def test_plot_against_meta_key(camp):
    res = camp.metric("A1", "max")
    fig, ax = res.plot(x="amp")
    x = ax.lines[0].get_xdata()
    np.testing.assert_allclose(x[:3], [0.06, 0.1, 0.2])
    assert np.isnan(x[3])  # test 33 has no "amp"
    assert ax.get_xlabel() == "amp"
    plt.close(fig)
    with pytest.raises(TypeError):
        res.plot(x=3)


def test_plot_nan_values_not_drawn():
    r = make_result([1.0, np.nan, 3.0])
    fig, ax = r.plot()
    assert np.isnan(ax.lines[0].get_ydata()[1])
    plt.close(fig)
