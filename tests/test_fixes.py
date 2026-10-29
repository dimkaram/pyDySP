import numpy as np

from pydysp.channel import Channel


def make_sine(n: int = 500, dt: float = 0.01) -> Channel:
    t = np.arange(n) * dt
    return Channel(data=np.sin(2 * np.pi * 2.0 * t), dt=dt, name_user="Acc", units="g")


# ----------------------------------------------------------------------
# Channel.response_spectrum: default periods are not shared with results
# ----------------------------------------------------------------------


def test_response_spectrum_result_does_not_alias_default_periods():
    ch = make_sine()
    first = ch.response_spectrum()
    expected = np.linspace(0.05, 5.0, 100)
    np.testing.assert_array_equal(first.T, expected)
    first.T[:] = 99.0  # edit the returned periods in place
    second = ch.response_spectrum()
    np.testing.assert_array_equal(second.T, expected)
    assert second.T is not first.T


def test_response_spectrum_result_does_not_alias_user_periods():
    ch = make_sine()
    periods = np.array([0.1, 0.5, 1.0])
    rs = ch.response_spectrum(periods=periods)
    rs.T[0] = 42.0
    np.testing.assert_array_equal(periods, [0.1, 0.5, 1.0])


# ----------------------------------------------------------------------
# Test.info: channel-table columns match their headers
# ----------------------------------------------------------------------


def test_test_info_channel_table_columns_match_headers():
    from pydysp.test import Test

    ch = Channel(
        data=np.zeros(10), dt=0.1, name_user="Acc1", name_input="DAQ_07", units="g"
    )
    info = Test(name="T", channels=[ch]).info()
    lines = [ln for ln in info.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    row = [c.strip() for c in lines[2].strip("|").split("|")]
    columns = dict(zip(header, row))
    assert columns["name_user"] == "Acc1"
    assert columns["name_input"] == "DAQ_07"
    assert columns["units"] == "g"
