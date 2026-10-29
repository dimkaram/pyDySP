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
