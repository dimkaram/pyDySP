import numpy as np
import scipy.io as sio

from pydysp.downsample import downsample


def test_downsample_basic(tmp_path):
    """Verify that numeric arrays are downsampled and non-arrays preserved."""
    file_in = tmp_path / "in.mat"
    file_out = tmp_path / "out.mat"

    data = {
        # Column vector stored as (n,1)
        "vec": np.arange(10.0).reshape(-1, 1),
        # 2D matrix to downsample along axis 0
        "mat": np.arange(24.0).reshape(6, 4),
        # Non-array: should remain untouched
        "skip": [1, 2, 3],
    }

    sio.savemat(file_in, data)
    downsample(str(file_in), str(file_out), factor=2)
    out = sio.loadmat(file_out)

    assert out["vec"].shape == (5, 1)
    assert out["mat"].shape == (3, 4)
    assert "skip" in out


def test_downsample_bad_factor(tmp_path):
    """Invalid downsample factors (e.g., zero) should raise."""
    file_in = tmp_path / "in.mat"
    sio.savemat(file_in, {"x": np.arange(10.0)})
    file_out = tmp_path / "out.mat"

    import pytest

    with pytest.raises(ValueError):
        downsample(str(file_in), str(file_out), factor=0)
