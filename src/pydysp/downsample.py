# downsample.py
import warnings

import scipy.io as sio


def downsample(file_in: str, file_out: str, factor: int = 20) -> None:
    """
    Downsample array-like variables in a MATLAB ``.mat`` file.

    This function loads a ``.mat`` file, downsamples any array-like
    variables (NumPy arrays, MATLAB arrays, or objects exposing a ``shape``
    attribute with ``ndim >= 1``), and writes the result to a new ``.mat``
    file. Non-array containers such as Python ``list`` or ``tuple`` are left
    unchanged.

    Parameters
    ----------
    file_in : str
        Path to the input MATLAB ``.mat`` file.
    file_out : str
        Path to the output MATLAB ``.mat`` file where the downsampled
        variables will be saved.
    factor : int, optional
        Downsampling factor applied along the first dimension. Must be a
        positive integer. Default is ``20``.

    Raises
    ------
    FileNotFoundError
        If the input file cannot be found or opened.
    ValueError
        If ``factor`` is not a positive integer.

    Notes
    -----
    * Only objects exposing ``shape`` and having at least one dimension are
      downsampled. The operation is performed using slicing ``val[::factor]``.
    * Keys such as ``__header__``, ``__version__`` and ``__globals__`` are
      preserved and written unchanged.
    * SciPy warnings related to MATLAB struct fields beginning with an
      underscore are suppressed during saving.
    """
    if not isinstance(factor, int) or factor <= 0:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")

    try:
        imported_data = sio.loadmat(file_in)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(
            f"Input file not found or unreadable: {file_in}"
        ) from exc

    # Downsample only numeric/array-like objects (ndim >= 1); leave Python containers unchanged.
    for key, val in imported_data.items():
        if isinstance(val, (list, tuple)):
            continue
        if hasattr(val, "shape") and getattr(val, "ndim", 0) >= 1:
            imported_data[key] = val[::factor]

    # Suppress SciPy warnings about field names starting with an underscore during saving.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Starting field name with a underscore",
        )
        sio.savemat(file_out, imported_data)
