# __init__.py
from .arias import AriasResult
from .campaign import Campaign
from .channel import Channel
from .downsample import downsample
from .helpers import (
    annotate_fourier_peak,
    annotate_peak,
    annotate_psd_peak,
    annotate_response_peak,
    annotate_time_peak,
)
from .response import ResponseSpectrum
from .spectra import FourierSpectrum, WelchSpectrum
from .test import Test

__all__ = [  # noqa: RUF022 - grouped by category, not alphabetical
    # Main classes
    "Channel",
    "Test",
    "Campaign",
    # Additional classes
    "FourierSpectrum",
    "WelchSpectrum",
    "AriasResult",
    "ResponseSpectrum",
    # Helper functions
    "annotate_peak",
    "annotate_time_peak",
    "annotate_fourier_peak",
    "annotate_psd_peak",
    "annotate_response_peak",
    "downsample",
]
__version__ = "0.2.2"
