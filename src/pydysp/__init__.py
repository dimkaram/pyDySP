# __init__.py
from .channel import Channel
from .test import Test
from .campaign import Campaign
from .spectra import FourierSpectrum, WelchSpectrum
from .arias import AriasResult
from .response import ResponseSpectrum
from .downsample import downsample
from .helpers import (
    annotate_peak,
    annotate_time_peak,
    annotate_fourier_peak,
    annotate_psd_peak,
    annotate_response_peak,
)

__all__ = [
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
