"""
Core data model for DAS Explorer.

Defines the DASDataset container used throughout the application to
represent a single loaded DAS acquisition: the strain/acoustic matrix
plus its spatial/temporal axes and metadata.
"""

from dataclasses import dataclass, field
from typing import Optional
import datetime
import numpy as np


@dataclass
class DASDataset:
    """
    Container for a DAS dataset.

    Attributes
    ----------
    tr : np.ndarray
        2D array (n_dist, n_time) containing the strain/acoustic data.
    dist_m : np.ndarray
        1D array of distance values [m], length == tr.shape[0].
    time_s : np.ndarray
        1D array of time values [s] relative to start_datetime_utc,
        length == tr.shape[1].
    fs_hz : float
        Sampling frequency [Hz].
    start_datetime_utc : datetime.datetime, optional
        UTC timestamp corresponding to time_s[0].
    filename : str, optional
        Source file path or name.
    interrogator : str, optional
        Interrogator type used to acquire the data (e.g. "hdas2.5", "optodas").
    downsample : int, optional
        Channel stride applied before loading (1 or None = no subsampling).
    metadata : dict
        Free-form dictionary for any additional interrogator-specific info.
    units : str, optional
        Physical units of `tr`. Convention used in this app:
          - "DC"         : raw digital counts (e.g. HDAS 2.5, uncalibrated)
          - "nanostrain" : strain x 1e9 (e.g. OptaSense, matches das4whales
                           plot convention so config.json vmin/vmax align)
        Always check this field before interpreting absolute amplitude
        values or comparing across interrogators.
    """

    tr: np.ndarray
    dist_m: np.ndarray
    time_s: np.ndarray
    fs_hz: float
    start_datetime_utc: Optional[datetime.datetime] = None
    filename: Optional[str] = None
    interrogator: Optional[str] = None
    downsample: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    units: Optional[str] = None

    @property
    def n_dist(self) -> int:
        return self.tr.shape[0]

    @property
    def n_time(self) -> int:
        return self.tr.shape[1]
