"""
Generic data I/O formats for DAS Explorer.

These readers are not tied to a specific interrogator, they load
DASDataset objects previously exported by DAS Explorer itself (via
File > Save as NPZ / MAT), preserving all original metadata.
"""

import os
import re
import sys
import datetime
from typing import Optional
import numpy as np

#from .data_model import DASDataset
from dasexplorer.core.data_model import DASDataset

def read_npz(path: str) -> DASDataset:
    """
    Read a DAS dataset previously exported via File > Save as NPZ.

    The .npz stores tr, dist_m, time_s, fs_hz, and all metadata needed to
    reconstruct a DASDataset exactly as the original reader would have
    produced it (units, interrogator type, downsample, original filename,
    start time, and any free-form metadata as a JSON string).

    Parameters
    ----------
    path : str
        Path to the .npz file.

    Returns
    -------
    DASDataset
    """
    import json

    with np.load(path, allow_pickle=False) as npz:
        tr      = npz["tr"]
        dist_m  = npz["dist_m"]
        time_s  = npz["time_s"]
        fs_hz   = float(npz["fs_hz"])

        start_iso = str(npz["start_datetime_utc"]) if "start_datetime_utc" in npz else ""
        start_datetime_utc = None
        if start_iso:
            try:
                start_datetime_utc = datetime.datetime.fromisoformat(start_iso)
            except ValueError:
                start_datetime_utc = None

        filename     = str(npz["filename"]) if "filename" in npz else os.path.basename(path)
        interrogator = str(npz["interrogator"]) if "interrogator" in npz else None
        downsample   = int(npz["downsample"]) if "downsample" in npz else None
        units        = str(npz["units"]) if "units" in npz else None

        metadata = {}
        if "metadata_json" in npz:
            try:
                metadata = json.loads(str(npz["metadata_json"]))
            except (ValueError, TypeError):
                metadata = {}

    return DASDataset(
        tr=tr.astype(np.float32),
        dist_m=dist_m.astype(np.float64),
        time_s=time_s.astype(np.float64),
        fs_hz=fs_hz,
        start_datetime_utc=start_datetime_utc,
        filename=filename,
        interrogator=interrogator or None,
        downsample=downsample,
        metadata=metadata,
        units=units or None,
    )


def _mat_scalar(value):
    """scipy.io.loadmat wraps scalars as e.g. [[50.0]] — unwrap to a plain
    Python number."""
    arr = np.asarray(value)
    return arr.item() if arr.size == 1 else arr


def _mat_text(value) -> str:
    """scipy.io.loadmat wraps strings as e.g. array(['hello'], dtype='<U5'),
    and empty strings as a zero-size array — unwrap to a plain str, '' if
    empty."""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    return str(arr.reshape(-1)[0])


def read_mat(path: str) -> DASDataset:
    """
    Read a DAS dataset previously exported via File > Save as MAT.

    Same variable set and semantics as read_npz, stored in MATLAB .mat
    format (scipy.io.savemat/loadmat) instead of NumPy's .npz, for
    interoperability with MATLAB-based workflows.

    Parameters
    ----------
    path : str
        Path to the .mat file.

    Returns
    -------
    DASDataset
    """
    import json
    import scipy.io as sio

    mat = sio.loadmat(path)

    tr     = np.asarray(mat["tr"])
    dist_m = np.asarray(mat["dist_m"]).reshape(-1)
    time_s = np.asarray(mat["time_s"]).reshape(-1)
    fs_hz  = float(_mat_scalar(mat["fs_hz"]))

    start_iso = _mat_text(mat["start_datetime_utc"]) if "start_datetime_utc" in mat else ""
    start_datetime_utc = None
    if start_iso:
        try:
            start_datetime_utc = datetime.datetime.fromisoformat(start_iso)
        except ValueError:
            start_datetime_utc = None

    filename     = _mat_text(mat["filename"]) if "filename" in mat else os.path.basename(path)
    interrogator = _mat_text(mat["interrogator"]) if "interrogator" in mat else ""
    downsample_raw = _mat_scalar(mat["downsample"]) if "downsample" in mat else None
    downsample   = int(downsample_raw) if downsample_raw is not None else None
    units        = _mat_text(mat["units"]) if "units" in mat else ""

    metadata = {}
    if "metadata_json" in mat:
        try:
            metadata = json.loads(_mat_text(mat["metadata_json"]))
        except (ValueError, TypeError):
            metadata = {}

    return DASDataset(
        tr=tr.astype(np.float32),
        dist_m=dist_m.astype(np.float64),
        time_s=time_s.astype(np.float64),
        fs_hz=fs_hz,
        start_datetime_utc=start_datetime_utc,
        filename=filename or os.path.basename(path),
        interrogator=interrogator or None,
        downsample=downsample,
        metadata=metadata,
        units=units or None,
    )
