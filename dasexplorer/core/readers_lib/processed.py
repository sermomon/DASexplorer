"""
Processed / pre-processed DAS format readers.

These readers handle data that has already been processed (converted to
physical units, cropped, annotated) and distributed in standard scientific
formats (.mat, .npz). The data may have been originally recorded by any
interrogator.

Currently:
  - read_svalbard_v1: Svalbard DAS4Whale dataset (Bouffaut et al. 2022)
    https://doi.org/10.5281/zenodo.5823343
"""
import os
import re
import datetime
from typing import Optional
import numpy as np
from dasexplorer.core.data_model import DASDataset
import scipy.io as sio


def read_svalbard_v1(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
) -> DASDataset:

    ######################################################################
    ### OPTODAS ASN/ ALCATEL SUBMARINE NETWORK - (.mat) SVALBARD-2020
    ### https://doi.org/10.5281/zenodo.5823343
    ######################################################################

    """
    Read a DAS acquisition from the Svalbard DAS4Whale dataset
    (Bouffaut et al., 2022, Front. Mar. Sci.).

    Data is stored as MATLAB HDF5 .mat files. Each file covers a
    subset of channels along the 120 km Svalbard fiber optic cable
    (Longyearbyen to open ocean through Isfjorden).

    Units
    -----
    Data is already in nanostrain — no scaling required.

    Parameters
    ----------
    path : str
        Full path to the .mat file.
    stride : int, optional
        Channel subsampling factor (tr[::stride, :]).

    Returns
    -------
    DASDataset
    """
    import scipy.io as sio

    mat = sio.loadmat(path)

    # Data: (n_channels, n_time), already in nanostrain
    tr = np.asarray(mat["data"]).astype(np.float32)

    # Sampling parameters
    fs_hz = float(np.asarray(mat["info_sampling_frequency_Hz"]).ravel()[0])
    dt_s  = float(np.asarray(mat["info_sample_interval_s"]).ravel()[0])
    dx_m  = float(np.asarray(mat["info_SSI_m"]).ravel()[0])
    gl_m  = float(np.asarray(mat["info_GL_m"]).ravel()[0])

    # Units
    #raw_units = np.asarray(mat["info_units"]).ravel()
    #units = str(raw_units[0]) if raw_units.size > 0 else "nanostrain"
    units = "nanostrain"

    # Channel distances from shore [m] — use x1_distance_from_shore_m
    dist_m = np.asarray(mat["x1_distance_from_shore_m"]).ravel().astype(np.float64)
    if dist_m.shape[0] != tr.shape[0]:
        dist_m = np.arange(tr.shape[0]) * dx_m

    # Time vector [s]
    time_s = np.asarray(mat["x2_time_s"]).ravel().astype(np.float64)
    if time_s.shape[0] != tr.shape[1]:
        time_s = np.arange(tr.shape[1]) * dt_s

    # UTC start time from info_timestamp (string: e.g. '20200627_052441')
    start_dt = None
    try:
        raw_ts = np.asarray(mat["info_timestamp"]).ravel()
        ts_str = str(raw_ts[0]).strip()
        start_dt = datetime.datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(
            tzinfo=datetime.timezone.utc
        )
    except Exception:
        # Fallback: parse from filename YYYYMMDD_HHMMSS_ch...
        m = re.search(r"(\d{8})_(\d{6})", os.path.basename(path))
        if m:
            start_dt = datetime.datetime.strptime(
                m.group(1) + m.group(2), "%Y%m%d%H%M%S"
            ).replace(tzinfo=datetime.timezone.utc)

    downsample = None
    if stride is not None and stride > 1:
        tr     = tr[::stride, :]
        dist_m = dist_m[::stride]
        downsample = stride

    # Spatial crop applied after stride.
    # channel_offset = original (stride=1) cable index of the first kept channel.
    channel_offset = 0
    if read_dmin_m is not None or read_dmax_m is not None:
        dmin = read_dmin_m if read_dmin_m is not None else float(dist_m[0])
        dmax = read_dmax_m if read_dmax_m is not None else float(dist_m[-1])
        mask = (dist_m >= dmin) & (dist_m <= dmax)
        first_idx = int(np.argmax(mask))
        channel_offset = first_idx * int(downsample or 1)
        tr     = tr[mask, :]
        dist_m = dist_m[mask]

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=start_dt,
        filename=os.path.basename(path),
        reader="svalbard_v1",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={
            "dx_m": dx_m,
            "gauge_length_m": gl_m,
        },
        units=units,
    )

#%% RE-IMPORT READERS ------------------------------------------------------------------------------------------


# def read_npz(path: str) -> DASDataset:
#     """
#     Read a DAS dataset previously exported via File > Save as NPZ.

#     The .npz stores tr, dist_m, time_s, fs_hz, and all metadata needed to
#     reconstruct a DASDataset exactly as the original reader would have
#     produced it (units, interrogator type, downsample, original filename,
#     start time, and any free-form metadata as a JSON string).

#     Parameters
#     ----------
#     path : str
#         Path to the .npz file.

#     Returns
#     -------
#     DASDataset
#     """
#     import json

#     with np.load(path, allow_pickle=False) as npz:
#         tr      = npz["tr"]
#         dist_m  = npz["dist_m"]
#         time_s  = npz["time_s"]
#         fs_hz   = float(npz["fs_hz"])

#         start_iso = str(npz["start_datetime_utc"]) if "start_datetime_utc" in npz else ""
#         start_datetime_utc = None
#         if start_iso:
#             try:
#                 start_datetime_utc = datetime.datetime.fromisoformat(start_iso)
#             except ValueError:
#                 start_datetime_utc = None

#         filename     = str(npz["filename"]) if "filename" in npz else os.path.basename(path)
#         interrogator = str(npz["interrogator"]) if "interrogator" in npz else None
#         downsample   = int(npz["downsample"]) if "downsample" in npz else None
#         units        = str(npz["units"]) if "units" in npz else None

#         metadata = {}
#         if "metadata_json" in npz:
#             try:
#                 metadata = json.loads(str(npz["metadata_json"]))
#             except (ValueError, TypeError):
#                 metadata = {}

#     return DASDataset(
#         tr=tr.astype(np.float32),
#         dist_m=dist_m.astype(np.float64),
#         time_s=time_s.astype(np.float64),
#         fs_hz=fs_hz,
#         start_datetime_utc=start_datetime_utc,
#         filename=filename,
#         interrogator=interrogator or None,
#         downsample=downsample,
#         metadata=metadata,
#         units=units or None,
#     )


# def _mat_scalar(value):
#     """scipy.io.loadmat wraps scalars as e.g. [[50.0]] — unwrap to a plain
#     Python number."""
#     arr = np.asarray(value)
#     return arr.item() if arr.size == 1 else arr


# def _mat_text(value) -> str:
#     """scipy.io.loadmat wraps strings as e.g. array(['hello'], dtype='<U5'),
#     and empty strings as a zero-size array — unwrap to a plain str, '' if
#     empty."""
#     arr = np.asarray(value)
#     if arr.size == 0:
#         return ""
#     return str(arr.reshape(-1)[0])


# def read_mat(path: str) -> DASDataset:
#     """
#     Read a DAS dataset previously exported via File > Save as MAT.

#     Same variable set and semantics as read_npz, stored in MATLAB .mat
#     format (scipy.io.savemat/loadmat) instead of NumPy's .npz, for
#     interoperability with MATLAB-based workflows.

#     Parameters
#     ----------
#     path : str
#         Path to the .mat file.

#     Returns
#     -------
#     DASDataset
#     """
#     import json
#     import scipy.io as sio

#     mat = sio.loadmat(path)

#     tr     = np.asarray(mat["tr"])
#     dist_m = np.asarray(mat["dist_m"]).reshape(-1)
#     time_s = np.asarray(mat["time_s"]).reshape(-1)
#     fs_hz  = float(_mat_scalar(mat["fs_hz"]))

#     start_iso = _mat_text(mat["start_datetime_utc"]) if "start_datetime_utc" in mat else ""
#     start_datetime_utc = None
#     if start_iso:
#         try:
#             start_datetime_utc = datetime.datetime.fromisoformat(start_iso)
#         except ValueError:
#             start_datetime_utc = None

#     filename     = _mat_text(mat["filename"]) if "filename" in mat else os.path.basename(path)
#     interrogator = _mat_text(mat["interrogator"]) if "interrogator" in mat else ""
#     downsample_raw = _mat_scalar(mat["downsample"]) if "downsample" in mat else None
#     downsample   = int(downsample_raw) if downsample_raw is not None else None
#     units        = _mat_text(mat["units"]) if "units" in mat else ""

#     metadata = {}
#     if "metadata_json" in mat:
#         try:
#             metadata = json.loads(_mat_text(mat["metadata_json"]))
#         except (ValueError, TypeError):
#             metadata = {}

#     return DASDataset(
#         tr=tr.astype(np.float32),
#         dist_m=dist_m.astype(np.float64),
#         time_s=time_s.astype(np.float64),
#         fs_hz=fs_hz,
#         start_datetime_utc=start_datetime_utc,
#         filename=filename or os.path.basename(path),
#         interrogator=interrogator or None,
#         downsample=downsample,
#         metadata=metadata,
#         units=units or None,
#     )


#%% READERS DICTIONARY ------------------------------------------------------------------------------------------
