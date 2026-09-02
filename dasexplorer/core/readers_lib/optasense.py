"""
OptaSense reader — Luna Innovations.

Reads HDF5 .h5 files from the OptaSense interrogator.
Reimplemented directly with h5py, replicating the das4whales
data_handle.get_metadata_optasense / load_das_data logic exactly.

Dataset: OOI RCA 2021 (https://doi.org/10.58046/5J60-FJ89)
"""

import os
from datetime import datetime
from typing import Optional

import h5py
import numpy as np

from dasexplorer.core.data_model import DASDataset


def read_optasense_v1(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
) -> DASDataset:
    """Read a DAS acquisition from a Luna Innovations OptaSense interrogator (.h5).

    Replicates the das4whales data_handle.get_metadata_optasense and
    load_das_data(interrogator="optasense") pipeline using h5py directly,
    without requiring das4whales as a dependency.

    The returned tr array is in nanostrain (strain x 1e9), consistent with
    the das4whales tutorial convention (vmin=0, vmax=0.4 nanostrain).

    Parameters
    ----------
    path : str
        Full path to the .h5 file.
    stride : int, optional
        Spatial decimation factor. If > 1, every stride-th channel is loaded.
        Applied directly at read time via HDF5 slicing (no full array loaded).
    read_dmin_m : float, optional
        Start of the spatial range to load [m]. None = start of cable.
    read_dmax_m : float, optional
        End of the spatial range to load [m]. None = end of cable.
    **kwargs
        Additional keyword arguments (ignored).

    Returns
    -------
    DASDataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, "r") as fp:
        acq   = fp["Acquisition"]
        raw0  = acq["Raw[0]"]

        # ── Metadata ─────────────────────────────────────────────────────
        fs_hz = float(raw0.attrs["OutputDataRate"])
        dx_m  = float(acq.attrs["SpatialSamplingInterval"])
        nx    = int(raw0.attrs["NumberOfLoci"])
        n_ri  = float(acq["Custom"].attrs["Fibre Refractive Index"])
        GL    = float(acq.attrs["GaugeLength"])

        # Scale factor: converts raw int16 counts → strain (dimensionless)
        # Same formula as das4whales data_handle.get_metadata_optasense
        scale_factor = (
            (2 * np.pi) / 2**16
            * (1550.12e-9)
            / (0.78 * 4 * np.pi * n_ri * GL)
        )

        start_dist_m = float(
            acq["Custom"].attrs["Output Channel Start (CSU)"]
        ) * dx_m

        # ── Timestamp ─────────────────────────────────────────────────────
        raw_time = raw0["RawDataTime"]
        file_start_datetime = datetime.utcfromtimestamp(float(raw_time[0]) * 1e-6)

        # ── Channel selection (stride + spatial crop) ─────────────────────
        effective_stride = int(stride) if (stride is not None and stride > 1) else 1

        # Convert spatial crop from metres to channel indices
        ch_start = 0
        ch_stop  = nx
        if read_dmin_m is not None:
            ch_start = max(0, int((read_dmin_m - start_dist_m) / dx_m))
        if read_dmax_m is not None:
            ch_stop  = min(nx, int((read_dmax_m - start_dist_m) / dx_m) + 1)

        # ── Load data with HDF5 slicing (no full array in memory) ─────────
        raw_data = raw0["RawData"]

        # Handle orientation: some files are (nx, ns), some (ns, nx)
        if raw_data.shape[0] == nx:
            tr = raw_data[ch_start:ch_stop:effective_stride, :].astype(np.float64)
        else:
            tr = raw_data[:, ch_start:ch_stop:effective_stride].T.astype(np.float64)

    # ── raw2strain: remove per-channel mean, apply scale factor ──────────
    # Multiply by 1e9 to express in nanostrain (das4whales convention)
    tr -= np.mean(tr, axis=1, keepdims=True)
    tr *= scale_factor * 1e9
    tr = tr.astype(np.float32)

    # ── Build axes ────────────────────────────────────────────────────────
    n_ch, n_t = tr.shape
    fs_hz = float(fs_hz)
    time_s = np.arange(n_t) / fs_hz
    dist_m = (
        start_dist_m
        + np.arange(n_ch) * dx_m * effective_stride
        + ch_start * dx_m
    )

    downsample    = effective_stride if effective_stride > 1 else None
    # channel_offset: index of the first loaded channel in the full cable (stride=1)
    channel_offset = ch_start

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=file_start_datetime,
        filename=os.path.basename(path),
        reader="optasense_v1",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={
            "dx_m": dx_m,
            "GL": GL,
            "n": n_ri,
            "scale_factor": scale_factor,
            "start_dist_m": start_dist_m,
        },
        units="nanostrain",
    )
