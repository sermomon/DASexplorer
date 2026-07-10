"""
OptoDAS / ASN readers.

Reads HDF5 files from Alcatel Submarine Networks (ASN) OptoDAS interrogators.
Two versions:
  - read_optodas_v1: OOI RCA 2025 UW deployment
  - read_optodas_v2: OOI RCA 2024 muxDAS deployment
"""
import os
import re
import datetime
from typing import Optional
import numpy as np
from dasexplorer.core.data_model import DASDataset


def read_optodas_v1(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
) -> DASDataset:

    ######################################################################
    ### OPTODAS / ASN - (.hdf5) OOI-RCA 2024 muxDAS / fsic042
    # https://doi.org/10.58046/4WEF-A282
    ######################################################################

    """
    Read a DAS acquisition from an ASN OptoDAS interrogator, OOI RCA 2024
    deployment (muxDAS experiment).

    Unlike the OOI RCA 2025 OptoDAS files (read_optodas_v1), this version
    stores data directly as float32 in physical units (dataScale == 1.0),
    so no int16 -> float scaling is needed.

    Units
    -----
    Data is returned as-is in rad/(s*m) (phase rate per distance), matching
    header/unit. The header also exposes header/sensitivities
    [rad/(strain*m)] for converting to strain rate if needed in the future.

    Parameters
    ----------
    path : str
        Full path to the .hdf5 file.
    stride : int, optional
        Channel subsampling factor (tr[::stride, :]).

    Returns
    -------
    DASDataset
    """
    import h5py

    with h5py.File(path, "r") as f:
        hdr = f["header"]

        dt_s = float(hdr["dt"][()])
        fs_hz = 1.0 / dt_s
        dx_m = float(hdr["dx"][()])
        gauge_length_m = float(hdr["gaugeLength"][()])
        data_scale = float(hdr["dataScale"][()])

        raw_unit = hdr["unit"][()]
        units = raw_unit.decode() if isinstance(raw_unit, bytes) else str(raw_unit)

        # Data is stored as (n_time, n_channels) -> transpose to (n_channels, n_time)
        raw = f["data"][:]
        tr = raw.T.astype(np.float32)
        if data_scale != 1.0:
            tr *= data_scale

        # Real per-channel distances from cableSpec
        dist_m = f["cableSpec/sensorDistances"][:].astype(np.float64)
        n_ch = tr.shape[0]
        dist_m = dist_m[:n_ch] if len(dist_m) >= n_ch else np.arange(n_ch) * dx_m

        # UTC start time: Unix seconds
        t0_s = float(hdr["time"][()])
        start_dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + \
                   datetime.timedelta(seconds=t0_s)

        # Extra metadata for future strain conversion
        sensitivities = hdr["sensitivities"][()] if "sensitivities" in hdr else None
        instrument = hdr["instrument"][()]
        instrument = instrument.decode() if isinstance(instrument, bytes) else str(instrument)

    time_s = np.arange(tr.shape[1]) * dt_s

    downsample = None
    if stride is not None and stride > 1:
        tr = tr[::stride, :]
        dist_m = dist_m[::stride]
        downsample = stride

    # Spatial crop
    channel_offset = 0
    if read_dmin_m is not None or read_dmax_m is not None:
        dmin = read_dmin_m if read_dmin_m is not None else float(dist_m[0])
        dmax = read_dmax_m if read_dmax_m is not None else float(dist_m[-1])
        mask = (dist_m >= dmin) & (dist_m <= dmax)
        first_idx = int(np.argmax(mask))
        channel_offset = first_idx * int(downsample or 1)
        tr = tr[mask, :]
        dist_m = dist_m[mask]

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=start_dt,
        filename=os.path.basename(path),
        reader="optodas_v2",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={
            "dx_m": dx_m,
            "gauge_length_m": gauge_length_m,
            "data_scale": data_scale,
            "instrument": instrument,
            "sensitivities_rad_per_strain_m": float(sensitivities[0, 0]) if sensitivities is not None else None,
        },
        units=units,
    )


def read_optodas_v2(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
) -> DASDataset:
    
    ######################################################################
    ### OPTODAS - ASN/ ALCATEL SUBMARINE NETWORK (.hdf5) OOI RCA 2025 (preliminary)
    # https://oceanobservatories.org/2025/01/regional-cabled-array-monitoring-axial-seamount-in-real-time/
    ######################################################################
    
    """
    Read a DAS acquisition from an OptoDAS interrogator (Alcatel Subsea Networks),
    HDF5 format as used in the 2024/2025 OOI RCA experiments.

    HDF5 structure (confirmed on UW OOI 2025 files)
    ------------------------------------------------
    /data                          int16  (n_time, n_channels)
    /header/dt                     float64  sampling period [s]
    /header/dx                     float64  nominal channel spacing [m]
    /header/time                   float64  UTC start time [Unix seconds]
    /header/gaugeLength            float64  gauge length [m]
    /header/dataScale              float64  int16 -> rad/(s.m) scale factor
    /header/wavelength             float64  laser wavelength [m]
    /header/unit                   str      physical unit string
    /cableSpec/sensorDistances     float64  (n_channels,) actual dist per channel [m]

    Units
    -----
    Raw int16 values are scaled by header/dataScale to give phase rate per
    distance [rad/(s.m)].  This is returned directly as the data unit.

    Parameters
    ----------
    path : str
        Path to the OptoDAS HDF5 file (.hdf5).
    stride : int, optional
        Channel subsampling factor.

    Returns
    -------
    DASDataset
    """
    import h5py

    with h5py.File(path, "r") as f:
        hdr = f["header"]

        # Sampling parameters
        dt_s   = float(hdr["dt"][()])
        fs_hz  = 1.0 / dt_s
        dx_m   = float(hdr["dx"][()])
        gl_m   = float(hdr["gaugeLength"][()])
        scale  = float(hdr["dataScale"][()])
        wl_m   = float(hdr["wavelength"][()])

        # Unit string
        raw_unit = hdr["unit"][()]
        units = raw_unit.decode() if isinstance(raw_unit, bytes) else str(raw_unit)

        # Start time: Unix seconds -> UTC datetime
        t0_s     = float(hdr["time"][()])
        start_dt = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc) + \
                   datetime.timedelta(seconds=t0_s)

        # Data: int16 (n_time, n_channels) -> float32 (n_channels, n_time)
        raw = f["data"][:]
        tr  = (raw.T.astype(np.float32)) * scale

        # Channel distances: use real per-channel sensorDistances when available
        if "cableSpec/sensorDistances" in f:
            dist_m = f["cableSpec/sensorDistances"][:].astype(np.float64)
            n_ch = tr.shape[0]
            if dist_m.shape[0] >= n_ch:
                dist_m = dist_m[:n_ch]
            else:
                dist_m = np.arange(n_ch) * dx_m
        else:
            dist_m = np.arange(tr.shape[0]) * dx_m

    # Time axis
    time_s = np.arange(tr.shape[1]) * dt_s

    # Optional channel stride
    downsample = None
    if stride is not None and stride > 1:
        tr         = tr[::stride, :]
        dist_m     = dist_m[::stride]
        downsample = stride

    # Spatial crop
    channel_offset = 0
    if read_dmin_m is not None or read_dmax_m is not None:
        dmin = read_dmin_m if read_dmin_m is not None else float(dist_m[0])
        dmax = read_dmax_m if read_dmax_m is not None else float(dist_m[-1])
        mask = (dist_m >= dmin) & (dist_m <= dmax)
        first_idx = int(np.argmax(mask))
        channel_offset = first_idx * int(downsample or 1)
        tr = tr[mask, :]
        dist_m = dist_m[mask]

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=start_dt,
        filename=os.path.basename(path),
        reader="optodas",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={
            "dx_m":           dx_m,
            "gauge_length_m": gl_m,
            "wavelength_m":   wl_m,
            "data_scale":     scale,
        },
        units=units,
    )
