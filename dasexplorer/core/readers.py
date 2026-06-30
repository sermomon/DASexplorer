"""
File readers for DAS Explorer.

Each interrogator type has its own reading function and returns a
DASDataset with a common interface. New interrogator types should be
added to the READERS dict at the bottom of this file.
"""

import os
import re
import sys
import datetime
from typing import Optional
import numpy as np

#from .data_model import DASDataset
from dasexplorer.core.data_model import DASDataset

INTERROGATOR_LABELS = ["HDAS 2.5 [.bin]", "OptaSense [.h5]", "Silixa [.tdms]", "OptoDAS [.hdf5]"]
INTERROGATOR_TYPES  = ["hdas2.5_v1", "optasense_v1", "silixa_v1", "optodas_v1"]

# Path to the DAS-ALME 'tools' package, if it is not already importable.
# Leave as None if 'tools' is already on PYTHONPATH.
DAS_ALME_TOOLS_PATH: Optional[str] = None


def _ensure_tools_importable() -> None:
    if DAS_ALME_TOOLS_PATH is not None and DAS_ALME_TOOLS_PATH not in sys.path:
        sys.path.append(DAS_ALME_TOOLS_PATH)


#%% EXTERNAL READERS -------------------------------------------------------------------------------------------

def read_hdas25_v1(
    path: str,
    num_files: int = 1,
    stride: Optional[int] = None
) -> DASDataset:
    
    ######################################################################
    ### HDAS 2.5 / ARAGON PHOTONICS LAB. - (.bin) UPV + APL EXPERIMENT 
    ######################################################################

    """
    Read a DAS acquisition from an Aragon Photonics HDAS 2.5 interrogator.

    Units
    -----
    The returned tr array is in raw digital counts (DC) — uncalibrated,
    instrument-specific amplitude units. No conversion to strain is applied.

    Parameters
    ----------
    path : str
        Full path to the first .bin file to load.
    num_files : int, optional
        Number of consecutive files to load. Default: 1.
    stride : int, optional
        Channel subsampling factor (tr[::stride, :]).

    Returns
    -------
    DASDataset
    """
    _ensure_tools_importable()

    from dasexplorer.tools.apl import hdas_reader
    from dasexplorer.tools.apl.utils_2_5 import get_datetime_from_filename

    directory, file_name = os.path.split(path)
    file_start_datetime = get_datetime_from_filename(file_name)

    hdas_data = hdas_reader.load_data(
        first_file=file_name,
        num_files=num_files,
        path=directory,
    )

    fs_hz = 500.0
    dx_m  = 10.0

    tr     = hdas_data.matrix
    dist_m = np.arange(tr.shape[0]) * dx_m
    time_s = np.arange(tr.shape[1]) / fs_hz

    downsample = None
    if stride is not None and stride > 1:
        tr     = tr[::stride, :]
        dist_m = dist_m[::stride]
        downsample = stride

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=file_start_datetime,
        filename=file_name,
        interrogator="hdas2.5",
        downsample=downsample,
        metadata={"num_files": num_files, "dx_m": dx_m},
        units="DC",
    )


def read_optasense_v1(
    path: str,
    selected_channels_m: Optional[list] = None,
    stride: Optional[int] = None,
) -> DASDataset:
    
    ######################################################################
    ### OPTASENCE / QUINETIQ, LUNA INNOVATIONS (.h5) OOI-RCA 2021
    ######################################################################

    """
    Read a DAS acquisition from an OptaSense interrogator (HDF5).

    Units
    -----
    The returned tr array is in nanostrain (strain x 1e9). das4whales'
    raw2strain() converts the raw optical phase to absolute strain
    (~1e-9 to 1e-10 range), and we additionally multiply by 1e9 here to
    match the convention used by das4whales' own plot functions
    (plot_tx, plot_tx_env, plot_tx_lined all do `abs(trace) * 1e9` before
    display). This keeps config.json vmin/vmax (e.g. 0-0.4) consistent
    with the das4whales tutorial.

    Parameters
    ----------
    path : str
        Path to the OptaSense HDF5 file.
    selected_channels_m : [start_m, stop_m, step_m], optional
        Channel range in metres. If None, all channels are loaded.
    stride : int, optional
        Channel subsampling factor applied after loading.

    Returns
    -------
    DASDataset
    """
    _ensure_tools_importable()

    import das4whales as dw

    metadata = dw.data_handle.get_acquisition_parameters(path, interrogator="optasense")

    fs_hz = metadata["fs"]
    dx_m  = metadata["dx"]
    nx    = metadata["nx"]

    if selected_channels_m is None:
        selected_channels_m = [0, nx * dx_m, dx_m]

    selected_channels = [int(c // dx_m) for c in selected_channels_m]

    tr, _time, _dist, file_start_datetime = dw.data_handle.load_das_data(
        path, selected_channels, metadata
    )
    tr = tr.copy()

    # das4whales stores raw strain (~1e-9 to 1e-10 range). All das4whales plot
    # functions (plot_tx, plot_tx_env, plot_tx_lined) multiply by 1e9 to display
    # in nanostrain with vmin/vmax around 0-0.4. We replicate that convention
    # here so our config.json vmin/vmax (also in nanostrain) match the data.
    tr = (tr * 1e9).astype(np.float32)

    # dist_m should reflect the real offset of the selected channel range
    # (e.g. 20000-65000 m), not start at 0.
    start_dist_m = selected_channels[0] * dx_m
    dist_m = start_dist_m + np.arange(tr.shape[0]) * dx_m
    time_s = np.arange(tr.shape[1]) / fs_hz

    downsample = None
    if stride is not None and stride > 1:
        tr     = tr[::stride, :]
        dist_m = dist_m[::stride]
        downsample = stride

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=file_start_datetime,
        filename=os.path.basename(path),
        interrogator="optasense",
        downsample=downsample,
        metadata={
            "gauge_length_m":      metadata.get("GL"),
            "scale_factor":        metadata.get("scale_factor"),
            "selected_channels_m": selected_channels_m,
        },
        units="nanostrain",
    )


def read_idas_v1(
    path: str,
    stride: Optional[int] = None,
) -> DASDataset:

    ######################################################################
    ### SILIXA iDAS - (.tdms) OOI RCA 2021
    ######################################################################

    """
    Read a DAS acquisition from a Silixa iDAS interrogator (.tdms format).

    Follows the same convention as das4whales.get_metadata_silixa /
    load_das_data for the OOI RCA 2021 deployment.

    Units
    -----
    The raw TDMS data is int16 unwrapped optical phase. It is converted to
    strain using: scale_factor = (116 * fs * 1e-9) / (GL * 2**13), then the
    per-channel mean is removed (standard das4whales raw2strain step).

    Parameters
    ----------
    path : str
        Full path to the .tdms file.
    stride : int, optional
        Channel subsampling factor (tr[::stride, :]).

    Returns
    -------
    DASDataset
    """
    from nptdms import TdmsFile

    tdms = TdmsFile.read(path)
    props = tdms.properties
    group = tdms["Measurement"]

    # Stack all numbered channels into a (n_channels, n_time) array
    # tr = np.asarray([channel.data for channel in group], dtype=np.float64) # deprecated!
    tr = np.asarray([channel.data for channel in group.channels()], dtype=np.float64)

    fs_hz = float(props["SamplingFrequency[Hz]"])
    dx_m = float(props["SpatialResolution[m]"])
    gauge_length_m = float(props["GaugeLength"])
    refractive_index = float(props["FibreIndex"])
    start_dist_m = float(props["StartPosition[m]"])

    # Phase -> strain conversion (das4whales raw2strain convention)
    scale_factor = (116.0 * fs_hz * 1e-9) / (gauge_length_m * 2 ** 13)
    tr -= np.mean(tr, axis=1, keepdims=True)
    tr *= scale_factor * 1e9  # convert strain -> nanostrain
    tr = tr.astype(np.float32)

    n_channels = tr.shape[0]
    dist_m = start_dist_m + np.arange(n_channels) * dx_m
    time_s = np.arange(tr.shape[1]) / fs_hz

    # UTC start time from the filename: OOIPacCity_UTC_YYYYMMDD_HHMMSS.mmm
    start_dt = None
    fname = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{6})[._](\d+)", fname) # pattern2: "(\d{8})_(\d{6})\.(\d+)"
    if m:
        date_str, time_str, ms_str = m.groups()
        start_dt = datetime.datetime.strptime(
            date_str + time_str, "%Y%m%d%H%M%S"
        ).replace(tzinfo=datetime.timezone.utc)
        start_dt += datetime.timedelta(milliseconds=int(ms_str.ljust(3, "0")[:3]))

    downsample = None
    if stride is not None and stride > 1:
        tr = tr[::stride, :]
        dist_m = dist_m[::stride]
        downsample = stride

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=start_dt,
        filename=fname,
        interrogator="silixa",
        downsample=downsample,
        metadata={
            "dx_m": dx_m,
            "gauge_length_m": gauge_length_m,
            "refractive_index": refractive_index,
            "start_dist_m": start_dist_m,
            "scale_factor": scale_factor,
        },
        units="nanostrain",
    )


def read_optodas_v1(
    path: str,
    stride: Optional[int] = None,
) -> DASDataset:
    
    ######################################################################
    ### OPTODAS - ASN/ ALCATEL SUBMARINE NETWORK (.hdf5) OOI RCA 2025
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

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=start_dt,
        filename=os.path.basename(path),
        interrogator="optodas",
        downsample=downsample,
        metadata={
            "dx_m":           dx_m,
            "gauge_length_m": gl_m,
            "wavelength_m":   wl_m,
            "data_scale":     scale,
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


READERS = {
    "hdas2.5_v1":   read_hdas25_v1,
    "optasense_v1": read_optasense_v1,
    "silixa_v1": read_idas_v1,
    "optodas_v1":   read_optodas_v1,
}


def read_das_file(path: str, interrogator: str, **kwargs) -> DASDataset:
    """
    Dispatch to the appropriate reader based on interrogator type.

    Parameters
    ----------
    path : str
        Path to the DAS file.
    interrogator : str
        One of INTERROGATOR_TYPES.
    **kwargs
        Forwarded to the specific reader.

    Returns
    -------
    DASDataset
    """
    if interrogator not in READERS:
        raise ValueError(
            f"Unknown interrogator type '{interrogator}'. "
            f"Available: {INTERROGATOR_TYPES}"
        )
    return READERS[interrogator](path, **kwargs)


def generate_synthetic_dataset(
    n_dist: int = 600,
    n_time: int = 3000,
    fs_hz: float = 50.0,
    dx_m: float = 10.0,
) -> DASDataset:
    """
    Generate a synthetic DAS dataset for testing the GUI without real data.
    """
    rng = np.random.default_rng(0)
    tr  = rng.normal(0.0, 1.0, size=(n_dist, n_time)).astype(np.float32)

    t_idx   = np.arange(n_time)
    center  = (n_dist / 2.0) + (n_dist / 3.0) * np.sin(2 * np.pi * t_idx / n_time)
    width   = 8.0
    dist_idx = np.arange(n_dist)[:, None]
    envelope = np.exp(-0.5 * ((dist_idx - center[None, :]) / width) ** 2)
    carrier  = np.sin(2 * np.pi * 2.0 * t_idx / fs_hz)
    tr      += 5.0 * envelope * carrier[None, :]

    dist_m = np.arange(n_dist) * dx_m
    time_s = np.arange(n_time) / fs_hz

    return DASDataset(
        tr=tr,
        dist_m=dist_m,
        time_s=time_s,
        fs_hz=fs_hz,
        start_datetime_utc=datetime.datetime.now(datetime.timezone.utc),
        filename="synthetic",
        interrogator="synthetic",
        metadata={"note": "synthetic test data"},
        units="DC",
    )
