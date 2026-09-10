"""
OptaSense reader — Luna Innovations.

Reads HDF5 .h5 files from the OptaSense interrogator via the das4whales library.
Dataset: OOI RCA 2021 (https://doi.org/10.58046/5J60-FJ89)
"""
import os
import re
import datetime
from typing import Optional
import numpy as np
from dasexplorer.core.data_model import DASDataset
from dasexplorer.core.readers_lib import _ensure_tools_importable


def read_optasense_v1(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
) -> DASDataset:
    
    ######################################################################
    ### OPTASENCE / QUINETIQ, LUNA INNOVATIONS (.h5) OOI-RCA 2021
    # https://doi.org/10.58046/5J60-FJ89
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
    selected_channels_m = kwargs.get("selected_channels_m", None)

    import das4whales as dw

    metadata = dw.data_handle.get_acquisition_parameters(path, interrogator="optasense")

    fs_hz = metadata["fs"]
    dx_m  = metadata["dx"]
    nx    = metadata["nx"]

    if selected_channels_m is None:
        start_m = read_dmin_m if read_dmin_m is not None else 0.0
        stop_m  = read_dmax_m if read_dmax_m is not None else nx * dx_m
        selected_channels_m = [start_m, stop_m, dx_m]

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
    channel_offset = int(round(start_dist_m / dx_m))
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
        reader="optasense",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={
            "gauge_length_m":      metadata.get("GL"),
            "scale_factor":        metadata.get("scale_factor"),
            "selected_channels_m": selected_channels_m,
        },
        units="nanostrain",
    )
