"""
HDAS 2.5 reader — Aragon Photonics Lab.

Reads binary .bin files produced by the APL HDAS 2.5 interrogator.
Requires the proprietary hdas_reader binary (.pyd / .so) in tools/apl/.
"""
import os
import re
import datetime
from typing import Optional
import numpy as np
from dasexplorer.core.data_model import DASDataset
from dasexplorer.core.readers_lib import _ensure_tools_importable


#%% EXTERNAL READERS -------------------------------------------------------------------------------------------

def read_hdas25_v1(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
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
    num_files: int = kwargs.get("num_files", 1)

    from dasexplorer.tools.apl import hdas_reader
    from dasexplorer.tools.apl.utils_2_5 import get_datetime_from_filename

    directory, file_name = os.path.split(path)
    file_start_datetime = get_datetime_from_filename(file_name)

    hdas_data = hdas_reader.load_data(
        first_file=file_name,
        num_files=num_files,
        path=directory,
    )

    fs_hz = hdas_data.trigger_frequency  # fs_hz = 500.0
    dx_m = hdas_data.spatial_sampling_meters # dx_m  = 10.0

    tr     = hdas_data.matrix
    dist_m = np.arange(tr.shape[0]) * dx_m
    time_s = np.arange(tr.shape[1]) / fs_hz

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
        start_datetime_utc=file_start_datetime,
        filename=file_name,
        reader="hdas2.5",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={"num_files": num_files, "dx_m": dx_m},
        units="DC",
    )
