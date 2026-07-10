"""
Silixa iDAS reader.

Reads TDMS files produced by the Silixa iDAS interrogator via nptdms.
Dataset: OOI RCA 2021 (https://doi.org/10.58046/5J60-FJ89)
"""
import os
import re
import datetime
from typing import Optional
import numpy as np
from dasexplorer.core.data_model import DASDataset


def read_idas_v1(
    path: str,
    stride: Optional[int] = None,
    read_dmin_m: Optional[float] = None,
    read_dmax_m: Optional[float] = None,
    **kwargs,
) -> DASDataset:

    ######################################################################
    ### SILIXA iDAS - (.tdms) OOI RCA 2021
    # https://doi.org/10.58046/5J60-FJ89
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
        filename=fname,
        reader="silixa",
        downsample=downsample,
        channel_offset=channel_offset,
        metadata={
            "dx_m": dx_m,
            "gauge_length_m": gauge_length_m,
            "refractive_index": refractive_index,
            "start_dist_m": start_dist_m,
            "scale_factor": scale_factor,
        },
        units="nanostrain",
    )
