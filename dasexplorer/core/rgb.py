"""
Multispectral RGB composition for DAS data.

Each output channel (R, G, B) is built independently:
  1. Bandpass-filter the RAW trace at the given [fmin, fmax] band.
  2. Take the absolute value.
  3. Scale by a percentile of that band (clips outliers to white).
  4. Convert to uint8 [0, 255].

Bands are processed one at a time and written directly into the output
array -- only one filtered band is ever held in memory at once.
"""

import numpy as np
import scipy.signal as sp


def _band_to_uint8(tr: np.ndarray, fs_hz: float, fmin: float, fmax: float,
                    percentile: float, order: int = 5) -> np.ndarray:
    """
    Filter tr to [fmin, fmax] Hz, take |signal|, scale by its own
    percentile, and return as uint8 in [0, 255]. Single band only --
    nothing else is held in memory beyond this one array and its
    immediate filtering intermediate.
    """
    nyq = fs_hz / 2.0
    fmin_n = max(fmin, 1e-6) / nyq
    fmax_n = min(fmax, nyq - 1e-6) / nyq
    if fmin_n >= fmax_n:
        raise ValueError(f"Invalid band [{fmin}, {fmax}] Hz for fs={fs_hz} Hz")

    sos = sp.butter(order, [fmin_n, fmax_n], btype="bandpass", output="sos")
    filtered = sp.sosfiltfilt(sos, tr, axis=1)
    filtered = np.abs(filtered)

    ref = np.percentile(filtered, percentile)
    if ref <= 0:
        ref = 1e-12

    scaled = filtered / ref
    np.clip(scaled, 0.0, 1.0, out=scaled)
    return (scaled * 255).astype(np.uint8)


def compute_rgb_composite(
    tr: np.ndarray,
    fs_hz: float,
    r_band: tuple,
    g_band: tuple,
    b_band: tuple,
    percentile: float = 90.0,
    order: int = 5,
) -> np.ndarray:
    """
    Compute a multispectral RGB composite from a raw DAS trace.

    Parameters
    ----------
    tr : np.ndarray, shape (n_dist, n_time)
        RAW (unfiltered) trace. Each band is filtered from this directly.
    fs_hz : float
        Sampling rate.
    r_band, g_band, b_band : (fmin, fmax) tuples in Hz
    percentile : float
        Percentile of |filtered signal| used as the per-band scaling ceiling.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray, shape (n_dist, n_time, 3), dtype uint8
    """
    n_dist, n_time = tr.shape
    rgb = np.zeros((n_dist, n_time, 3), dtype=np.uint8)

    rgb[:, :, 0] = _band_to_uint8(tr, fs_hz, r_band[0], r_band[1], percentile, order)
    rgb[:, :, 1] = _band_to_uint8(tr, fs_hz, g_band[0], g_band[1], percentile, order)
    rgb[:, :, 2] = _band_to_uint8(tr, fs_hz, b_band[0], b_band[1], percentile, order)

    return rgb
