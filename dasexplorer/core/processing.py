"""
core/processing.py — Public signal processing functions for DAS data.

All functions operate on raw numpy arrays and are intentionally kept
separate from DASDataset so they can be used independently.
"""

import numpy as np
import scipy.signal as sp
from scipy.fft import next_fast_len


def bandpass_filter(
    tr: np.ndarray,
    fs_hz: float,
    fmin: float,
    fmax: float,
    order: int = 5,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along the time axis.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).
    fs_hz : float
        Sampling frequency [Hz].
    fmin : float
        Low-frequency cutoff [Hz].
    fmax : float
        High-frequency cutoff [Hz].
    order : int
        Filter order. Default 5.

    Returns
    -------
    np.ndarray
        Filtered array, same shape as input, float32.
    """
    nyq = fs_hz / 2.0
    if fmin <= 0:
        fmin = 1e-3
    if fmax >= nyq:
        fmax = nyq - 1e-3
    sos = sp.butter(order, [fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    return sp.sosfiltfilt(sos, tr, axis=1).astype(np.float32)


def hilbert_envelope(tr: np.ndarray) -> np.ndarray:
    """Compute the Hilbert envelope (instantaneous amplitude) along the time axis.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).

    Returns
    -------
    np.ndarray
        Envelope array, same shape as input, float32.
    """
    n    = tr.shape[1]
    nfft = next_fast_len(n)
    env  = np.abs(sp.hilbert(tr, N=nfft, axis=1)[:, :n])
    return env.astype(np.float32)
