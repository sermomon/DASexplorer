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


def downsample_signal(
    tr: np.ndarray,
    fs_hz: float,
    factor: int = None,
    fs_target: float = None,
    mode: str = 'decimate',
) -> tuple:
    """Temporally downsample a DAS strain-rate array.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).
    fs_hz : float
        Current sampling frequency [Hz].
    factor : int, optional
        Integer decimation factor. Either ``factor`` or ``fs_target`` must
        be provided.
    fs_target : float, optional
        Target sampling frequency [Hz]. The factor is computed as
        ``round(fs_hz / fs_target)`` and must be a positive integer.
    mode : {'decimate', 'simple'}
        ``'decimate'`` (default) — applies a low-pass Chebyshev type I
        anti-aliasing filter before decimation using
        ``scipy.signal.decimate``. Correct for scientific use.
        ``'simple'`` — takes every N-th sample with no pre-filtering.
        Fast but may introduce aliasing.

    Returns
    -------
    tr_down : np.ndarray
        Downsampled array, shape (n_channels, n_time // factor), float32.
    fs_new : float
        New sampling frequency after downsampling [Hz].
    factor : int
        Decimation factor actually applied.
    """
    if factor is None and fs_target is None:
        raise ValueError("Either 'factor' or 'fs_target' must be provided.")
    if fs_target is not None:
        factor = max(1, round(fs_hz / fs_target))
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return tr.astype(np.float32), float(fs_hz), 1

    if mode == 'decimate':
        import scipy.signal as sp
        tr_down = sp.decimate(tr.astype(np.float64), q=factor,
                              ftype='iir', axis=1, zero_phase=True)
        tr_down = tr_down.astype(np.float32)
    elif mode == 'simple':
        tr_down = tr[:, ::factor].astype(np.float32)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'decimate' or 'simple'.")

    fs_new = fs_hz / factor
    return tr_down, float(fs_new), int(factor)


def upsample_signal(
    tr: np.ndarray,
    fs_hz: float,
    factor: int = None,
    fs_target: float = None,
    mode: str = 'linear',
) -> tuple:
    """Temporally upsample a DAS strain-rate array.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).
    fs_hz : float
        Current sampling frequency [Hz].
    factor : int, optional
        Integer upsampling factor. Either ``factor`` or ``fs_target`` must
        be provided.
    fs_target : float, optional
        Target sampling frequency [Hz]. The factor is computed as
        ``round(fs_target / fs_hz)`` and must be a positive integer.
    mode : {'linear', 'cubic', 'fft'}
        ``'linear'`` (default) — linear interpolation using
        ``numpy.interp``. Fast and stable.
        ``'cubic'`` — cubic spline interpolation using
        ``scipy.interpolate.CubicSpline``. Smoother but slower.
        ``'fft'`` — FFT-based resampling using
        ``scipy.signal.resample``. Best spectral fidelity but slowest.

    Returns
    -------
    tr_up : np.ndarray
        Upsampled array, shape (n_channels, n_time * factor), float32.
    fs_new : float
        New sampling frequency after upsampling [Hz].
    factor : int
        Upsampling factor actually applied.
    """
    if factor is None and fs_target is None:
        raise ValueError("Either 'factor' or 'fs_target' must be provided.")
    if fs_target is not None:
        factor = max(1, round(fs_target / fs_hz))
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return tr.astype(np.float32), float(fs_hz), 1

    n_t     = tr.shape[1]
    n_t_new = n_t * factor

    if mode == 'linear':
        t_old = np.arange(n_t)
        t_new = np.linspace(0, n_t - 1, n_t_new)
        tr_up = np.stack([np.interp(t_new, t_old, tr[ch])
                          for ch in range(tr.shape[0])], axis=0).astype(np.float32)

    elif mode == 'cubic':
        from scipy.interpolate import CubicSpline
        t_old = np.arange(n_t, dtype=np.float64)
        t_new = np.linspace(0, n_t - 1, n_t_new)
        tr_up = np.stack([CubicSpline(t_old, tr[ch].astype(np.float64))(t_new)
                          for ch in range(tr.shape[0])], axis=0).astype(np.float32)

    elif mode == 'fft':
        import scipy.signal as sp
        tr_up = sp.resample(tr.astype(np.float64), n_t_new, axis=1).astype(np.float32)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'linear', 'cubic', or 'fft'.")

    fs_new = fs_hz * factor
    return tr_up, float(fs_new), int(factor)


def detrend(
    tr: np.ndarray,
    mode: str = 'linear',
) -> np.ndarray:
    """Remove trend from each channel along the time axis.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).
    mode : {'linear', 'constant'}
        ``'linear'`` (default) — remove best-fit linear trend per channel.
        ``'constant'`` — remove the mean (DC offset) per channel.

    Returns
    -------
    np.ndarray
        Detrended array, same shape as input, float32.
    """
    import scipy.signal as sp
    detrend_type = 'linear' if mode == 'linear' else 'constant'
    return sp.detrend(tr.astype(np.float64), axis=1,
                      type=detrend_type).astype(np.float32)


def taper(
    tr: np.ndarray,
    alpha: float = 0.05,
    mode: str = 'tukey',
) -> np.ndarray:
    """Apply a tapering window to the time edges of each channel.

    Tapering smoothly reduces the signal to zero at both ends of the time
    axis, suppressing edge artefacts before filtering or computing FFTs.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).
    alpha : float
        Fraction of the trace tapered at each end. Default 0.05 (5 %).
        Only used for ``mode='tukey'`` and ``mode='cosine'``.
    mode : {'tukey', 'hann', 'cosine'}
        ``'tukey'`` (default) — Tukey (tapered cosine) window. Flat in the
        centre, cosine taper at both edges. ``alpha`` controls the fraction
        tapered.
        ``'hann'`` — full Hann window. Tapers the entire trace.
        ``'cosine'`` — pure cosine (half-cosine) taper at both edges only,
        equivalent to a split cosine-bell. Same effect as ``'tukey'`` but
        computed directly.

    Returns
    -------
    np.ndarray
        Tapered array, same shape as input, float32.
    """
    import scipy.signal as sp
    n = tr.shape[1]
    if mode == 'tukey':
        win = sp.windows.tukey(n, alpha=alpha * 2)
    elif mode == 'hann':
        win = sp.windows.hann(n)
    elif mode == 'cosine':
        win = np.ones(n)
        n_tap = max(1, int(n * alpha))
        cos_taper = 0.5 * (1 - np.cos(np.pi * np.arange(n_tap) / n_tap))
        win[:n_tap] = cos_taper
        win[-n_tap:] = cos_taper[::-1]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'tukey', 'hann', or 'cosine'.")
    return (tr.astype(np.float32) * win[np.newaxis, :].astype(np.float32))


def normalize(
    tr: np.ndarray,
    mode: str = 'rms',
    axis: int = 1,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Normalize each channel (or trace) of a DAS array.

    Parameters
    ----------
    tr : np.ndarray
        Input array, shape (n_channels, n_time).
    mode : {'rms', 'peak', 'zscore'}
        ``'rms'`` (default) — divide each channel by its RMS amplitude.
        ``'peak'`` — divide each channel by its absolute maximum.
        ``'zscore'`` — subtract mean and divide by standard deviation
        (zero-mean, unit-variance normalisation).
    axis : int
        Axis along which normalisation is computed. Default 1 (time axis).
    epsilon : float
        Small value added to the denominator to avoid division by zero.
        Default 1e-10.

    Returns
    -------
    np.ndarray
        Normalised array, same shape as input, float32.
    """
    tr = tr.astype(np.float64)
    if mode == 'rms':
        scale = np.sqrt(np.mean(tr ** 2, axis=axis, keepdims=True)) + epsilon
    elif mode == 'peak':
        scale = np.max(np.abs(tr), axis=axis, keepdims=True) + epsilon
    elif mode == 'zscore':
        mean  = np.mean(tr, axis=axis, keepdims=True)
        scale = np.std(tr,  axis=axis, keepdims=True) + epsilon
        return ((tr - mean) / scale).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'rms', 'peak', or 'zscore'.")
    return (tr / scale).astype(np.float32)
