"""
F-K filter implementation for DAS data.

Reimplemented from das4whales.dsp (Bouffaut et al., 2023, GPL-3.0) using
numpy, scipy and scipy.fft only — no OpenCV or sparse dependencies required.

The algorithm is identical to hybrid_ninf_gs_filter_design +
fk_filter_sparsefilt in das4whales:
  1. Build frequency and wavenumber axes.
  2. For each frequency in [fmin, fmax], set the wavenumber passband
     defined by the apparent velocity range [c_min, c_max].
  3. Apply Gaussian smoothing to the filter edges (replaces cv2.GaussianBlur).
  4. Symmetrize the filter (flipud + fliplr).
  5. Apply via FFT2 → multiply → IFFT2.
"""

import numpy as np
import scipy.signal as sp_sig
import scipy.ndimage as sp_nd
from scipy.fft import fft2, ifft2


def fk_filter_design(
    trace_shape: tuple,
    dx: float,
    fs: float,
    c_min: float,
    c_max: float,
    fmin: float,
    fmax: float,
    stride: int = 1,
    gaussian_sigma: float = 40.0,
) -> np.ndarray:
    """Design a bandpass F-K filter for DAS strain data.

    Equivalent to das4whales.dsp.hybrid_ninf_gs_filter_design with
    display_filter=False, reimplemented with scipy instead of OpenCV.

    Parameters
    ----------
    trace_shape : tuple
        Shape of the data array (n_channels, n_time).
    dx : float
        Channel spacing [m].
    fs : float
        Sampling frequency [Hz].
    c_min : float
        Minimum apparent velocity [m/s].
    c_max : float
        Maximum apparent velocity [m/s].
    fmin : float
        Minimum frequency [Hz].
    fmax : float
        Maximum frequency [Hz].
    stride : int
        Channel stride applied during loading. Effective dx = dx * stride.
    gaussian_sigma : float
        Standard deviation of the Gaussian smoothing applied to the filter
        edges (in pixels). Default 40 matches das4whales cv2.GaussianBlur.

    Returns
    -------
    fk_filter : np.ndarray, shape (n_channels, n_time), float32
        The F-K filter in the fftshifted domain.
    """
    nnx, nns = trace_shape
    effective_dx = dx * stride

    # Frequency and wavenumber axes (fftshifted)
    freq = np.fft.fftshift(np.fft.fftfreq(nns, d=1.0 / fs))
    knum = np.fft.fftshift(np.fft.fftfreq(nnx, d=effective_dx))

    fmin_idx = int(np.argmax(freq >= fmin))
    fmax_idx = int(np.argmax(freq >= fmax))

    fk_filter = np.zeros((len(knum), len(freq)), dtype=np.float32)

    for i in range(fmin_idx, fmax_idx):
        kp_min = freq[i] / c_max
        kp_max = freq[i] / c_min
        fk_filter[:, i] = ((knum > kp_min) & (knum < kp_max)).astype(np.float32)

    # Apply Gaussian smoothing to the positive-frequency quadrant
    # (mirrors das4whales cv2.GaussianBlur(sub_matrix, (0,0), 40))
    half_k = len(knum) // 2
    half_f = len(freq) // 2
    sub = fk_filter[half_k:, half_f:].copy()
    sub = sp_nd.gaussian_filter(sub, sigma=gaussian_sigma)
    fk_filter[half_k:, half_f:] = sub

    # Symmetrize (das4whales: += fliplr + flipud)
    fk_filter += np.fliplr(fk_filter)
    fk_filter += np.flipud(fk_filter)

    return fk_filter


def fk_filter_apply(
    tr: np.ndarray,
    fk_filter: np.ndarray,
    tapering: bool = False,
) -> np.ndarray:
    """Apply a pre-designed F-K filter to DAS strain data.

    Equivalent to das4whales.dsp.fk_filter_sparsefilt, reimplemented with
    scipy.fft for parallelism without sparse or OpenCV dependencies.

    Parameters
    ----------
    tr : np.ndarray, shape (n_channels, n_time)
        Input strain data (bandpass-filtered, no Hilbert envelope).
    fk_filter : np.ndarray, shape (n_channels, n_time)
        F-K filter from fk_filter_design, in fftshifted domain.
    tapering : bool
        If True, apply a Tukey window along the time axis before filtering
        to reduce edge effects. Use False when data is already bandpass-filtered.

    Returns
    -------
    np.ndarray, shape (n_channels, n_time), float32
        F-K filtered data in the spatio-temporal domain.
    """
    tr = np.asarray(tr, dtype=np.complex64)

    if tapering:
        win = sp_sig.windows.tukey(tr.shape[1], alpha=0.03)
        tr = tr * win[np.newaxis, :]

    # FFT2 → fftshift → multiply → ifftshift → IFFT2
    fk_tr = np.fft.fftshift(fft2(tr, workers=-1))
    fk_tr *= fk_filter
    tr_filt = ifft2(np.fft.ifftshift(fk_tr), workers=-1)

    return tr_filt.real.astype(np.float32)
