"""
Analysis functions for DAS Explorer annotation windows.

All functions operate on numpy arrays and return plain numpy results,
with no Qt or pyqtgraph dependency.
"""

import numpy as np
import scipy.signal as sp


def compute_spectrogram(signal: np.ndarray, fs: float,
                        nperseg: int = 256, noverlap: int = None,
                        nfft: int = None):
    """
    Compute the spectrogram of a 1-D signal (PSD, dB scale).

    Returns
    -------
    f : np.ndarray  frequency axis [Hz]
    t : np.ndarray  time axis [s]
    Sxx_db : np.ndarray  (n_freq, n_time) power in dB
    """
    n_samples = len(signal)

    # Clamp nperseg to the number of available samples.
    nperseg = min(nperseg, n_samples)

    if noverlap is None:
        noverlap = nperseg // 2
    else:
        # Clamp noverlap: must be strictly less than (possibly clamped) nperseg.
        noverlap = min(noverlap, nperseg - 1)

    if nfft is None:
        nfft = nperseg
    else:
        nfft = max(nfft, nperseg)  # nfft must be >= nperseg

    f, t, Sxx = sp.spectrogram(
        signal, fs=fs,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        window='hann', scaling='density',
    )
    Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-20))
    return f, t, Sxx_db


def compute_spectrum(signal: np.ndarray, fs: float,
                     nfft: int = 2048,
                     window: str = 'hann') -> tuple:
    """
    Compute the magnitude spectrum of a 1-D signal.

    Returns
    -------
    freqs : np.ndarray  frequency axis [Hz]
    mag   : np.ndarray  magnitude spectrum
    """
    n = len(signal)
    if window == 'hann':
        w = np.hanning(n)
    elif window == 'hamming':
        w = np.hamming(n)
    elif window == 'blackman':
        w = np.blackman(n)
    elif window == 'bartlett':
        w = np.bartlett(n)
    else:
        w = np.ones(n)

    F = np.fft.rfft(signal * w, n=nfft)
    mag   = np.abs(F)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mag


def select_channels_for_spectral(di0: int, di1: int,
                                  max_n: int = 30) -> list:
    """
    Return up to max_n channel indices uniformly sampled in [di0, di1).
    """
    n = di1 - di0
    if n <= 0:
        return [di0]
    if n <= max_n:
        return list(range(di0, di1))
    idx = np.round(np.linspace(di0, di1 - 1, max_n)).astype(int)
    return sorted(set(idx.tolist()))
