"""
core/msr.py — Multispectral Representation of DAS data.

Implements the MSR pipeline as described in:

    Morell-Monzó et al., "Multispectral Representation of DAS" (submitted).

Pipeline
--------
1. Spectral decomposition — Butterworth bandpass filter per band k.
2. Instantaneous power   — E_k(s,t) = |x_k(s,t)|²
3. Band normalisation    — Ê_k = clip(E_k / v_k, 0, 1),
                           v_k = percentile_p(E_k)
4. Band stacking         — spectral cube Ê(s, t, k), float32 in [0, 1]

The output cube can be used directly for machine learning feature extraction
or visualised via the helper functions in this module.
"""

import numpy as np
import scipy.signal as sp


# ── Core pipeline ─────────────────────────────────────────────────────────────

def multispectral_representation(
    tr: np.ndarray,
    fs_hz: float,
    bands: list,
    percentile: float = 95.0,
    order: int = 5,
) -> np.ndarray:
    """Compute the Multispectral Representation (MSR) of a DAS array.

    Each band is bandpass-filtered, squared to obtain instantaneous power,
    and independently normalised by its p-th percentile before stacking
    into a spectral cube.

    Parameters
    ----------
    tr : np.ndarray
        Raw (unfiltered) DAS array, shape (n_channels, n_time).
    fs_hz : float
        Sampling frequency [Hz].
    bands : list of (fmin, fmax) tuples
        Frequency bands [Hz]. Each band must satisfy
        0 < fmin < fmax < fs_hz / 2.
    percentile : float
        Percentile p used as the per-band normalisation ceiling.
        Default 95, as used in the published methodology.
    order : int
        Butterworth filter order. Default 5.

    Returns
    -------
    np.ndarray
        Spectral cube Ê(s, t, k), shape (n_channels, n_time, n_bands),
        dtype float32, values in [0, 1].

    Raises
    ------
    ValueError
        If tr is not 2D, fs_hz is not positive, or any band is invalid.

    Notes
    -----
    The normalisation assumes that the band-filtered signal is approximately
    zero-centred and symmetrically distributed — a reasonable assumption for
    DAS strain data filtered around a central frequency. Under this assumption
    the single-sided percentile v_k is equivalent to two symmetric bounds
    ±sqrt(v_k) on the signed amplitude distribution.
    """
    if tr.ndim != 2:
        raise ValueError(f"tr must be 2D, got shape {tr.shape}.")
    if fs_hz <= 0:
        raise ValueError(f"fs_hz must be positive, got {fs_hz}.")
    if not bands:
        raise ValueError("bands must contain at least one (fmin, fmax) tuple.")

    nyq = fs_hz / 2.0
    for fmin, fmax in bands:
        if not (0 < fmin < fmax < nyq):
            raise ValueError(
                f"Invalid band [{fmin}, {fmax}] Hz for fs={fs_hz} Hz. "
                f"Each band must satisfy 0 < fmin < fmax < nyquist ({nyq} Hz)."
            )

    n_channels, n_time = tr.shape
    n_bands = len(bands)
    cube = np.zeros((n_channels, n_time, n_bands), dtype=np.float32)

    for k, (fmin, fmax) in enumerate(bands):
        sos = sp.butter(order, [fmin / nyq, fmax / nyq],
                        btype="bandpass", output="sos")
        x_k = sp.sosfiltfilt(sos, tr.astype(np.float64), axis=1)

        # Instantaneous power: E_k = |x_k|²
        e_k = (x_k ** 2).astype(np.float32)

        # Per-band normalisation: v_k = percentile_p(E_k)
        v_k = float(np.percentile(e_k, percentile))
        if v_k <= 0:
            v_k = 1e-30

        # Normalised band: Ê_k = clip(E_k / v_k, 0, 1)
        np.clip(e_k / v_k, 0.0, 1.0, out=e_k)
        cube[:, :, k] = e_k

    return cube


# ── Visualisation helpers ─────────────────────────────────────────────────────

def msr_to_rgb(
    cube: np.ndarray,
    r_band: int = 0,
    g_band: int = 1,
    b_band: int = 2,
) -> np.ndarray:
    """Map three bands of an MSR cube to an RGB image.

    Parameters
    ----------
    cube : np.ndarray
        Spectral cube, shape (n_channels, n_time, n_bands), float32 in [0, 1].
    r_band, g_band, b_band : int
        Indices of the bands to assign to R, G, B respectively.

    Returns
    -------
    np.ndarray
        RGB image, shape (n_channels, n_time, 3), dtype uint8.
    """
    n_bands = cube.shape[2]
    for name, idx in [("r_band", r_band), ("g_band", g_band), ("b_band", b_band)]:
        if not (0 <= idx < n_bands):
            raise ValueError(
                f"{name}={idx} out of range for cube with {n_bands} bands."
            )
    rgb = np.stack([
        cube[:, :, r_band],
        cube[:, :, g_band],
        cube[:, :, b_band],
    ], axis=2)
    return (rgb * 255).astype(np.uint8)


def msr_to_grayscale(
    cube: np.ndarray,
    band: int = 0,
    colormap: str = 'gray',
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> np.ndarray:
    """Map a single MSR band to an image using a colormap.

    Applies a matplotlib colormap to a single band of the spectral cube,
    mapping the [vmin, vmax] range to the full colormap extent.
    With the default ``colormap='gray'`` the result is a standard grayscale
    image. Any matplotlib colormap can be used for false-colour rendering.

    Parameters
    ----------
    cube : np.ndarray
        Spectral cube, shape (n_channels, n_time, n_bands), float32 in [0, 1].
    band : int
        Index of the band to render. Default 0.
    colormap : str
        Matplotlib colormap name. Default ``'gray'``. Examples: ``'viridis'``,
        ``'turbo'``, ``'inferno'``, ``'nipy_spectral'``.
    vmin : float
        Lower bound of the display range. Values below are clipped to the
        first colormap colour. Default 0.0.
    vmax : float
        Upper bound of the display range. Values above are clipped to the
        last colormap colour. Default 1.0.

    Returns
    -------
    np.ndarray
        Colour image, shape (n_channels, n_time, 3), dtype uint8.
    """
    import matplotlib.cm as cm

    n_bands = cube.shape[2]
    if not (0 <= band < n_bands):
        raise ValueError(f"band={band} out of range for cube with {n_bands} bands.")
    if vmin >= vmax:
        raise ValueError(f"vmin ({vmin}) must be less than vmax ({vmax}).")

    cmap = cm.get_cmap(colormap)
    band_data = cube[:, :, band].astype(np.float64)

    # Scale to [0, 1] within [vmin, vmax]
    scaled = np.clip((band_data - vmin) / (vmax - vmin), 0.0, 1.0)

    # Apply colormap — returns RGBA float [0,1]
    rgba = cmap(scaled)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ── Export functions ──────────────────────────────────────────────────────────

def export_npz(
    cube: np.ndarray,
    path: str,
    bands: list = None,
    dist_m: np.ndarray = None,
    time_s: np.ndarray = None,
    fs_hz: float = None,
    percentile: float = None,
) -> None:
    """Save an MSR spectral cube to a compressed NumPy archive (.npz).

    Parameters
    ----------
    cube : np.ndarray
        Spectral cube, shape (n_channels, n_time, n_bands), float32 in [0, 1].
    path : str
        Output file path (e.g. ``"output/msr.npz"``).
    bands : list of (fmin, fmax), optional
        Frequency bands used to generate the cube.
    dist_m : np.ndarray, optional
        Along-cable distance axis [m].
    time_s : np.ndarray, optional
        Time axis [s].
    fs_hz : float, optional
        Sampling frequency [Hz].
    percentile : float, optional
        Normalisation percentile used.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    arrays = {"cube": cube}
    if dist_m is not None:
        arrays["dist_m"] = dist_m
    if time_s is not None:
        arrays["time_s"] = time_s
    if fs_hz is not None:
        arrays["fs_hz"] = np.float32(fs_hz)
    if percentile is not None:
        arrays["percentile"] = np.float32(percentile)
    if bands is not None:
        arrays["bands"] = np.array(bands, dtype=np.float32)

    np.savez_compressed(path, **arrays)


def export_tiff(
    cube: np.ndarray,
    path: str,
    bands: list = None,
    dist_m: np.ndarray = None,
    time_s: np.ndarray = None,
) -> None:
    """Save an MSR spectral cube to a multi-page TIFF file.

    Each band is saved as a separate page in the TIFF, preserving the
    float32 values in [0, 1]. Suitable for GIS tools and scientific image
    viewers that support multi-band TIFF.

    Parameters
    ----------
    cube : np.ndarray
        Spectral cube, shape (n_channels, n_time, n_bands), float32 in [0, 1].
    path : str
        Output file path (e.g. ``"output/msr.tiff"``).
    bands : list of (fmin, fmax), optional
        Frequency bands used to generate the cube — written to TIFF metadata
        if tifffile is available.
    dist_m : np.ndarray, optional
        Along-cable distance axis [m] — stored as metadata if available.
    time_s : np.ndarray, optional
        Time axis [s] — stored as metadata if available.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    try:
        import tifffile
        # Write as (n_bands, n_channels, n_time) — standard band-first order
        data = np.moveaxis(cube, 2, 0)  # (k, s, t)
        metadata = {}
        if bands is not None:
            metadata["bands_hz"] = str(bands)
        if dist_m is not None:
            metadata["dist_m_min"] = float(dist_m[0])
            metadata["dist_m_max"] = float(dist_m[-1])
        if time_s is not None:
            metadata["time_s_min"] = float(time_s[0])
            metadata["time_s_max"] = float(time_s[-1])
        tifffile.imwrite(path, data, metadata=metadata if metadata else None)
    except ImportError:
        # Fallback: PIL — only supports uint8, convert each band as a page
        try:
            from PIL import Image
            pages = []
            n_bands = cube.shape[2]
            for k in range(n_bands):
                band_uint8 = (cube[:, :, k] * 255).astype(np.uint8)
                pages.append(Image.fromarray(band_uint8, mode='L'))
            pages[0].save(
                path, save_all=True,
                append_images=pages[1:],
                compression="tiff_lzw",
            )
        except ImportError:
            raise ImportError(
                "save_tiff requires 'tifffile' (preferred) or 'Pillow'. "
                "Install with: pip install tifffile"
            )


# ── MSRcube — high-level container for spectral cubes ─────────────────────────

class MSRcube:
    """HIGH-LEVEL container for a Multispectral Representation (MSR) spectral cube.

    Wraps a float32 spectral cube (n_channels, n_time, n_bands) and exposes
    methods for visualisation and export. Returned by
    :func:`multispectral_representation` when ``as_cube=True`` and by
    :meth:`DASdataset.msr`.

    Parameters
    ----------
    array : np.ndarray
        Spectral cube, shape (n_channels, n_time, n_bands), float32 in [0, 1].
    bands : list of (fmin, fmax)
        Frequency bands used to generate the cube.
    dist_m : np.ndarray, optional
        Along-cable distance axis [m].
    time_s : np.ndarray, optional
        Time axis [s].
    fs_hz : float, optional
        Sampling frequency [Hz].
    percentile : float, optional
        Normalisation percentile used to generate the cube.

    Examples
    --------
    >>> cube = ds.msr(bands=[(1,5),(5,15),(15,40)])
    >>> cube.to_rgb().shape
    (500, 3000, 3)
    >>> cube.export_npz("output/msr.npz")
    """

    def __init__(
        self,
        array: np.ndarray,
        bands: list,
        dist_m: np.ndarray = None,
        time_s: np.ndarray = None,
        fs_hz: float = None,
        percentile: float = None,
    ):
        self._array     = array
        self.bands      = bands
        self.dist_m     = dist_m
        self.time_s     = time_s
        self.fs_hz      = fs_hz
        self.percentile = percentile

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def array(self) -> np.ndarray:
        """Raw spectral cube, shape (n_channels, n_time, n_bands), float32."""
        return self._array

    @property
    def n_channels(self) -> int:
        return self._array.shape[0]

    @property
    def n_time(self) -> int:
        return self._array.shape[1]

    @property
    def n_bands(self) -> int:
        return self._array.shape[2]

    @property
    def shape(self) -> tuple:
        return self._array.shape

    # ── Visualisation ─────────────────────────────────────────────────────────

    def to_rgb(
        self,
        r_band: int = 0,
        g_band: int = 1,
        b_band: int = 2,
    ) -> np.ndarray:
        """Map three bands to an RGB image.

        Parameters
        ----------
        r_band, g_band, b_band : int
            Band indices for R, G, B channels. Default 0, 1, 2.

        Returns
        -------
        np.ndarray
            RGB image, shape (n_channels, n_time, 3), dtype uint8.
        """
        return msr_to_rgb(self._array, r_band=r_band,
                          g_band=g_band, b_band=b_band)

    def to_grayscale(
        self,
        band: int = 0,
        colormap: str = 'gray',
        vmin: float = 0.0,
        vmax: float = 1.0,
    ) -> np.ndarray:
        """Map a single band to an image using a colormap.

        Parameters
        ----------
        band : int
            Band index. Default 0.
        colormap : str
            Matplotlib colormap name. Default ``'gray'``.
        vmin, vmax : float
            Display range. Default [0.0, 1.0].

        Returns
        -------
        np.ndarray
            Colour image, shape (n_channels, n_time, 3), dtype uint8.
        """
        return msr_to_grayscale(self._array, band=band,
                                colormap=colormap, vmin=vmin, vmax=vmax)

    # ── Export ────────────────────────────────────────────────────────────────

    def export_npz(self, path: str) -> "MSRcube":
        """Save the spectral cube to a compressed NumPy archive.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``"output/msr.npz"``).

        Returns
        -------
        MSRcube
            Self, for method chaining.
        """
        export_npz(
            self._array, path,
            bands=self.bands,
            dist_m=self.dist_m,
            time_s=self.time_s,
            fs_hz=self.fs_hz,
            percentile=self.percentile,
        )
        return self

    def export_tiff(self, path: str) -> "MSRcube":
        """Save the spectral cube to a multi-band TIFF file.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``"output/msr.tiff"``).

        Returns
        -------
        MSRcube
            Self, for method chaining.
        """
        export_tiff(
            self._array, path,
            bands=self.bands,
            dist_m=self.dist_m,
            time_s=self.time_s,
        )
        return self

    # ── Dunder ────────────────────────────────────────────────────────────────

    def to_xarray(self):
        """Convert this spectral cube to an xarray DataArray.

        Returns a DataArray with dimensions ``("distance", "time", "band")``,
        physical coordinates, and band labels as human-readable strings
        (e.g. ``"1-5Hz"``, ``"5-15Hz"``).

        Requires the optional ``xarray`` package.

        Returns
        -------
        xarray.DataArray
            Shape (n_channels, n_time, n_bands),
            dims ("distance", "time", "band").

        Examples
        --------
        >>> cube = ds.msr(bands=[(1,5),(5,15),(15,40)])
        >>> da = cube.to_xarray()
        >>> da.sel(band="5-15Hz").plot()       # plot single band
        >>> da.mean(dim="distance")            # temporal profile
        >>> da.to_netcdf("output/msr.nc")
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for to_xarray(). "
                "Install with: pip install xarray"
            )
        band_labels = [f"{f0}-{f1}Hz" for f0, f1 in self.bands]
        coords = {
            "band": ("band", band_labels, {"units": "Hz"}),
        }
        if self.dist_m is not None:
            coords["distance"] = ("distance", self.dist_m, {"units": "m"})
        if self.time_s is not None:
            coords["time"] = ("time", self.time_s, {"units": "s"})
        attrs = {
            "percentile": float(self.percentile) if self.percentile else None,
            "fs_hz":      float(self.fs_hz) if self.fs_hz else None,
            "bands_hz":   str(self.bands),
        }
        return xr.DataArray(
            data=self._array,
            dims=["distance", "time", "band"],
            coords=coords,
            attrs={k: v for k, v in attrs.items() if v is not None},
            name="msr_cube",
        )

    def __repr__(self) -> str:
        bands_str = ", ".join(f"{f0}-{f1}Hz" for f0, f1 in self.bands)
        return (
            f"MSRcube(shape={self.shape}, "
            f"bands=[{bands_str}], "
            f"percentile={self.percentile})"
        )

    def __array__(self, dtype=None, copy=False) -> np.ndarray:
        """Allow numpy to treat MSRcube as an array directly."""
        arr = self._array
        if dtype is not None:
            arr = arr.astype(dtype, copy=copy)
        return arr
