"""
core/data.py — High-level DASdataset class with processing methods.

DASdataset inherits from _DASRecord (the internal dataclass) and adds:
  - Convenience properties: dx, dt, duration_s, cable_length_m, nyquist_hz
  - Processing methods: bandpass, envelope, fk_filter, rgb
  - Export methods: save_npz, save_mat
  - Constructors: from_file, from_dataset
"""

import os
import dataclasses
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio

from dasexplorer.core.data_model import _DASRecord
from dasexplorer.core.processing import bandpass_filter, hilbert_envelope


# TODO: Migrate DASdataset to xarray.DataArray
#
# Currently DASdataset inherits from _DASRecord (a plain Python dataclass)
# and stores `tr` as a raw numpy array with `dist_m`, `time_s`, `fs_hz` as
# separate attributes. This works but misses the benefits of a proper
# N-dimensional labelled array.
#
# The right long-term design — as demonstrated by DASCore (Patch) — is to
# base DASdataset on xarray.DataArray instead of _DASRecord. The DataArray
# would store:
#   - data   = tr (n_channels, n_time) float32
#   - dims   = ("distance", "time")
#   - coords = {"distance": dist_m [m], "time": time_s [s]}
#   - attrs  = {fs_hz, units, reader, filename, channel_stride, ...}
#
# Benefits:
#   - .sel(distance=slice(20000, 30000)) — selection by physical value
#   - .to_netcdf("output.nc") — standard scientific export format
#   - Native interoperability with DASCore, ObsPy, hvPlot, Dask
#   - Rolling, resample, groupby operations built-in
#   - Units support via pint-xarray
#   - to_xarray() would simply return self instead of building a new object
#
# Architecture after migration — two independent layers:
#
#   _DASRecord (data_model.py) — KEEP AS IS
#       Plain dataclass. Used internally by the GUI (waterfall.py,
#       analysis_dialogs.py, main_window.py). The GUI works with raw
#       numpy arrays (.tr, .dist_m, .time_s) — no xarray needed there.
#       Lightweight, no external dependencies beyond numpy.
#
#   DASdataset (api.py) — MIGRATE: inherit from xr.DataArray, not _DASRecord
#       Public API class. from_dataset(_DASRecord) converts between the two
#       layers (this method already exists). Processing methods registered
#       as xarray accessors (xr.register_dataarray_accessor) or kept as
#       standalone functions in processing.py.
#
# What changes in the migration:
#   - DASdataset.__init__: build a DataArray instead of calling _DASRecord
#   - DASdataset.tr → DASdataset.values (numpy array access)
#   - DASdataset.dist_m → DASdataset.coords["distance"].values
#   - DASdataset.time_s → DASdataset.coords["time"].values
#   - save_npz / save_mat: use .values for the array
#   - from_file() / from_dataset(): return DataArray directly
#   - to_xarray(): return self
#
# What does NOT change:
#   - _DASRecord — stays exactly as is, GUI keeps using it
#   - All processing functions in processing.py, fk_filter.py, rgb.py, msr.py
#     — they operate on numpy arrays, no change needed
#   - DASannotations — independent of DASdataset internals
#   - MSRcube — independent
#
# See: DASdataset.to_xarray() for the target structure.
#      DASCore Patch for a reference implementation.
#      https://docs.xarray.dev/en/stable/internals/extending-xarray.html

class DASdataset(_DASRecord):
    """PUBLIC API — High-level DAS dataset with built-in processing and export methods.

    This is the class users should interact with. The internal dataclass
    ``_DASRecord`` (in ``core/data_model.py``) should never be used directly
    in user code or the public API.

    Inherits all fields from _DASRecord and adds convenience properties
    and methods for signal processing, spectral analysis, and data export.
    All processing methods return a **new** DASdataset instance, leaving
    the original unchanged.

    Parameters
    ----------
    Same as _DASRecord — see dasexplorer.core.data_model._DASRecord.

    Examples
    --------
    >>> from dasexplorer.core.data import DASdataset
    >>> ds = DASdataset.from_file("data.h5", reader="optasense_v1")
    >>> ds_fk = ds.bandpass(10, 30).fk_filter(1400, 3500)
    >>> rgb = ds.rgb(r_band=(1, 5), g_band=(5, 15), b_band=(15, 40))
    >>> ds.save_npz("output/data.npz")
    """

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def dx(self) -> float:
        """Channel spacing [m]."""
        return float(self.dist_m[1] - self.dist_m[0]) if len(self.dist_m) > 1 else 1.0

    @property
    def dt(self) -> float:
        """Sampling interval [s] = 1 / fs_hz."""
        return 1.0 / self.fs_hz

    @property
    def duration_s(self) -> float:
        """Total file duration [s]."""
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0

    @property
    def cable_length_m(self) -> float:
        """Spatial extent of the loaded array [m]."""
        return float(self.dist_m[-1] - self.dist_m[0]) if len(self.dist_m) > 1 else 0.0

    @property
    def nyquist_hz(self) -> float:
        """Nyquist frequency [Hz] = fs_hz / 2."""
        return self.fs_hz / 2.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _replace_tr(self, tr: np.ndarray, processing_tag: str,
                    time_s: np.ndarray = None, fs_hz: float = None,
                    downsample: int = None) -> "DASdataset":
        """Return a new DASdataset with tr replaced and metadata annotated."""
        meta = dict(self.metadata)
        history = list(meta.get("processing", []))
        history.append(processing_tag)
        meta["processing"] = history
        return DASdataset(
            tr=tr,
            dist_m=self.dist_m,
            time_s=time_s if time_s is not None else self.time_s,
            fs_hz=fs_hz if fs_hz is not None else self.fs_hz,
            start_datetime_utc=self.start_datetime_utc,
            filename=self.filename,
            reader=self.reader,
            channel_stride=self.channel_stride,
            channel_offset=self.channel_offset,
            downsample=downsample if downsample is not None else self.downsample,
            metadata=meta,
            units=self.units,
        )

    # ── Processing methods ────────────────────────────────────────────────────

    def bandpass(self, fmin: float, fmax: float, order: int = 5) -> "DASdataset":
        """Apply a zero-phase Butterworth bandpass filter.

        Parameters
        ----------
        fmin : float
            Low-frequency cutoff [Hz].
        fmax : float
            High-frequency cutoff [Hz].
        order : int
            Filter order. Default 5.

        Returns
        -------
        DASdataset
            New instance with filtered tr.
        """
        tr_bp = bandpass_filter(self.tr, self.fs_hz, fmin, fmax, order)
        return self._replace_tr(tr_bp, f"bandpass({fmin},{fmax},order={order})")

    def envelope(self) -> "DASdataset":
        """Compute the Hilbert envelope (instantaneous amplitude).

        Returns
        -------
        DASdataset
            New instance with envelope tr.
        """
        tr_env = hilbert_envelope(self.tr)
        return self._replace_tr(tr_env, "envelope")

    def fk_filter(
        self,
        c_min: float,
        c_max: float,
        fmin: float,
        fmax: float,
        tapering: bool = False,
        gaussian_sigma: float = 40.0,
    ) -> "DASdataset":
        """Apply an F-K bandpass filter.

        Parameters
        ----------
        c_min : float
            Minimum apparent velocity [m/s].
        c_max : float
            Maximum apparent velocity [m/s].
        fmin : float
            Minimum frequency [Hz].
        fmax : float
            Maximum frequency [Hz].
        tapering : bool
            Apply Tukey window before filtering. Default False.
        gaussian_sigma : float
            Gaussian smoothing sigma for filter edges. Default 40.

        Returns
        -------
        DASdataset
            New instance with F-K filtered tr.
        """
        from dasexplorer.core.fk_filter import fk_filter_design, fk_filter_apply
        stride = int(self.channel_stride or 1)
        fk = fk_filter_design(
            trace_shape=self.tr.shape,
            dx=self.dx / stride,
            fs=self.fs_hz,
            c_min=c_min,
            c_max=c_max,
            fmin=fmin,
            fmax=fmax,
            stride=stride,
            gaussian_sigma=gaussian_sigma,
        )
        tr_fk = fk_filter_apply(self.tr, fk, tapering=tapering)
        return self._replace_tr(
            tr_fk,
            f"fk_filter(c={c_min}-{c_max},f={fmin}-{fmax})"
        )


    def detrend(self, mode: str = 'linear') -> "DASdataset":
        """Remove trend from each channel along the time axis.

        Parameters
        ----------
        mode : {'linear', 'constant'}
            ``'linear'`` (default) — remove best-fit linear trend.
            ``'constant'`` — remove the mean (DC offset).

        Returns
        -------
        DASdataset
            New instance with detrended tr.
        """
        from dasexplorer.core.processing import detrend as _detrend
        return self._replace_tr(_detrend(self.tr, mode=mode),
                                f"detrend(mode={mode})")

    def taper(self, alpha: float = 0.05, mode: str = 'tukey') -> "DASdataset":
        """Apply a tapering window to the time edges of each channel.

        Parameters
        ----------
        alpha : float
            Fraction of the trace tapered at each end. Default 0.05.
        mode : {'tukey', 'hann', 'cosine'}
            Window type. Default ``'tukey'``.

        Returns
        -------
        DASdataset
            New instance with tapered tr.
        """
        from dasexplorer.core.processing import taper as _taper
        return self._replace_tr(_taper(self.tr, alpha=alpha, mode=mode),
                                f"taper(alpha={alpha},mode={mode})")

    def normalize(self, mode: str = 'rms') -> "DASdataset":
        """Normalize each channel by its RMS, peak or z-score.

        Parameters
        ----------
        mode : {'rms', 'peak', 'zscore'}
            Normalisation method. Default ``'rms'``.

        Returns
        -------
        DASdataset
            New instance with normalized tr.
        """
        from dasexplorer.core.processing import normalize as _normalize
        return self._replace_tr(_normalize(self.tr, mode=mode),
                                f"normalize(mode={mode})")

    # ── Temporal resampling ───────────────────────────────────────────────────

    def downsample_time(
        self,
        factor: int = None,
        fs_target: float = None,
        mode: str = 'decimate',
    ) -> "DASdataset":
        """Temporally downsample the dataset.

        Parameters
        ----------
        factor : int, optional
            Integer decimation factor. Either ``factor`` or ``fs_target``
            must be provided.
        fs_target : float, optional
            Target sampling frequency [Hz].
        mode : {'decimate', 'simple'}
            ``'decimate'`` (default) — anti-aliasing filter + decimation
            (correct for scientific use). ``'simple'`` — take every N-th
            sample with no pre-filtering.

        Returns
        -------
        DASdataset
            New instance with updated tr, time_s, fs_hz and downsample.
        """
        from dasexplorer.core.processing import downsample_signal
        tr_down, fs_new, q = downsample_signal(
            self.tr, self.fs_hz,
            factor=factor, fs_target=fs_target, mode=mode
        )
        n_new = tr_down.shape[1]
        time_new = np.arange(n_new) / fs_new
        cum_ds = (self.downsample or 1) * q
        return self._replace_tr(
            tr_down,
            f"downsample_time(factor={q},mode={mode})",
            time_s=time_new,
            fs_hz=fs_new,
            downsample=cum_ds,
        )

    def upsample_time(
        self,
        factor: int = None,
        fs_target: float = None,
        mode: str = 'linear',
    ) -> "DASdataset":
        """Temporally upsample the dataset.

        Parameters
        ----------
        factor : int, optional
            Integer upsampling factor. Either ``factor`` or ``fs_target``
            must be provided.
        fs_target : float, optional
            Target sampling frequency [Hz].
        mode : {'linear', 'cubic', 'fft'}
            ``'linear'`` (default) — linear interpolation.
            ``'cubic'`` — cubic spline interpolation.
            ``'fft'`` — FFT-based resampling (best spectral fidelity).

        Returns
        -------
        DASdataset
            New instance with updated tr, time_s, fs_hz.
        """
        from dasexplorer.core.processing import upsample_signal
        tr_up, fs_new, q = upsample_signal(
            self.tr, self.fs_hz,
            factor=factor, fs_target=fs_target, mode=mode
        )
        n_new = tr_up.shape[1]
        time_new = np.arange(n_new) / fs_new
        return self._replace_tr(
            tr_up,
            f"upsample_time(factor={q},mode={mode})",
            time_s=time_new,
            fs_hz=fs_new,
        )

    # ── Representation ────────────────────────────────────────────────────────

    def msr(
        self,
        bands: list,
        percentile: float = 95.0,
        order: int = 5,
    ) -> np.ndarray:
        """Compute the Multispectral Representation (MSR) spectral cube.

        Parameters
        ----------
        bands : list of (fmin, fmax) tuples
            Frequency bands [Hz]. E.g. [(1,5), (5,15), (15,40)].
        percentile : float
            Per-band normalisation percentile. Default 95.
        order : int
            Butterworth filter order. Default 5.

        Returns
        -------
        np.ndarray
            Spectral cube, shape (n_channels, n_time, n_bands),
            float32 in [0, 1].
        """
        from dasexplorer.core.msr import multispectral_representation, MSRcube
        array = multispectral_representation(
            self.tr, self.fs_hz,
            bands=bands,
            percentile=percentile,
            order=order,
        )
        return MSRcube(
            array=array,
            bands=bands,
            dist_m=self.dist_m,
            time_s=self.time_s,
            fs_hz=self.fs_hz,
            percentile=percentile,
        )


    def rgb(
        self,
        r_band: Tuple[float, float] = (1.0, 5.0),
        g_band: Tuple[float, float] = (5.0, 15.0),
        b_band: Tuple[float, float] = (15.0, 40.0),
        percentile: float = 90.0,
        order: int = 5,
    ) -> np.ndarray:
        """Compute the multispectral RGB composite.

        Parameters
        ----------
        r_band : tuple
            (fmin, fmax) for the red channel [Hz].
        g_band : tuple
            (fmin, fmax) for the green channel [Hz].
        b_band : tuple
            (fmin, fmax) for the blue channel [Hz].
        percentile : float
            Percentile for colour scale normalisation. Default 90.
        order : int
            Butterworth filter order. Default 5.

        Returns
        -------
        np.ndarray
            RGB array, shape (n_channels, n_time, 3), dtype uint8.
        """
        from dasexplorer.core.rgb import compute_rgb_composite
        return compute_rgb_composite(
            self.tr, self.fs_hz,
            r_band=r_band,
            g_band=g_band,
            b_band=b_band,
            percentile=percentile,
            order=order,
        )

    # ── Interoperability ──────────────────────────────────────────────────────

    def to_xarray(self):
        """Convert this dataset to an xarray DataArray.

        Returns a DataArray with dimensions ``("distance", "time")``,
        physical coordinates, and all metadata stored as attributes.
        This is the structure DASdataset will be based on in a future
        version — ``to_xarray()`` will eventually return ``self``.

        Requires the optional ``xarray`` package.

        Returns
        -------
        xarray.DataArray
            Shape (n_channels, n_time), dims ("distance", "time"),
            coords: distance [m], time [s].

        Examples
        --------
        >>> da = ds.to_xarray()
        >>> da.sel(distance=slice(20000, 30000)).plot()
        >>> da.to_netcdf("output/das.nc")
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError(
                "xarray is required for to_xarray(). "
                "Install with: pip install xarray"
            )
        attrs = {
            "fs_hz":               float(self.fs_hz),
            "units":               self.units or "",
            "reader":              self.reader or "",
            "filename":            self.filename or "",
            "channel_stride":      int(self.channel_stride or 1),
            "channel_offset":      int(self.channel_offset or 0),
            "start_datetime_utc":  str(self.start_datetime_utc or ""),
        }
        attrs.update(self.metadata or {})
        return xr.DataArray(
            data=self.tr,
            dims=["distance", "time"],
            coords={
                "distance": ("distance", self.dist_m, {"units": "m"}),
                "time":     ("time",     self.time_s, {"units": "s"}),
            },
            attrs=attrs,
            name="strain_rate",
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def save_npz(self, path: str) -> None:
        """Save the dataset to a compressed NumPy archive (.npz).

        Parameters
        ----------
        path : str
            Output file path (e.g. "output/data.npz").
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            tr=self.tr,
            dist_m=self.dist_m,
            time_s=self.time_s,
            fs_hz=np.float32(self.fs_hz),
            units=self.units or "",
            reader=self.reader or "",
            filename=self.filename or "",
            start_datetime_utc=str(self.start_datetime_utc or ""),
        )

    def save_mat(self, path: str) -> None:
        """Save the dataset to a MATLAB file (.mat).

        Parameters
        ----------
        path : str
            Output file path (e.g. "output/data.mat").
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sio.savemat(
            path,
            {
                "tr":                  self.tr,
                "dist_m":              self.dist_m,
                "time_s":              self.time_s,
                "fs_hz":               float(self.fs_hz),
                "units":               self.units or "",
                "reader":              self.reader or "",
                "filename":            self.filename or "",
                "start_datetime_utc":  str(self.start_datetime_utc or ""),
            },
            do_compression=True,
        )

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str, reader: str, **kwargs) -> "DASdataset":
        """Read a DAS file and return a DASdataset instance.

        Parameters
        ----------
        path : str
            Path to the DAS file.
        reader : str
            Reader key (e.g. "optasense_v1", "hdas2.5_v1", "svalbard_v1").
        **kwargs
            Additional arguments passed to read_das_file (stride,
            read_dmin_m, read_dmax_m, num_files, ...).

        Returns
        -------
        DASdataset
        """
        from dasexplorer.core.readers import read_das_file
        ds = read_das_file(path, reader=reader, **kwargs)
        return cls.from_dataset(ds)

    @classmethod
    def from_dataset(cls, ds: _DASRecord) -> "DASdataset":
        """Convert an existing _DASRecord dataclass to a DASdataset instance.

        Parameters
        ----------
        ds : _DASRecord
            Source dataclass instance.

        Returns
        -------
        DASdataset
        """
        return cls(
            tr=ds.tr,
            dist_m=ds.dist_m,
            time_s=ds.time_s,
            fs_hz=ds.fs_hz,
            start_datetime_utc=ds.start_datetime_utc,
            filename=ds.filename,
            reader=ds.reader,
            channel_stride=ds.channel_stride,
            channel_offset=ds.channel_offset,
            metadata=dict(ds.metadata),
            units=ds.units,
        )

    def __repr__(self) -> str:
        proc = self.metadata.get("processing", [])
        proc_str = " → ".join(proc) if proc else "raw"
        return (
            f"DASdataset("
            f"shape=({self.n_dist}, {self.n_time}), "
            f"fs={self.fs_hz:.1f}Hz, "
            f"dx={self.dx:.1f}m, "
            f"duration={self.duration_s:.2f}s, "
            f"processing=[{proc_str}])"
        )


# ══════════════════════════════════════════════════════════════════════════════
# DASAnnotations
# ══════════════════════════════════════════════════════════════════════════════

class DASannotations:
    """High-level annotation container for DAS data.

    Groups all four annotation types (BBox, OBBox, Keypoints, Lines) and
    provides a unified interface for loading, filtering, adding, saving,
    and exporting annotations.

    Examples
    --------
    >>> ann = DASAnnotations.from_stem("path/to/North-C1_2021-11-04T020002Z")
    >>> print(ann)
    DASAnnotations(bbox=12, obb=3, kp=0, line=0)
    >>> for bbox in ann.bbox:
    ...     print(bbox.id, bbox.t0, bbox.t1)
    >>> ann.add_bbox("FW", t0=10.0, t1=20.0, d0=25000, d1=30000)
    >>> ann.save("output/")
    >>> ann.to_yolo("output/yolo/")
    """

    def __init__(self):
        from dasexplorer.core.annotations_model import AnnotationModel, AnnType
        self._models = {
            AnnType.BBOX: AnnotationModel(AnnType.BBOX),
            AnnType.OBB:  AnnotationModel(AnnType.OBB),
            AnnType.KP:   AnnotationModel(AnnType.KP),
            AnnType.LINE: AnnotationModel(AnnType.LINE),
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def bbox(self):
        """List of BBox annotations."""
        from dasexplorer.core.annotations_model import AnnType
        return list(self._models[AnnType.BBOX].annotations)

    @property
    def obb(self):
        """List of OBBox annotations."""
        from dasexplorer.core.annotations_model import AnnType
        return list(self._models[AnnType.OBB].annotations)

    @property
    def kp(self):
        """List of keypoint annotations."""
        from dasexplorer.core.annotations_model import AnnType
        return list(self._models[AnnType.KP].annotations)

    @property
    def line(self):
        """List of line annotations."""
        from dasexplorer.core.annotations_model import AnnType
        return list(self._models[AnnType.LINE].annotations)

    @property
    def n_bbox(self) -> int:
        """Number of BBox annotations."""
        return len(self.bbox)

    @property
    def n_obb(self) -> int:
        """Number of OBBox annotations."""
        return len(self.obb)

    @property
    def n_kp(self) -> int:
        """Number of keypoint annotations."""
        return len(self.kp)

    @property
    def n_line(self) -> int:
        """Number of line annotations."""
        return len(self.line)

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_stem(cls, stem: str) -> "DASannotations":
        """Load all annotation CSV files for a given file stem.

        Automatically loads _bbox.csv, _obb.csv, _kp.csv, _lin.csv
        if they exist next to the stem path.

        Parameters
        ----------
        stem : str
            Path stem of the DAS file, without extension and without
            annotation suffix (e.g. "/data/North-C1_2021-11-04T020002Z").

        Returns
        -------
        DASAnnotations
        """
        from dasexplorer.core.annotations_model import AnnType, ANN_SUFFIX
        obj = cls()
        for ann_type, model in obj._models.items():
            suffix = ANN_SUFFIX[ann_type]
            path = f"{stem}{suffix}"
            if os.path.exists(path):
                model.load(path)
        return obj

    @classmethod
    def from_csv(cls, path: str) -> "DASannotations":
        """Load a single annotation CSV file (auto-detects type from suffix).

        Parameters
        ----------
        path : str
            Path to the CSV file (e.g. "data_bbox.csv").

        Returns
        -------
        DASAnnotations
        """
        from dasexplorer.core.annotations_model import AnnType, ANN_SUFFIX
        obj = cls()
        import os as _os
        for ann_type, suffix in ANN_SUFFIX.items():
            if _os.path.basename(path).endswith(suffix):
                obj._models[ann_type].load(path)
                return obj
        raise ValueError(
            f"Cannot determine annotation type from filename: {path}. "
            f"Expected suffix one of {list(ANN_SUFFIX.values())}."
        )

    # ── Add annotations ───────────────────────────────────────────────────────

    def add_bbox(self, id: str, t0: float, t1: float,
                 d0: float, d1: float, comment: str = "",
                 ds: "DASdataset" = None) -> None:
        """Add a BBox annotation.

        Parameters
        ----------
        id : str
            Event identifier.
        t0, t1 : float
            Time bounds [s].
        d0, d1 : float
            Distance bounds [m].
        comment : str
            Optional comment.
        ds : DASdataset, optional
            Dataset used to compute array indices. If None, indices are set to 0.
        """
        from dasexplorer.core.annotations_model import (
            AnnType, BBoxAnnotation, AnnotationModel
        )
        ti0, ti1, di0, di1 = 0, 0, 0, 0
        nt, nx, downsample = 0, 0, 1
        if ds is not None:
            ti0 = int(np.searchsorted(ds.time_s, t0))
            ti1 = int(np.searchsorted(ds.time_s, t1))
            di0 = int(np.searchsorted(ds.dist_m, d0))
            di1 = int(np.searchsorted(ds.dist_m, d1))
            nt  = ds.n_time
            nx  = ds.n_dist
            downsample = int(ds.channel_stride or 1)
        ann = BBoxAnnotation(
            ann_type="bbox", id=id, comment=comment,
            t0=t0, t1=t1, d0=d0, d1=d1,
            ti0=ti0, ti1=ti1, di0=di0, di1=di1,
            nt=nt, nx=nx, downsample=downsample,
            start_datetime_utc="",
        )
        self._models[AnnType.BBOX].add(ann)

    # ── Filter ────────────────────────────────────────────────────────────────

    def filter(self, id: str = None, t_min: float = None,
               t_max: float = None, d_min: float = None,
               d_max: float = None) -> "DASannotations":
        """Return a new DASAnnotations with matching annotations only.

        Parameters
        ----------
        id : str, optional
            Keep only annotations with this ID.
        t_min, t_max : float, optional
            Keep only annotations overlapping this time range [s].
        d_min, d_max : float, optional
            Keep only annotations overlapping this distance range [m].

        Returns
        -------
        DASAnnotations
        """
        result = DASannotations()
        for ann_type, model in self._models.items():
            for ann in model.annotations:
                if id is not None and ann.id != id:
                    continue
                if t_min is not None and ann.t1 < t_min:
                    continue
                if t_max is not None and ann.t0 > t_max:
                    continue
                if hasattr(ann, 'd0'):
                    if d_min is not None and ann.d1 < d_min:
                        continue
                    if d_max is not None and ann.d0 > d_max:
                        continue
                result._models[ann_type].add(ann)
        return result

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, output_dir: str, stem: str = "annotations") -> None:
        """Save all annotation types to CSV files.

        Parameters
        ----------
        output_dir : str
            Directory where CSV files will be written.
        stem : str
            File stem used as prefix (e.g. "myfile" → "myfile_bbox.csv").
        """
        from dasexplorer.core.annotations_model import ANN_SUFFIX
        os.makedirs(output_dir, exist_ok=True)
        for ann_type, model in self._models.items():
            if len(model) == 0:
                continue
            suffix = ANN_SUFFIX[ann_type]
            path = os.path.join(output_dir, f"{stem}{suffix}")
            model.save(path)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_yolo(self, output_dir: str) -> None:
        """Export all BBox annotations to YOLO format.

        Parameters
        ----------
        output_dir : str
            Output directory for YOLO label files.
        """
        from dasexplorer.core.annotation_export import export_yolo
        from dasexplorer.core.annotations_model import ANN_SUFFIX, AnnType
        os.makedirs(output_dir, exist_ok=True)
        # Export requires a CSV path — save temp CSV and export
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"ann{ANN_SUFFIX[AnnType.BBOX]}.csv")
            self._models[AnnType.BBOX].save(path)
            export_yolo(path, output_dir)

    def to_coco(self, output_dir: str) -> None:
        """Export all BBox annotations to COCO JSON format.

        Parameters
        ----------
        output_dir : str
            Output directory for COCO JSON files.
        """
        from dasexplorer.core.annotation_export import export_coco
        from dasexplorer.core.annotations_model import ANN_SUFFIX, AnnType
        os.makedirs(output_dir, exist_ok=True)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"ann{ANN_SUFFIX[AnnType.BBOX]}.csv")
            self._models[AnnType.BBOX].save(path)
            export_coco(path, output_dir)

    def to_raven(self, output_dir: str, fs_hz: float) -> None:
        """Export all BBox annotations to Raven Pro format.

        Parameters
        ----------
        output_dir : str
            Output directory for Raven selection table files.
        fs_hz : float
            Sampling frequency [Hz], required by the Raven format.
        """
        from dasexplorer.core.annotation_export import export_raven
        from dasexplorer.core.annotations_model import ANN_SUFFIX, AnnType
        os.makedirs(output_dir, exist_ok=True)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"ann{ANN_SUFFIX[AnnType.BBOX]}.csv")
            self._models[AnnType.BBOX].save(path)
            export_raven(path, output_dir, fs_hz=fs_hz)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DASannotations("
            f"bbox={self.n_bbox}, "
            f"obb={self.n_obb}, "
            f"kp={self.n_kp}, "
            f"line={self.n_line})"
        )

    def __len__(self) -> int:
        return self.n_bbox + self.n_obb + self.n_kp + self.n_line
