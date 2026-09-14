"""
core/geometry.py — Fiber geometry loading and processing.

Reads geographic coordinates (WGS84) from GeoJSON, Shapefile, CSV or TXT,
computes geodetic distances along the cable, and interpolates coordinates
to the DAS channel distance axis.

All coordinate inputs must be in WGS84 (EPSG:4326).
Geodetic distances are computed on the WGS84 ellipsoid using pyproj.Geod,
which is correct for long cables (hundreds of km) without needing to choose
a projected CRS.
"""

import os
import numpy as np

__all__ = [
    "load_geometry",
    "compute_geodetic_distance",
    "interpolate_geometry_to_channels",
    "FiberGeometry",
]


class FiberGeometry:
    """Container for fiber optic cable geographic coordinates.

    Parameters
    ----------
    lons : np.ndarray
        Longitude [degrees, WGS84].
    lats : np.ndarray
        Latitude [degrees, WGS84].
    elevs : np.ndarray, optional
        Elevation or depth [m]. Positive up (elevation) or negative
        down (depth) — convention follows the input file.
    dist_m : np.ndarray
        Cumulative geodetic distance along the cable [m].
    crs : str
        Coordinate reference system. Always ``'EPSG:4326'``.
    source_path : str, optional
        Path of the file from which coordinates were loaded.
    """

    def __init__(self, lons, lats, elevs, dist_m, crs="EPSG:4326",
                 source_path=None):
        self.lons        = np.asarray(lons,  dtype=np.float64)
        self.lats        = np.asarray(lats,  dtype=np.float64)
        self.elevs       = (np.asarray(elevs, dtype=np.float64)
                            if elevs is not None else None)
        self.dist_m      = np.asarray(dist_m, dtype=np.float64)
        self.crs         = crs
        self.source_path = source_path
        self.is_3d       = elevs is not None

    @property
    def n_points(self):
        return len(self.lons)

    @property
    def total_length_m(self):
        return float(self.dist_m[-1]) if len(self.dist_m) > 0 else 0.0

    def __repr__(self):
        return (
            f"FiberGeometry(n_points={self.n_points}, "
            f"length={self.total_length_m/1000:.2f} km, "
            f"3D={self.is_3d}, crs='{self.crs}')"
        )


def compute_geodetic_distance(lons: np.ndarray,
                              lats: np.ndarray) -> np.ndarray:
    """Compute cumulative geodetic distance along a sequence of points.

    Uses the WGS84 ellipsoid via ``pyproj.Geod`` — correct for long cables
    without requiring a projected CRS.

    Parameters
    ----------
    lons : np.ndarray
        Longitude [degrees].
    lats : np.ndarray
        Latitude [degrees].

    Returns
    -------
    np.ndarray
        Cumulative distance [m], starting at 0.
    """
    try:
        from pyproj import Geod
    except ImportError:
        raise ImportError(
            "pyproj is required for geodetic distance computation. "
            "Install with: pip install pyproj"
        )
    geod     = Geod(ellps="WGS84")
    n        = len(lons)
    dist     = np.zeros(n, dtype=np.float64)
    if n < 2:
        return dist
    _, _, d  = geod.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    dist[1:] = np.cumsum(np.abs(d))
    return dist


def _read_geojson(path: str):
    """Read a GeoJSON file and return (lons, lats, elevs_or_None)."""
    import json
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)

    lons, lats, elevs = [], [], []
    has_z = False

    features = gj.get("features", [])
    if not features:
        # Maybe it's a bare geometry
        features = [{"geometry": gj}]

    for feat in features:
        geom = feat.get("geometry", feat)
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if gtype == "Point":
            coords = [coords]
        elif gtype in ("LineString", "MultiPoint"):
            pass
        elif gtype == "MultiLineString":
            coords = [pt for line in coords for pt in line]
        else:
            continue

        for c in coords:
            lons.append(c[0])
            lats.append(c[1])
            if len(c) > 2:
                elevs.append(c[2])
                has_z = True
            else:
                elevs.append(np.nan)

    return (np.array(lons), np.array(lats),
            np.array(elevs) if has_z else None)


def _read_shapefile(path: str):
    """Read a Shapefile and return (lons, lats, elevs_or_None)."""
    try:
        import shapefile
    except ImportError:
        raise ImportError(
            "pyshp is required to read Shapefiles. "
            "Install with: pip install pyshp"
        )
    sf    = shapefile.Reader(path)
    lons, lats, elevs = [], [], []
    has_z = False

    for shape in sf.shapes():
        pts  = shape.points
        zs   = getattr(shape, "z", None)
        for i, (x, y) in enumerate(pts):
            lons.append(x)
            lats.append(y)
            if zs:
                elevs.append(zs[i])
                has_z = True
            else:
                elevs.append(np.nan)

    return (np.array(lons), np.array(lats),
            np.array(elevs) if has_z else None)


def _read_csv_txt(path: str):
    """Read a CSV/TXT file with columns: lon, lat [, elev].

    Accepts comma, semicolon, space or tab delimiters.
    First row may be a header.
    """
    import csv
    lons, lats, elevs = [], [], []
    has_z = False

    with open(path, encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        # Detect delimiter
        sniffer  = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample, delimiters=",;\t ")
        except csv.Error:
            dialect = csv.excel
        has_header = sniffer.has_header(sample)
        reader = csv.reader(f, dialect)
        if has_header:
            next(reader)
        for row in reader:
            row = [r.strip() for r in row if r.strip()]
            if len(row) < 2:
                continue
            try:
                lons.append(float(row[0]))
                lats.append(float(row[1]))
                if len(row) >= 3:
                    elevs.append(float(row[2]))
                    has_z = True
                else:
                    elevs.append(np.nan)
            except ValueError:
                continue

    return (np.array(lons), np.array(lats),
            np.array(elevs) if has_z else None)


def load_geometry(path: str,
                  fmt: str = "auto") -> FiberGeometry:
    """Load fiber cable geometry from a file.

    Parameters
    ----------
    path : str
        Path to the geometry file.
    fmt : {'auto', 'geojson', 'shapefile', 'csv', 'txt'}
        File format. ``'auto'`` detects from the file extension.

    Returns
    -------
    FiberGeometry
        Loaded geometry with geodetic distances pre-computed.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format cannot be determined or the file has fewer than
        2 valid points.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Geometry file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if fmt == "auto":
        if ext in (".geojson", ".json"):
            fmt = "geojson"
        elif ext in (".shp",):
            fmt = "shapefile"
        elif ext in (".csv",):
            fmt = "csv"
        elif ext in (".txt", ".xyz", ".dat"):
            fmt = "txt"
        else:
            raise ValueError(
                f"Cannot determine format from extension '{ext}'. "
                f"Pass fmt='geojson'|'shapefile'|'csv'|'txt' explicitly."
            )

    if fmt == "geojson":
        lons, lats, elevs = _read_geojson(path)
    elif fmt == "shapefile":
        lons, lats, elevs = _read_shapefile(path)
    elif fmt in ("csv", "txt"):
        lons, lats, elevs = _read_csv_txt(path)
    else:
        raise ValueError(f"Unknown format '{fmt}'.")

    if len(lons) < 2:
        raise ValueError(
            f"Geometry file must contain at least 2 points, "
            f"got {len(lons)}."
        )

    dist_m = compute_geodetic_distance(lons, lats)

    return FiberGeometry(
        lons=lons, lats=lats, elevs=elevs,
        dist_m=dist_m, crs="EPSG:4326",
        source_path=path,
    )


def interpolate_geometry_to_channels(
    geom: FiberGeometry,
    dist_das: np.ndarray,
    geometry_offset_m: float = 0.0,
    warn: bool = True,
) -> FiberGeometry:
    """Interpolate geometry to the DAS channel distance axis.

    Maps the geometry points onto the ``dist_m`` axis of the DAS dataset
    using linear interpolation, accounting for a distance offset between
    the data origin (channel 0) and the geometry origin.

    Parameters
    ----------
    geom : FiberGeometry
        Source geometry loaded with :func:`load_geometry`.
    dist_das : np.ndarray
        Along-cable distance axis of the DAS dataset [m].
    geometry_offset_m : float
        Distance offset [m] between the data origin (channel 0) and the
        geometry origin (first geometry point). Positive means the
        geometry starts further along the cable than the data origin
        (e.g. 5000 if 5 km of cable on land precede the geometry).
        Negative means the geometry starts before the data origin.

    Returns
    -------
    FiberGeometry
        New FiberGeometry with coordinates interpolated at every
        channel position in ``dist_das``.
    """
    import warnings
    d_src = geom.dist_m
    # Shift data distances into geometry coordinate system
    dist_shifted = dist_das - geometry_offset_m

    # Warn if any channels fall before the geometry start
    n_before = int(np.sum(dist_shifted < d_src[0]))
    if warn and n_before > 0:
        warnings.warn(
            f"{n_before} channel(s) have distances below the geometry start "
            f"(dist_das[0]={dist_das[0]:.0f} m, "
            f"geometry_offset_m={geometry_offset_m:.0f} m, "
            f"geometry_start={d_src[0]:.0f} m after offset). "
            "Coordinates for those channels will be extrapolated "
            "from the first geometry point and may not be reliable. "
            "Consider increasing geometry_offset_m.",
            UserWarning, stacklevel=2
        )

    # Warn if any channels fall beyond the geometry end
    n_after = int(np.sum(dist_shifted > d_src[-1]))
    if warn and n_after > 0:
        warnings.warn(
            f"{n_after} channel(s) have distances beyond the geometry end "
            f"(dist_das[-1]={dist_das[-1]:.0f} m, "
            f"geometry_end={d_src[-1] + geometry_offset_m:.0f} m). "
            "Coordinates for those channels will be extrapolated "
            "from the last geometry point and may not be reliable.",
            UserWarning, stacklevel=2
        )

    lons_i  = np.interp(dist_shifted, d_src, geom.lons)
    lats_i  = np.interp(dist_shifted, d_src, geom.lats)
    elevs_i = (np.interp(dist_shifted, d_src, geom.elevs)
               if geom.is_3d else None)

    return FiberGeometry(
        lons=lons_i, lats=lats_i, elevs=elevs_i,
        dist_m=dist_das.copy(),
        crs=geom.crs,
        source_path=geom.source_path,
    )
