"""
core/map_view.py — Generate interactive Leaflet maps for DASexplorer.

Produces a self-contained HTML string (via folium) that can be loaded
into a QWebEngineView. Requires the optional ``folium`` package.
"""

__all__ = ["build_fiber_map", "BASEMAPS", "load_basemaps"]

_BASEMAPS_DEFAULT_PATH = None  # resolved at first call


def _find_basemaps_path() -> str:
    """Locate cfg/basemaps/basemaps.json relative to this package."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    # gui/ -> dasexplorer/ -> cfg/basemaps/basemaps.json
    pkg_root = os.path.dirname(here)
    return os.path.join(pkg_root, "cfg", "basemaps", "basemaps.json")


def load_basemaps(path: str = None) -> dict:
    """Load basemap definitions from a JSON file.

    Parameters
    ----------
    path : str, optional
        Path to a basemaps JSON file. If None, uses the default
        ``cfg/basemaps/basemaps.json`` inside the package.

    Returns
    -------
    dict
        Ordered dict of basemap name → config dict with keys:
        ``tiles``, ``attr``, ``type`` ('xyz' or 'wms'),
        and optionally ``layers`` (WMS), ``description``.
    """
    import json
    if path is None:
        path = _find_basemaps_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# BASEMAPS — loaded from cfg/basemaps/basemaps.json at import time.
# Add new basemaps by editing that file — no code changes needed.
try:
    BASEMAPS = load_basemaps()
except Exception as _e:
    import warnings
    warnings.warn(f"Could not load basemaps.json: {_e}. Using minimal fallback.")
    BASEMAPS = {
        "OpenStreetMap": {
            "tiles": "OpenStreetMap",
            "attr": None,
            "type": "xyz",
        }
    }


def build_fiber_map(
    coords_lon,
    coords_lat,
    coords_z=None,
    sensed_lon=None,
    sensed_lat=None,
    line_color: str = "#ff2222",
    full_cable_color: str = "#444444",
    line_weight: int = 3,
    line_opacity: float = 0.85,
    basemap: str = "OpenStreetMap",
    show_endpoints: bool = True,
) -> str:
    """Generate a self-contained Leaflet HTML map showing the fiber cable.

    Parameters
    ----------
    coords_lon, coords_lat : array-like
        Full cable geometry longitude/latitude [degrees, WGS84].
        Drawn as a dashed black line.
    coords_z : array-like, optional
        Elevation / depth [m]. Shown in endpoint popups if provided.
    sensed_lon, sensed_lat : array-like, optional
        Sensed portion of the cable (determined by read_dmin_m/read_dmax_m
        and the loaded dataset). Drawn as a solid coloured line on top.
        If None, the full geometry is drawn in colour (no dashed overlay).
    line_color : str
        Hex or named colour for the sensed cable line. Default ``'#00aaff'``.
    line_weight : int
        Line width in pixels. Default 3.
    line_opacity : float
        Line opacity [0, 1]. Default 0.85.
    basemap : str
        Base map layer name. One of :data:`BASEMAPS`.
        Default ``'OpenStreetMap'``.
    show_endpoints : bool
        If True, place markers at the first and last sensed point.

    Returns
    -------
    str
        Self-contained HTML string ready to load in a QWebEngineView.
    """
    try:
        import folium
    except ImportError:
        raise ImportError(
            "folium is required for the Map view. "
            "Install with: pip install folium"
        )

    import numpy as np
    lons = np.asarray(coords_lon, dtype=np.float64)
    lats = np.asarray(coords_lat, dtype=np.float64)

    if len(lons) < 2:
        raise ValueError("At least 2 coordinate points are required.")

    # Centre and bounds
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))

    # Build base map
    bm = BASEMAPS.get(basemap, BASEMAPS["OpenStreetMap"])
    if bm.get("type") == "wms":
        # WMS basemaps: start with blank base, add WMS as overlay
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=None,
        )
        folium.raster_layers.WmsTileLayer(
            url=bm["tiles"],
            name=basemap,
            layers=bm.get("layers", ""),
            fmt="image/png",
            transparent=False,
            attr=bm.get("attr", ""),
            show=True,
        ).add_to(m)
    elif bm["attr"]:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=bm["tiles"],
            attr=bm["attr"],
        )
    else:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles=bm["tiles"],
        )

    # Add all basemap layers as layer control options
    for name, cfg in BASEMAPS.items():
        if name == basemap:
            continue
        if cfg.get("type") == "wms":
            folium.raster_layers.WmsTileLayer(
                url=cfg["tiles"],
                name=name,
                layers=cfg.get("layers", ""),
                fmt="image/png",
                transparent=True,
                attr=cfg.get("attr", ""),
                show=False,
            ).add_to(m)
        elif cfg["attr"]:
            folium.TileLayer(
                tiles=cfg["tiles"],
                attr=cfg["attr"],
                name=name,
                show=False,
            ).add_to(m)
        else:
            folium.TileLayer(
                tiles=cfg["tiles"],
                name=name,
                show=False,
            ).add_to(m)

    # Full cable geometry — dashed black line (total geometry extent)
    has_sensed = (sensed_lon is not None and sensed_lat is not None
                  and len(sensed_lon) > 1)
    full_locations = [[float(lats[i]), float(lons[i])] for i in range(len(lons))]

    if has_sensed:
        # Draw full geometry as dashed line
        folium.PolyLine(
            locations=full_locations,
            color=full_cable_color,
            weight=2,
            opacity=0.7,
            dash_array="8 6",
            tooltip="Full cable geometry",
        ).add_to(m)
        # Draw sensed portion as solid coloured line
        s_lons = np.asarray(sensed_lon, dtype=np.float64)
        s_lats = np.asarray(sensed_lat, dtype=np.float64)
        sensed_locations = [[float(s_lats[i]), float(s_lons[i])]
                            for i in range(len(s_lons))]
        folium.PolyLine(
            locations=sensed_locations,
            color=line_color,
            weight=line_weight,
            opacity=line_opacity,
            tooltip="Sensed portion",
        ).add_to(m)
        draw_lons, draw_lats = s_lons, s_lats
    else:
        # No sensed subset — draw full geometry in colour
        folium.PolyLine(
            locations=full_locations,
            color=line_color,
            weight=line_weight,
            opacity=line_opacity,
            tooltip="DAS fiber cable",
        ).add_to(m)
        draw_lons, draw_lats = lons, lats

    # Endpoint markers on the active (sensed) line
    if show_endpoints and len(draw_lons) >= 2:
        for idx, label in [(0, "Sensed start" if has_sensed else "Cable start"),
                           (-1, "Sensed end" if has_sensed else "Cable end")]:
            popup_lines = [
                f"<b>{label}</b>",
                f"Lon: {draw_lons[idx]:.5f}°",
                f"Lat: {draw_lats[idx]:.5f}°",
            ]
            if coords_z is not None and not has_sensed:
                zs = np.asarray(coords_z, dtype=np.float64)
                popup_lines.append(f"Z: {zs[idx]:.1f} m")
            folium.CircleMarker(
                location=[float(draw_lats[idx]), float(draw_lons[idx])],
                radius=3,
                color=line_color,
                fill=True,
                fill_color=line_color,
                fill_opacity=0.9,
                weight=1.5,
                popup=folium.Popup("<br>".join(popup_lines), max_width=200),
                tooltip=label,
            ).add_to(m)

    # Fit map to cable extent
    m.fit_bounds([
        [float(lats.min()), float(lons.min())],
        [float(lats.max()), float(lons.max())],
    ])


    return m.get_root().render()


def build_no_geometry_html() -> str:
    """Return a simple HTML page shown when no geometry is loaded."""
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {
    background: #1e1e1e;
    color: #888;
    font-family: sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
    flex-direction: column;
    gap: 12px;
  }
  .icon { font-size: 48px; }
  .title { font-size: 18px; color: #aaa; }
  .sub { font-size: 13px; color: #666; text-align: center; max-width: 360px; }
</style>
</head>
<body>
  <div class="icon">🗺️</div>
  <div class="title">No geometry loaded</div>
  <div class="sub">Add a geometry file path in the profile configuration
  and enable <em>Load geometry</em> to display the fiber cable on the map.</div>
</body>
</html>"""
