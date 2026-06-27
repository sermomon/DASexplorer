"""
Configuration loader for DAS Explorer.

Reads config.json from the cfg/ directory. Supports the profile-based
schema (config.json > profiles) introduced in v0.9.31, with automatic
backwards-compatibility for the legacy interrogator-based schema.
"""

import json
import os
from typing import Any, Optional

# importlib.resources finds cfg/config.json whether installed as a wheel
# or run directly from source.
try:
    from importlib.resources import files as _res_files
    _CONFIG_PATH = str(_res_files("dasexplorer").joinpath("cfg/config.json"))
except Exception:
    _CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg", "config.json")

# Built-in fallback defaults used when config.json is absent or corrupt.
_PROFILE_DEFAULTS: dict = {
    "label":             "HDAS 2.5 — Default",
    "interrogator":      "hdas2.5",
    "file_extensions":   [".bin"],
    "num_files":         1,
    "tmin_s":            None,
    "tmax_s":            None,
    "dmin_m":            None,
    "dmax_m":            None,
    "vmin":              0,
    "vmax":              12,
    "colormap":          "Rainbow",
    "fmin_hz":           1.0,
    "fmax_hz":           None,
    "fmax_offset_hz":    0.01,
    "stride":            1,
    "envelope":          False,
    "fk_cmin_ms":        100.0,
    "fk_cmax_ms":        100000.0,
    "fk_fmin_hz":        1.0,
    "fk_fmax_hz":        None,
    "fk_fmax_offset_hz": 0.01,
    "fk_envelope":       False,
    "default_view":      "raw",
    "rgb_rmin_hz":       1.0,
    "rgb_rmax_hz":       5.0,
    "rgb_gmin_hz":       5.0,
    "rgb_gmax_hz":       15.0,
    "rgb_bmin_hz":       15.0,
    "rgb_bmax_hz":       40.0,
    "rgb_percentile":    90.0,
}


def _load() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        return {}
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[config] Warning: could not read {_CONFIG_PATH}: {exc}. Using defaults.")
        return {}


def _save(cfg: dict) -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
    except OSError as exc:
        print(f"[config] Warning: could not write {_CONFIG_PATH}: {exc}")


# ── Module-level singleton ────────────────────────────────────────────────────
_cfg = _load()


def _migrate_legacy(cfg: dict) -> dict:
    """
    Convert old interrogator-based config to the new profile-based schema.
    Called transparently if config.json has 'interrogators' but not 'profiles'.
    """
    if "profiles" in cfg or "interrogators" not in cfg:
        return cfg

    from dasexplorer.core.readers import INTERROGATOR_TYPES, INTERROGATOR_LABELS
    ext_map = {
        "hdas2.5":   [".bin"],
        "optasense": [".h5", ".hdf5"],
        "optodas":   [".hdf5"],
    }

    profiles = {}
    for intr_key, intr_label in zip(INTERROGATOR_TYPES, INTERROGATOR_LABELS):
        old = cfg["interrogators"].get(intr_key, {})
        pid = f"{intr_key.replace('.', '')}_default"
        label = intr_label.split(" [")[0] + " — Default"
        p = {k: v for k, v in _PROFILE_DEFAULTS.items()}
        p.update({k: v for k, v in old.items() if not k.startswith("_")})
        p["label"]           = label
        p["interrogator"]    = intr_key
        p["file_extensions"] = ext_map.get(intr_key, [])
        profiles[pid] = p

    migrated = dict(cfg)
    migrated["profiles"] = profiles
    migrated["default_profile"] = list(profiles.keys())[0]
    # Keep legacy key for reference but don't use it
    return migrated


def get_all_profiles() -> dict:
    """Return the full profiles dict from config (possibly after migration)."""
    cfg = _migrate_legacy(_cfg)
    return cfg.get("profiles", {"default": dict(_PROFILE_DEFAULTS)})


def get_profile(profile_key: str) -> dict:
    """
    Return the config dict for a given profile key.
    Falls back to built-in defaults for any missing key.
    """
    profiles = get_all_profiles()
    user = profiles.get(profile_key, {})
    return {**_PROFILE_DEFAULTS, **{k: v for k, v in user.items()
                                    if not k.startswith("_")}}


def get_default_profile_key() -> str:
    """Return the key of the default profile."""
    cfg = _migrate_legacy(_cfg)
    profiles = cfg.get("profiles", {})
    default  = cfg.get("default_profile", "")
    if default in profiles:
        return default
    if profiles:
        return next(iter(profiles))
    return "default"


def get_ui_defaults() -> dict:
    """Return the UI defaults block from config."""
    ui_defaults = {
        "colormap":              "Rainbow",
        "panel_width_fraction":  0.2,
        "theme":                 "dark",
    }
    user = _cfg.get("ui", {})
    return {**ui_defaults, **{k: v for k, v in user.items() if not k.startswith("_")}}


def set_ui_theme(theme_name: str) -> None:
    """Persist the chosen UI theme to config.json."""
    global _cfg
    if "ui" not in _cfg:
        _cfg["ui"] = {}
    _cfg["ui"]["theme"] = theme_name
    _save(_cfg)


def save_profiles(profiles: dict, default_profile: str) -> None:
    """Write updated profiles back to config.json."""
    global _cfg
    cfg = dict(_cfg)
    cfg["profiles"] = profiles
    cfg["default_profile"] = default_profile
    _save(cfg)
    _cfg = cfg


def reload() -> None:
    """Reload config from disk."""
    global _cfg
    _cfg = _load()


# ── Backwards-compatibility shim ─────────────────────────────────────────────
def get_interrogator_defaults(interrogator: str) -> dict:
    """
    Legacy API: return defaults for the first profile whose interrogator
    matches the given key.  Used by ConfigurationProfileDialog internals.
    """
    for p in get_all_profiles().values():
        if p.get("interrogator") == interrogator:
            return {**_PROFILE_DEFAULTS, **{k: v for k, v in p.items()
                                            if not k.startswith("_")}}
    return dict(_PROFILE_DEFAULTS)
