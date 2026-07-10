"""
Reader registry for DAS Explorer.

Each interrogator/format has its own reader module in core/readers_lib/.
This file only wires them together via the READERS dict.

To add a new reader:
  1. Create core/readers_lib/<name>.py with the read_xxx() function
  2. Import it here and add it to READERS
  3. Create a matching profile in cfg/config.json
"""

from dasexplorer.core.data_model import DASDataset
from dasexplorer.core.readers_lib.hdas      import read_hdas25_v1
from dasexplorer.core.readers_lib.optasense import read_optasense_v1
from dasexplorer.core.readers_lib.idas      import read_idas_v1
from dasexplorer.core.readers_lib.optodas   import read_optodas_v1, read_optodas_v2
from dasexplorer.core.readers_lib.processed import read_svalbard_v1


# ── Reader registry ───────────────────────────────────────────────────────────
# Map reader keys (used in config.json profiles) to their reader functions.
# READER_TYPES and READER_LABELS are derived automatically from this dict
# and from the profile labels in cfg/config.json — no manual lists to maintain.

READERS = {
    "hdas2.5_v1":   read_hdas25_v1,
    "optasense_v1": read_optasense_v1,
    "silixa_v1":    read_idas_v1,
    "optodas_v1":   read_optodas_v1,
    "optodas_v2":   read_optodas_v2,
    "svalbard_v1":  read_svalbard_v1,
}

READER_TYPES: list = list(READERS.keys())


def _build_reader_labels() -> list:
    """Return human-readable labels for each reader key, sourced from config.json profiles."""
    try:
        from dasexplorer.core.config import get_all_profiles
        profiles = get_all_profiles()
        label_map = {}
        for p in profiles.values():
            r = p.get("reader", "")
            if r and r not in label_map:
                label_map[r] = p.get("label", r)
        return [label_map.get(k, k) for k in READER_TYPES]
    except Exception:
        return list(READER_TYPES)


READER_LABELS: list = _build_reader_labels()


# ── Synthetic dataset (for testing and UI development) ────────────────────────

def generate_synthetic_dataset(
    n_dist: int = 600,
    n_time: int = 3000,
    fs_hz: float = 50.0,
    dx_m: float = 10.0,
) -> DASDataset:
    """Generate a synthetic DAS dataset for testing the UI without real data."""
    import numpy as np
    rng = np.random.default_rng(42)
    dist_m = np.arange(n_dist) * dx_m
    time_s = np.arange(n_time) / fs_hz
    freqs  = np.linspace(1.0, fs_hz / 2.0 * 0.8, n_dist)
    tr = np.array([
        np.sin(2 * np.pi * f * time_s) + 0.1 * rng.standard_normal(n_time)
        for f in freqs
    ], dtype=np.float32)
    return DASDataset(
        tr=tr, dist_m=dist_m, time_s=time_s, fs_hz=fs_hz,
        filename="synthetic.bin", reader="synthetic", units="DC",
        metadata={"dx_m": dx_m},
    )


# ── Dispatch ──────────────────────────────────────────────────────────────────

def read_das_file(path: str, reader: str, **kwargs) -> DASDataset:
    """
    Dispatch to the appropriate reader function.

    Parameters
    ----------
    path : str
        Path to the DAS file.
    reader : str
        Key from the READERS dict (e.g. "hdas2.5_v1").
    **kwargs
        Forwarded to the specific reader function.

    Returns
    -------
    DASDataset
    """
    if reader not in READERS:
        raise ValueError(
            f"Unknown reader '{reader}'. "
            f"Available: {list(READERS)}"
        )
    return READERS[reader](path, **kwargs)
