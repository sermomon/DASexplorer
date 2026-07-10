# What is a reading profile?

A **reading profile** is the combination of two things that together tell DASexplorer how to load and display data from a specific acquisition system:

1. **A reader function** — a Python function that knows how to open a particular file format and return a standardised `DASDataset` object.
2. **A profile entry in `config.json`** — a block of configuration that links the profile to its reader function and defines the default visualisation parameters (colour scale, frequency range, F-K filter settings, RGB bands, etc.).

Together they make DASexplorer interrogator-agnostic: adding support for a new system is a matter of writing a reader and registering a profile, without changing the rest of the application.

---

## The reader function

The reader function handles everything that is specific to one file format and one acquisition system: opening the file, extracting the raw data array, converting it to physical units, reading the sampling frequency and channel distances, and parsing the UTC timestamp. It accepts a standard set of arguments:

```python
def my_reader(
    path: str,
    stride: Optional[int] = None,         # channel decimation factor
    read_dmin_m: Optional[float] = None,  # spatial crop start [m]
    read_dmax_m: Optional[float] = None,  # spatial crop end [m]
    **kwargs,                             # format-specific extras
) -> DASDataset:
```

It always returns a `DASDataset` — a standardised container that the rest of the application (waterfall display, annotation tools, analysis dialogs, export functions) consumes without needing to know anything about the original file format.

Format-specific parameters that not all readers share — such as `num_files` for HDAS 2.5 or `selected_channels_m` for OptaSense — are passed through `**kwargs` and documented in the reader's docstring.

Reader functions live in `core/readers_lib/`, one file per interrogator family (e.g. `hdas.py`, `optasense.py`, `idas.py`). They are imported and registered in `core/readers.py` via the `READERS` dictionary.

---

## The profile entry

The profile entry in `config.json` has two roles:

**1. Linking to the reader** — the `"reader"` field identifies which function to call when a file is loaded with this profile:

```json
"reader": "my_reader"
```

This must match a key in the `READERS` dictionary exactly.

**2. Default visualisation parameters** — everything else in the profile defines how the data looks when first loaded: colour scale limits, bandpass filter range, F-K velocity bounds, RGB frequency bands, and so on. The user can change any of these interactively from the left panel; the profile values are just the starting point.

```json
"myreader_v1": {
    "label": "MyInterrogator (.ext) Description",
    "reader": "my_reader",
    "file_extensions": [".ext"],
    "vmin": 0.0,
    "vmax": 0.5,
    "fmin_hz": 1.0,
    ...
}
```

Profiles are managed from **Settings → Configuration Profiles** in the GUI, or by editing `config.json` directly.

---

## Summary

| Component | Where it lives | What it does |
|---|---|---|
| Reader function | `core/readers_lib/<name>.py` | Opens the file, returns `DASDataset` |
| Reader registration | `core/readers.py` (`READERS` dict) | Maps the reader key to the function |
| Profile entry | `cfg/config.json` | Links the key to default display settings |

To add a new interrogator, see [Adding a reading profile](profiles.md).
