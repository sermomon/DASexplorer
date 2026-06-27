<div align="center">
  <img src="dasexplorer/icons/icon_2.ico" alt="DASexplorer Logo" width="200"/>

  <h1></h1>

  <p>
    <strong>A desktop application for visualization, analysis and annotation of Distributed Acoustic Sensing data</strong>
  </p>

  <p>
<a href="https://github.com/sermomon/DASexplorer/releases"><img src="https://img.shields.io/badge/release-v1.0.0-red" alt="Latest Release"/></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
    <a href="https://doi.org/10.xxxx/joss.xxxxx"><img src="https://img.shields.io/badge/JOSS-paper-orange" alt="JOSS Paper"/></a>
    <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="Platform"/>
  </p>

  <p>
    <a href="#installation">Installation</a> ·
    <a href="#quickstart">Quickstart</a> ·
    <a href="#features">Features</a> ·
    <a href="#supported-interrogators">Interrogators</a> ·
    <a href="#annotation-export">Annotation Export</a> ·
    <a href="#contributing">Contributing</a> ·
    <a href="#citation">Citation</a>
  </p>
  <br/>

  <img src="dasexplorer/icons/screenshot.png" alt="DAS Explorer main window" width="900"/>

</div>

---

## Overview

**DASexplorer** is an open-source desktop application built for researchers working with Distributed Acoustic Sensing (DAS) data. It provides an integrated environment for loading, visualising, filtering, and annotating DAS recordings from multiple interrogator systems, without requiring any prior programming knowledge.

DASexplorer was designed for efficient interactive navigation of large DAS datasets. Built on PyQt5 and pyqtgraph, it uses spatial decimation, parameter-based caching and smart rendering strategies to handle long acquisitions with minimal memory overhead. The application is interrogator-agnostic: a profile-based configuration system maps each acquisition system to its reader, file extensions and default visualisation parameters, making it straightforward to add support for new instruments.

Key design principles:

- **No-code** — the full analysis workflow is accessible through an intuitive GUI.
- **Multi-interrogator** — natively supports HDAS 2.5 (Aragón Photonics), OptaSense (Luna Innovations) and OptoDAS (ASN/ Alcatel Subsea Networks) file formats; extensible to new systems.
- **Research-ready** — exports data and annotations in formats directly usable by machine learning and bioacoustic analysis pipelines (NPZ, MAT, YOLO, COCO JSON, Raven Pro).

---

## Features

### 🌊 Data Visualisation

| Feature | Description |
|---|---|
| **Raw waterfall** | Time–distance plot with interactive colour scale and zoom |
| **F-K filtered waterfall** | Wavenumber–frequency filtering with configurable apparent velocity and frequency bounds |
| **RGB Multispectral (MSR-DAS)** | Three-channel composite image mapping independent frequency bands to R, G, B channels — a physically interpretable representation for feature extraction |
| **Hilbert envelope** | Instantaneous amplitude display for Raw and F-K views, independently configurable |
| **Colourmap selector** | Rainbow, Viridis, Turbo, Magma, Seismic, Grayscale |
| **Synchronised views** | All three waterfall tabs share the same time/distance cursor and colour scale controls |

### 🔍 Signal Analysis

Seven interactive analysis dialogs, accessible via right-click on any annotation:

| Dialog | Description |
|---|---|
| **Spectrogram** | Time–frequency spectrogram with NPERSEG, overlap, NFFT, zoom and colour scale controls |
| **Spectral analysis** | Per-channel FFT magnitude spectra with a highlighted average spectrum; log/linear scale |
| **Signal (time domain)** | Waveform display with fixed reference channels for multi-channel comparison |
| **Signal (frequency domain)** | FFT magnitude of a single channel with log/linear scale |
| **Hilbert envelope** | Instantaneous amplitude of the signal |
| **Instantaneous phase** | Unwrapped instantaneous phase |
| **Velocity estimation** | Interactive point-picking on the waterfall for apparent velocity estimation with linear regression and R² score |

### ✏️ Annotation

- **Interactive bounding-box drawing** on any waterfall view — click and drag to define time–distance events
- **Annotation table** with ID, time range, distance range, and comment fields
- **Per-annotation analysis** — any annotation can be sent directly to any of the seven analysis tools
- **CSV export** — annotations saved as structured CSV files that preserve all physical coordinates (`t0`, `t1`, `d0`, `d1`, `ti0`, `ti1`, `di0`, `di1`, `nt`, `nx`, `fs_hz`, `start_datetime_utc`)
- **Previous / Next** file navigation with annotation persistence across files in the same directory

### ⚙️ Configuration

All default parameters are stored in `cfg/config.json` and applied automatically at startup — no code editing required. Parameters configurable per interrogator type:

```jsonc
{
  "hdas2.5": {
    "tmin_s": null,          // Default time range (null = full file)
    "tmax_s": null,
    "dmin_m": null,          // Default distance range
    "dmax_m": null,
    "vmin": 0,               // Colour scale limits
    "vmax": 12,
    "colormap": "Rainbow",
    "fmin_hz": 1.0,          // Bandpass filter
    "fmax_hz": null,         // null → Nyquist − fmax_offset_hz
    "stride": 1,             // Spatial decimation
    "envelope": false,       // Hilbert envelope on load
    "fk_cmin_ms": 100.0,     // F-K velocity range
    "fk_fmax_hz": null,
    "fk_envelope": false,    // Independent FK envelope
    "rgb_rmin_hz": 1.0,      // MSR-DAS band definitions
    "rgb_rmax_hz": 5.0,
    ...
  }
}
```

A graphical **Configuration Profile** editor (`Settings → Configuration Profile`) provides a table-based interface to edit all parameters without touching the JSON file directly.

### 🔄 Batch Conversion

#### Data conversion (`Conversion → Batch Conversion → Data`)

Convert entire directories of raw DAS files to portable formats:

- **NPZ** (NumPy compressed) — preserves all array data and metadata; directly loadable with `np.load()`
- **MAT** (MATLAB) — compatible with MATLAB, SciPy and any HDF5-aware tool

All conversions retain the full metadata block: `fs_hz`, `dx_m`, `start_datetime_utc`, `interrogator`, `downsample`, `units`.

#### Annotation export (`Conversion → Batch Conversion → Annotations`)

Convert DAS Explorer annotation CSVs to three standard formats:

| Format | Description | Use case |
|---|---|---|
| **YOLO** | One `.txt` per image group + `classes.txt` | Object detection training (YOLOv5/v8/v10, RT-DETR) |
| **COCO JSON** | Single structured JSON per CSV | Detectron2, MMDetection, co-DETR |
| **Raven** | Tab-separated Selection Table (`_raven.csv`) | Raven Pro, PAMGuard, BIANET-C workflows |

Output filenames are derived from the input CSV stem with format-specific suffixes (`_coco.json`, `_raven.csv`) to prevent collisions with the original annotation files. YOLO label files (`.txt`) are unambiguously distinguishable by extension. String class IDs are mapped to integer indices automatically; a `classes.txt` file listing the mapping is always written alongside the YOLO labels.

Annotations can be grouped by:
- `start_datetime_utc` — one output file per acquisition window (recommended)
- `id` — one output file per event class
- all-in-one — single output file

---

## Supported Interrogators

| Interrogator | File format | Notes |
|---|---|---|
| **Aragón Photonics HDAS 2.5** | `.bin` | Requires `hdas_reader` binary extension (`.pyd` on Windows, `.so` on Linux); CPython 3.10 only |
| **OptaSense** | `.h5` / `.hdf5` | Standard HDF5; reads `fs`, `dx` and timestamps from file metadata |
| **NPZ (DAS Explorer)** | `.npz` | Re-importable via `File → Import from NPZ` |
| **MAT (DAS Explorer)** | `.mat` | Re-importable via `File → Import from MAT` |

Adding support for new interrogators requires implementing a single reader function in `core/readers.py` and registering the file extension in `FILE_EXTENSIONS`.

---

## Installation

### Requirements

- Python 3.10 or later
- PyQt5 ≥ 5.15
- PyQtGraph ≥ 0.13
- NumPy, SciPy, pandas

### From source (recommended during development)

```bash
git clone https://github.com/IGIC-UPV/DASExplorer.git
cd DASExplorer
pip install -e .
```

### Running the application

```bash
python main.py
```

Or, once installed as a package:

```bash
das-explorer
```

---

## Quickstart

1. **Open a file** — `File → Open file…` or drag a file onto the file list panel.
2. **Select your interrogator** — HDAS 2.5 or OptaSense. The combo box remembers your last choice.
3. **Adjust the view** — use the left panel to set the time/distance range, colour scale, bandpass filter and stride.
4. **Explore** — switch between Raw, F-K and RGB tabs. The RGB tab computes a multispectral composite using the frequency bands defined in the left panel.
5. **Annotate** — click **Annotate** in the left panel, then click-and-drag on the waterfall to draw a bounding box. Assign an ID and optional comment.
6. **Analyse** — right-click any annotation to open Spectrogram, Spectral Analysis, Signal, or Velocity dialogs.
7. **Export** — `File → Save CSV` to save annotations, or use `Conversion → Batch Conversion` to export data and annotations in bulk.

---

## Project Structure

```
DASExplorer/
├── main.py                    # Entry point
├── version.py                 # Version file
├── cfg/
│   └── config.json            # User-editable configuration
├── icons/
│   ├── icon_1.ico             # Window icon
│   ├── icon_2.ico             # Taskbar icon
│   └── logo.png               # Project logo
├── core/
│   ├── readers.py             # File I/O (HDAS, OptaSense, OptoDAS, ...)
│   ├── annotations.py         # Annotation data model and CSV I/O
│   ├── annotation_export.py   # YOLO / COCO / Raven exporters
│   ├── analysis.py            # Signal processing (FFT, spectrogram, F-K)
│   ├── rgb.py                 # RGB composite computation
│   ├── data_model.py          # DASDataset dataclass
│   └── config.py              # Configuration loader
└── gui/
    ├── main_window.py         # Main application window
    ├── waterfall.py           # Waterfall plot widget
    ├── analysis_dialogs.py    # Signal analysis dialogs (A–G)
    ├── annotation_widget.py   # Annotation panel
    ├── batch_data_dialog.py   # Batch data conversion
    ├── batch_annotations_dialog.py  # Batch annotation export
    ├── config_profile_dialog.py     # Configuration editor
    ├── tab_bar.py             # Custom RGB tab indicator
    └── theme.py               # Dark / Light theme system
```

---

## The MSR-DAS Framework

The RGB Multispectral Representation of DAS data (MSR-DAS) is a physically interpretable visualisation framework introduced in:

> Morell-Monzó, S., Diego-Tortosa, D., Pérez-Arjona, I., & Espinosa, V. (2025). *Multispectral Representation of Distributed Acoustic Sensing Data: A Framework for Physically Interpretable Feature Extraction and Visualization*. arXiv:2604.07290. Submitted to *Expert Systems with Applications*.

Each channel of the RGB composite corresponds to a user-defined frequency band, filtered with a zero-phase Butterworth bandpass filter and normalised by a per-band percentile. The resulting three-channel image can be used directly as input to standard image-based deep learning architectures.

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

```bash
# Fork the repository, then:
git checkout -b feature/your-feature-name
# ... make your changes ...
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a pull request on GitHub
```

### Adding a new interrogator

1. Implement a reader function `read_<interrogator>(path, **kwargs) -> DASDataset` in `core/readers.py`
2. Add the interrogator key and display label to `INTERROGATOR_TYPES` and `INTERROGATOR_LABELS`
3. Register its file extensions in `FILE_EXTENSIONS` (in `gui/main_window.py`)
4. Add a default parameter block in `cfg/config.json` and `core/config.py`

### Adding a new annotation export format

1. Implement the converter in `core/annotation_export.py` following the `export_yolo` / `export_coco` / `export_raven` pattern
2. Add a button and handler in `gui/batch_annotations_dialog.py`

---

## Roadmap

- [ ] **Image export pipeline** — export DAS arrays as PNG/TIFF/NPZ images with configurable preprocessing (bandpass → F-K → MSR-DAS), with optional sliding-window tiling and paired annotation export
- [ ] **Live mode** — real-time display from an active interrogator stream
- [ ] **Plugin system** — user-defined analysis modules loaded at runtime
- [ ] **Pascal VOC export** — additional annotation format for legacy pipelines

---

## Authors and Affiliations

**Sergio Morell-Monzó** ¹ [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8883--2618-green?logo=orcid)](https://orcid.org/0000-0001-8883-2618)
— Lead developer, sermomon@upv.es

¹ Instituto de Investigación para la Gestión Integrada de Zonas Costeras (IGIC-UPV), Universitat Politècnica de València, Gandia, Spain

---

This work was supported by MAR.IA - Modelos de Inteligencia Artificial para el Análisis de Datos de Acústica Submarina (PAID-10-25) Financed by RRHH Universitat Politècnica de València.

## Citation

If you use DASexplorer in your research, please cite:

```bibtex
@article{morellmonzo2025dasexplorer,
  title   = {{DAS Explorer}: An open-source desktop application for visualization,
             analysis and annotation of Distributed Acoustic Sensing data},
  author  = {Morell-Monzó, Sergio and Diego-Tortosa, Dídac and
             Pérez-Arjona, Isabel and Espinosa, Víctor},
  journal = {Journal of Open Source Software},
  year    = {2025},
  doi     = {10.xxxx/joss.xxxxx}
}
```

If you use the MSR-DAS multispectral representation, please also cite:

```bibtex
@article{morellmonzo2025msrdas,
  title   = {Multispectral Representation of Distributed Acoustic Sensing Data:
             A Framework for Physically Interpretable Feature Extraction and Visualization},
  author  = {Morell-Monzó, Sergio and Diego-Tortosa, Dídac and
             Pérez-Arjona, Isabel and Espinosa, Víctor},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {arXiv:2604.07290}
}
```

---

## License

DAS Explorer is released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>
    Developed at the <a href="https://igic.upv.es">Underwater Acoustics Group (IGIC)</a> ·
    Universitat Politècnica de València ·
    <a href="mailto:sermomon@upv.es">sermomon@upv.es</a>
  </sub>
</div>
