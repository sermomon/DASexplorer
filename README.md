<div align="center">
  <img src="https://raw.githubusercontent.com/sermomon/DASexplorer/main/dasexplorer/icons/icon_2.ico" alt="DASexplorer Logo" width="200"/>

  <h1></h1>

  <p>
    <strong>A desktop application for visualization, analysis and annotation of Distributed Acoustic Sensing data</strong>
  </p>

  <p>
<a href="https://github.com/sermomon/DASexplorer/releases"><img src="https://img.shields.io/badge/release-v1.0.2-red" alt="Latest Release"/></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPLv3-blue" alt="GPL v3 License"/></a>
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

  <img src="https://raw.githubusercontent.com/sermomon/DASexplorer/main/dasexplorer/icons/screenshot.png" alt="DASexplorer main window" width="900"/>

</div>

---

## Overview

**DASexplorer** is an open-source desktop application built for researchers working with Distributed Acoustic Sensing (DAS) data. It provides an integrated environment for loading, visualising, filtering, and annotating DAS recordings from multiple interrogator systems, without requiring any prior programming knowledge.

DASexplorer was designed for efficient interactive navigation of large DAS datasets. Built on PyQt5 and pyqtgraph, it uses spatial decimation, parameter-based caching and smart rendering strategies to handle long acquisitions with minimal memory overhead. The application is interrogator-agnostic: a profile-based configuration system maps each acquisition system to its reader, file extensions and default visualisation parameters, making it straightforward to add support for new instruments.

Key design principles:

- **No-code** — the full analysis workflow is accessible through an intuitive GUI, including F-K filtering, RGB representation, spectral analysis, spectrogram, apparent velocity, phase and Hilbert envelope.
- **Multi-interrogator** — natively supports HDAS 2.5 (Aragón Photonics), OptaSense (Luna Innovations) and OptoDAS (ASN/ Alcatel Subsea Networks) file formats; extensible to new systems.
- **Research-ready** — exports data and annotations in formats directly usable by machine learning and bioacoustic analysis pipelines (NPZ, MAT, YOLO, COCO JSON, Raven Pro).

## Installation

### From PyPI (recommended)

```bash
pip install dasexplorer
dasexplorer
```

### From source (GitHub)

This tool is currently under active development, so new features are being added every day. Clone the updated repository and install in editable mode:

```bash
git clone https://github.com/sermomon/DASexplorer.git
cd DASexplorer
pip install -e .
dasexplorer
```

## Examples

DASexplorer is currently capable of reading data from the following interrogators: HDAS 2.5, OptaSence, iDAS, and OptoDAS. You can try out the tool with public datasets such as:

**Rapid: A Community Test of Distributed Acoustic Sensing on the Ocean Observatories Initiative Regional Cabled Array:** https://doi.org/10.58046/5J60-FJ89  

Interrogator: Luna Innovations OptaSense

Data example: [North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5](http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5)

Format: HDF5 (.h5)

![Example OptaSence](https://raw.githubusercontent.com/sermomon/DASexplorer/main/dasexplorer/icons/optasence_example_v1.png)

**Rapid: A Community Test of Distributed Acoustic Sensing on the Ocean Observatories Initiative Regional Cabled Array:** https://doi.org/10.58046/5J60-FJ89

Interrogator: Silixa iDAS

Data example: [OOIPacCity_UTC_20211101_163806.423.tdms](http://piweb.ooirsn.uw.edu/das/data/Silixa/DAS/North65km/acquisition/OOIPacCity_UTC_20211101_163806.423.tdms)

Format: TDMS (.tdms)

![Example iDAS](https://raw.githubusercontent.com/sermomon/DASexplorer/main/dasexplorer/icons/idas_example_v1.png)

**Longyearbyen, Svalbard (Norway) 2020: DAS4Whales: Svalbard distributed acoustic sensing dataset for baleen whale monitoring** https://zenodo.org/records/5823343

Interrogator: Alcatel Submarine Network OptoDAS

Data example: [20200627_052441_ch10001_to_ch15000_whale_raw_L160s.mat](https://zenodo.org/records/5823343/files/20200627_052441_ch10001_to_ch15000_whale_raw_L160s.mat?download=1)

Format: MATLAB (.mat) processed

![Example Svalbard](https://raw.githubusercontent.com/sermomon/DASexplorer/main/dasexplorer/icons/svalbard_example_v1.png)

> **Note:** Reading HDAS 2.5 `.bin` files requires the proprietary `hdas_reader` binary (provided by Aragón Photonics), which must be placed in `dasexplorer/tools/apl/`. OptaSense and OptoDAS support works out of the box.

## Citation

If you use DASexplorer in your research, please cite: 

Morell-Monzó, S. (2026). DASexplorer (Version 1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21032549

```bibtex
@software{morellmonzo2026dasexplorer,
  author       = {Morell-Monzó, Sergio},
  title        = {DASexplorer},
  version      = {1.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21032550},
  url          = {https://doi.org/10.5281/zenodo.21032549}
}
```

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21032550-blue)](https://doi.org/10.5281/zenodo.21032549)
