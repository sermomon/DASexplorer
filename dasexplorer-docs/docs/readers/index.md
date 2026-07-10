# Built-in reading profiles

DASexplorer ships with the following built-in reading profiles. Each profile pairs a **reader function** with a set of default visualisation parameters tuned for a specific dataset.

| Interrogator | Company | Format | Profile |
|---|---|---|---|
| HDAS 2.5 | Aragón Photonics Lab. | `.bin` | [APL HDAS 2.5 (.bin) UPV + APL](hdas.md) |
| OptaSense | Luna Innovations | `.h5` | [Luna Innov. OptaSense (.h5) OOI-RCA 2021](optasense.md) |
| Silixa iDAS | Silixa | `.tdms` | [Silixa iDAS (.tdms) OOI-RCA 2021](idas.md) |
| ASN OptoDAS | Alcatel Submarine Networks | `.hdf5` | [ASN OptoDAS (.hdf5) OOI-RCA 2024](optodas.md) |

DASexplorer can also read pre-processed data distributed in standard scientific formats:

| Dataset | Format | Profile |
|---|---|---|
| Bouffaut et al. 2022 — Svalbard DAS4Whales | `.mat` | [ASN OptoDAS (.mat) Svalbard 2020](svalbard.md) |

!!! warning "HDAS 2.5 — proprietary binary required"
    Reading `.bin` files requires the `hdas_reader*.pyd` (Windows) or `hdas_reader*.so` (Linux) binary provided by Aragón Photonics Lab. This binary is not distributed via PyPI and must be placed manually in `dasexplorer/tools/apl/` inside your installation directory. Run `python -c "import dasexplorer; print(dasexplorer.__file__)"` to find it.

## Adding a new reader

To add a new reading profile, see [Adding a reading profile](../profiles.md).

If you add a reader and would like it included in a future release, open a Pull Request on [GitHub](https://github.com/sermomon/dasexplorer) or send an email to [sermomon@upv.es](mailto:sermomon@upv.es) with some sample data.
