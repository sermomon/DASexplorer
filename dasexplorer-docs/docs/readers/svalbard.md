# ASN OptoDAS (.mat) Svalbard 2020

**Interrogator:** Alcatel Submarine Networks (ASN) OptoDAS (processed)  
**Format:** MATLAB (`.mat`)  
**Profile key:** `svalbard_v1`  
**Dataset:** [Bouffaut et al. 2022 — Svalbard (2020) DAS4Whales](https://doi.org/10.5281/zenodo.5823343)

This profile reads pre-processed data from the Svalbard DAS4Whales dataset
(Bouffaut et al., 2022, *Frontiers in Marine Science*). The data is distributed
as MATLAB `.mat` files already converted to nanostrain, cropped to the channel
range of interest, and annotated with fin whale and blue whale vocalisation events.
This is not raw interrogator output — it is a processed scientific dataset.

## Reader function

::: dasexplorer.core.readers_lib.processed.read_svalbard_v1
    options:
      show_source: true
      heading_level: 3
