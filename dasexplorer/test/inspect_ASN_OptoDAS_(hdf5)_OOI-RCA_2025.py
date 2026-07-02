#%%
"""
Inspect an OOI-RCA 2024 OptoDAS (ASN) HDF5 file to discover its structure.
Run this in a notebook against your downloaded file.
"""

import h5py
import numpy as np

path = "C:/Lab/projects/DASexplorer/dasexplorer/datasets/optodas/ooi-rca2024/203935.hdf5"

#%%

def visit(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}  shape={obj.shape} dtype={obj.dtype}")
    else:
        print(f"{indent}{name}/")
    if obj.attrs:
        for k, v in obj.attrs.items():
            val = v if np.ndim(v) == 0 else f"array{np.shape(v)}"
            print(f"{indent}  @{k}: {val}")

with h5py.File(path, "r") as f:
    print("=== Full tree ===")
    f.visititems(visit)

    print("\n=== Root attrs ===")
    for k, v in f.attrs.items():
        print(f"  @{k}: {v}")
# %%

with h5py.File(path, "r") as f:
    print("=== Key header values ===")
    print(f"data.shape:              {f['data'].shape}")
    print(f"data.dtype:               {f['data'].dtype}")
    print(f"header/dt:                {f['header/dt'][()]}")
    print(f"header/dx:                {f['header/dx'][()]}")
    print(f"header/time:              {f['header/time'][()]}")
    print(f"header/gaugeLength:       {f['header/gaugeLength'][()]}")
    print(f"header/dataScale:         {f['header/dataScale'][()]}")
    print(f"header/dataType:          {f['header/dataType'][()]}")
    print(f"header/unit:              {f['header/unit'][()]}")
    print(f"header/name:              {f['header/name'][()]}")
    print(f"header/instrument:        {f['header/instrument'][()]}")
    print(f"header/experiment:        {f['header/experiment'][()]}")
    print(f"header/sensitivities:     {f['header/sensitivities'][()]}")
    print(f"header/sensitivityUnits:  {f['header/sensitivityUnits'][()]}")
    print(f"header/sensorType:        {f['header/sensorType'][()]}")
    print(f"header/dimensionSizes:    {f['header/dimensionSizes'][()]}")
    print(f"header/dimensionNames:    {f['header/dimensionNames'][()]}")
    print(f"header/dimensionUnits:    {f['header/dimensionUnits'][()]}")
    print()
    print(f"cableSpec/sensorDistances[:5]:   {f['cableSpec/sensorDistances'][:5]}")
    print(f"cableSpec/sensorDistances[-5:]:  {f['cableSpec/sensorDistances'][-5:]}")
    print(f"cableSpec/refractiveIndex:       {f['cableSpec/refractiveIndex'][()]}")
    print(f"cableSpec/fiberOverLength:       {f['cableSpec/fiberOverLength'][()]}")
    print(f"cableSpec/zeta:                  {f['cableSpec/zeta'][()]}")
    print()
    print(f"acqSpec/rate:             {f['acqSpec/rate'][()]}")
    print()
    print(f"data[:2,:5]:\n{f['data'][:2, :5]}")
    print(f"data min/max: {f['data'][()].min()}, {f['data'][()].max()}")

    import os
    print(f"\nFilename: {os.path.basename(path)}")
# %%
