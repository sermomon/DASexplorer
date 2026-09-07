# dasexplorer_api_example.py
#
# Comprehensive example of the DASexplorer Python API.
# Covers: reading files, synthetic data, signal processing,
# F-K filtering, RGB, annotations, and data export.
#
# Run with:
#   python dasexplorer_api_example.py

import os
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "dasexplorer_output"
DATA_FILE  = None           # Set to a real .h5 / .bin / .mat path to use it
READER_KEY = "optasense_v1"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import dasexplorer
from dasexplorer.api import DASdataset, DASannotations
from dasexplorer.core.processing import bandpass_filter, hilbert_envelope
from dasexplorer.core.fk_filter import fk_filter_design, fk_filter_apply
from dasexplorer.core.rgb import compute_rgb_composite
from dasexplorer.core.readers import read_das_file, generate_synthetic_dataset

print(f"DASexplorer {dasexplorer.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. READ A DAS FILE  (or generate synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

if DATA_FILE is not None:
    print(f"\nReading {DATA_FILE} ...")
    ds = DASdataset.from_file(
        DATA_FILE,
        reader=READER_KEY,
        stride=2,
        read_dmin_m=20000.0,
        read_dmax_m=65000.0,
    )
else:
    print("\nGenerating synthetic dataset ...")
    ds = DASdataset.from_dataset(
        generate_synthetic_dataset(n_dist=500, n_time=3000, fs_hz=200.0, dx_m=10.0)
    )

print(ds)
print(f"  dx          = {ds.dx} m")
print(f"  dt          = {ds.dt:.5f} s")
print(f"  nyquist     = {ds.nyquist_hz} Hz")
print(f"  duration    = {ds.duration_s:.2f} s")
print(f"  cable       = {ds.cable_length_m:.0f} m")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CREATE A DASdataset FROM SCRATCH
# ─────────────────────────────────────────────────────────────────────────────

print("\nCreating DASdataset from scratch ...")
n_ch, n_t = 200, 1000
fs_hz, dx_m = 100.0, 5.0

tr = np.random.randn(n_ch, n_t).astype(np.float32)
t  = np.arange(n_t) / fs_hz
k0 = 20.0 / 1500.0
for i in range(n_ch):
    tr[i] += 0.3 * np.sin(2 * np.pi * 20.0 * t - 2 * np.pi * k0 * i * dx_m)

from dasexplorer.core.data_model import DASDataset
custom_ds = DASdataset.from_dataset(DASDataset(
    tr=tr,
    dist_m=np.arange(n_ch) * dx_m,
    time_s=np.arange(n_t) / fs_hz,
    fs_hz=fs_hz,
    start_datetime_utc=datetime(2024, 6, 15, 12, 0, 0),
    filename="synthetic_finwhale.npy",
    reader="custom",
    downsample=None,
    channel_offset=0,
    metadata={"description": "Synthetic fin whale 20 Hz call at 1500 m/s"},
    units="nanostrain",
))
print(custom_ds)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROCESSING — STANDALONE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\nProcessing with standalone functions ...")

tr_bp  = bandpass_filter(ds.tr, ds.fs_hz, fmin=10.0, fmax=30.0)
tr_env = hilbert_envelope(tr_bp)
fk     = fk_filter_design(ds.tr.shape, dx=ds.dx, fs=ds.fs_hz,
                           c_min=1400.0, c_max=3500.0, fmin=10.0, fmax=30.0)
tr_fk  = fk_filter_apply(tr_bp, fk, tapering=False)
rgb    = compute_rgb_composite(ds.tr, ds.fs_hz,
                                r_band=(1.0, 5.0),
                                g_band=(5.0, 15.0),
                                b_band=(15.0, 40.0),
                                percentile=90.0)

print(f"  bandpass:   {tr_bp.shape}  {tr_bp.dtype}")
print(f"  envelope:   {tr_env.shape}  {tr_env.dtype}")
print(f"  fk:         {tr_fk.shape}  {tr_fk.dtype}")
print(f"  rgb:        {rgb.shape}  {rgb.dtype}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PROCESSING — METHOD CHAINING
# ─────────────────────────────────────────────────────────────────────────────

print("\nProcessing with DASdataset method chaining ...")

result = (
    ds
    .bandpass(10, 30)
    .fk_filter(1400, 3500, fmin=10, fmax=30)
    .envelope()
)
print(result)

rgb2 = ds.rgb(r_band=(1, 5), g_band=(5, 15), b_band=(15, 40), percentile=90)
print(f"  rgb shape:  {rgb2.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\nWorking with DASannotations ...")

ann = DASannotations()
ann.add_bbox("FW", t0=10.0, t1=20.0,
             d0=float(ds.dist_m[50]), d1=float(ds.dist_m[150]),
             comment="fin whale", ds=ds)
ann.add_bbox("FW", t0=30.0, t1=40.0,
             d0=float(ds.dist_m[80]), d1=float(ds.dist_m[180]),
             comment="fin whale 2", ds=ds)
print(ann)

for bbox in ann.bbox:
    print(f"  [{bbox.id}]  t={bbox.t0:.1f}-{bbox.t1:.1f}s  "
          f"d={bbox.d0:.0f}-{bbox.d1:.0f}m  comment={bbox.comment}")

fw_late = ann.filter(id="FW", t_min=25.0)
print(f"  filter(t_min=25s): {fw_late}")

ann.save(OUTPUT_DIR, stem="example")
ann2 = DASannotations.from_stem(os.path.join(OUTPUT_DIR, "example"))
print(f"  reloaded: {ann2}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. EXPORT ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\nExporting annotations ...")
ann.to_yolo(os.path.join(OUTPUT_DIR, "yolo"))
ann.to_coco(os.path.join(OUTPUT_DIR, "coco"))
ann.to_raven(os.path.join(OUTPUT_DIR, "raven"), fs_hz=ds.fs_hz)
print(f"  YOLO  → {OUTPUT_DIR}/yolo")
print(f"  COCO  → {OUTPUT_DIR}/coco")
print(f"  Raven → {OUTPUT_DIR}/raven")

# ─────────────────────────────────────────────────────────────────────────────
# 8. EXPORT DATA
# ─────────────────────────────────────────────────────────────────────────────

print("\nExporting data ...")
ds.save_npz(os.path.join(OUTPUT_DIR, "das_raw.npz"))
result.save_npz(os.path.join(OUTPUT_DIR, "das_fk_env.npz"))
ds.save_mat(os.path.join(OUTPUT_DIR, "das_raw.mat"))
result.save_mat(os.path.join(OUTPUT_DIR, "das_fk_env.mat"))
np.savez_compressed(os.path.join(OUTPUT_DIR, "das_rgb.npz"),
                    rgb=rgb, dist_m=ds.dist_m, time_s=ds.time_s)
print(f"  das_raw.npz     OK")
print(f"  das_fk_env.npz  OK")
print(f"  das_raw.mat     OK")
print(f"  das_fk_env.mat  OK")
print(f"  das_rgb.npz     OK")

# ─────────────────────────────────────────────────────────────────────────────
# 9. BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

print("\nBatch processing example ...")
batch_dir = os.path.join(OUTPUT_DIR, "batch")
os.makedirs(batch_dir, exist_ok=True)

for i in range(3):
    synth = DASdataset.from_dataset(
        generate_synthetic_dataset(n_dist=300, n_time=2000, fs_hz=200.0, dx_m=10.0)
    )
    synth.filename = f"file_{i:03d}.h5"
    out_path = os.path.join(batch_dir, f"file_{i:03d}_fk.npz")
    synth.bandpass(10, 30).fk_filter(1400, 3500, 10, 30).save_npz(out_path)
    print(f"  file_{i:03d} → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Output files:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for f in sorted(files):
        full = os.path.join(root, f)
        size = os.path.getsize(full) / 1024
        print(f"  {full}  ({size:.1f} KB)")
print("=" * 60)
print("Done.")
