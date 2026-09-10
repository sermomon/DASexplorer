# dasexplorer_api_example.py
#
# Comprehensive example of the DASexplorer Python API.
#
# PUBLIC API — use these:
#   from dasexplorer.api import DASdataset, DASannotations
#   from dasexplorer.core.processing import bandpass_filter, hilbert_envelope,
#       detrend, taper, normalize, downsample_signal, upsample_signal
#   from dasexplorer.core.fk_filter import fk_filter_design, fk_filter_apply
#   from dasexplorer.core.rgb import compute_rgb_composite
#   from dasexplorer.core.readers import read_das_file, generate_synthetic_dataset
#   from dasexplorer.core.io_formats import read_npz, read_mat
#
# PRIVATE — do NOT use directly:
#   dasexplorer.core.data_model._DASRecord  (internal dataclass)
#   dasexplorer.core.data_model.DASDataset  (alias for _DASRecord, GUI only)
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
from dasexplorer.core.processing import (
    bandpass_filter, hilbert_envelope,
    detrend, taper, normalize,
    downsample_signal, upsample_signal,
)
from dasexplorer.core.fk_filter import fk_filter_design, fk_filter_apply
from dasexplorer.core.rgb import compute_rgb_composite
from dasexplorer.core.readers import read_das_file, generate_synthetic_dataset
from dasexplorer.core.io_formats import read_npz, read_mat

print(f"DASexplorer {dasexplorer.__version__}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. READ A DAS FILE  (or generate synthetic data)
# ─────────────────────────────────────────────────────────────────────────────

if DATA_FILE is not None:
    print(f"\nReading {DATA_FILE} ...")
    # read_das_file returns a DASdataset directly
    ds = read_das_file(
        DATA_FILE,
        reader=READER_KEY,
        stride=2,
        read_dmin_m=20000.0,
        read_dmax_m=65000.0,
    )
else:
    print("\nGenerating synthetic dataset ...")
    # generate_synthetic_dataset also returns a DASdataset directly
    ds = generate_synthetic_dataset(n_dist=500, n_time=3000, fs_hz=200.0, dx_m=10.0)

print(ds)
print(f"  shape       = {ds.tr.shape}  (n_channels x n_time)")
print(f"  dx          = {ds.dx} m")
print(f"  dt          = {ds.dt:.5f} s")
print(f"  nyquist     = {ds.nyquist_hz} Hz")
print(f"  duration    = {ds.duration_s:.2f} s")
print(f"  cable       = {ds.cable_length_m:.0f} m")
print(f"  channel_stride = {ds.channel_stride}")
print(f"  downsample     = {ds.downsample}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CREATE A DASdataset FROM SCRATCH
#    Use DASdataset directly — do NOT use _DASRecord or DASDataset
# ─────────────────────────────────────────────────────────────────────────────

print("\nCreating DASdataset from scratch ...")
n_ch, n_t = 200, 1000
fs_hz_syn, dx_m_syn = 100.0, 5.0

tr_syn = np.random.randn(n_ch, n_t).astype(np.float32)
t_syn  = np.arange(n_t) / fs_hz_syn
k0     = 20.0 / 1500.0
for i in range(n_ch):
    tr_syn[i] += 0.3 * np.sin(2 * np.pi * 20.0 * t_syn - 2 * np.pi * k0 * i * dx_m_syn)

# Correct way: pass all fields to _DASRecord via DASdataset — no need to
# import _DASRecord. Use DASdataset directly with the dataclass fields.
custom_ds = DASdataset(
    tr=tr_syn,
    dist_m=np.arange(n_ch) * dx_m_syn,
    time_s=t_syn,
    fs_hz=fs_hz_syn,
    start_datetime_utc=datetime(2024, 6, 15, 12, 0, 0),
    filename="synthetic_finwhale.npy",
    reader="custom",
    channel_stride=None,
    downsample=None,
    channel_offset=0,
    metadata={"description": "Synthetic fin whale 20 Hz call at 1500 m/s"},
    units="nanostrain",
)
print(custom_ds)
print(f"  dx={custom_ds.dx}m  nyquist={custom_ds.nyquist_hz}Hz")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PROCESSING — STANDALONE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\nProcessing with standalone functions ...")

tr_dt  = detrend(ds.tr, mode='linear')          # remove linear trend
tr_tap = taper(tr_dt, alpha=0.05, mode='tukey') # taper edges
tr_nm  = normalize(tr_tap, mode='rms')           # normalize by RMS
tr_bp  = bandpass_filter(tr_nm, ds.fs_hz, fmin=10.0, fmax=30.0)
tr_env = hilbert_envelope(tr_bp)
fk     = fk_filter_design(tr_bp.shape, dx=ds.dx, fs=ds.fs_hz,
                           c_min=1400.0, c_max=3500.0, fmin=10.0, fmax=30.0)
tr_fk  = fk_filter_apply(tr_bp, fk, tapering=False)
rgb    = compute_rgb_composite(ds.tr, ds.fs_hz,
                                r_band=(1.0, 5.0),
                                g_band=(5.0, 15.0),
                                b_band=(15.0, 40.0),
                                percentile=90.0)

print(f"  detrend:    {tr_dt.shape}  {tr_dt.dtype}")
print(f"  taper:      {tr_tap.shape}  {tr_tap.dtype}")
print(f"  normalize:  {tr_nm.shape}  {tr_nm.dtype}")
print(f"  bandpass:   {tr_bp.shape}  {tr_bp.dtype}")
print(f"  envelope:   {tr_env.shape}  {tr_env.dtype}")
print(f"  fk:         {tr_fk.shape}  {tr_fk.dtype}")
print(f"  rgb:        {rgb.shape}  {rgb.dtype}")

# downsample / upsample standalone
tr_d, fs_d, q = downsample_signal(ds.tr, ds.fs_hz, factor=4, mode='decimate')
tr_u, fs_u, _ = upsample_signal(tr_d, fs_d, fs_target=ds.fs_hz, mode='fft')
print(f"  downsample: {tr_d.shape}  fs={fs_d}Hz (factor={q})")
print(f"  upsample:   {tr_u.shape}  fs={fs_u}Hz")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PROCESSING — METHOD CHAINING (recommended)
# ─────────────────────────────────────────────────────────────────────────────

print("\nProcessing with DASdataset method chaining ...")

result = (
    ds
    .detrend(mode='linear')
    .taper(alpha=0.05, mode='tukey')
    .bandpass(10, 30)
    .normalize(mode='rms')
    .fk_filter(1400, 3500, fmin=10, fmax=30)
    .envelope()
)
print(result)
print(f"  processing history: {result.metadata['processing']}")

# Downsample + upsample via methods
ds_50  = ds.downsample_time(fs_target=50.0)          # decimate to 50 Hz
ds_50s = ds.downsample_time(factor=4, mode='simple') # simple (no anti-aliasing)
ds_200 = ds_50.upsample_time(fs_target=ds.fs_hz, mode='fft')
print(f"  downsample(50Hz):  fs={ds_50.fs_hz}Hz  n={ds_50.n_time}  downsample={ds_50.downsample}")
print(f"  upsample(200Hz):   fs={ds_200.fs_hz}Hz  n={ds_200.n_time}")

# RGB from method
rgb2 = ds.rgb(r_band=(1, 5), g_band=(5, 15), b_band=(15, 40), percentile=90)
print(f"  rgb shape: {rgb2.shape}")

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
          f"d={bbox.d0:.0f}-{bbox.d1:.0f}m  "
          f"ti={bbox.ti0}-{bbox.ti1}  di={bbox.di0}-{bbox.di1}")

# Filter annotations
fw_late = ann.filter(id="FW", t_min=25.0)
print(f"  filter(id=FW, t_min=25s): {fw_late}")

# Save and reload
ann.save(OUTPUT_DIR, stem="example")
ann2 = DASannotations.from_stem(os.path.join(OUTPUT_DIR, "example"))
print(f"  reloaded: {ann2}")

# Load single CSV
ann3 = DASannotations.from_csv(os.path.join(OUTPUT_DIR, "example_bbox.csv"))
print(f"  from_csv: {ann3}")

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

# Reload from NPZ and MAT
ds_npz = read_npz(os.path.join(OUTPUT_DIR, "das_raw.npz"))
ds_mat = read_mat(os.path.join(OUTPUT_DIR, "das_raw.mat"))
print(f"  read_npz: {ds_npz}")
print(f"  read_mat: {ds_mat}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

print("\nBatch processing example ...")
batch_dir = os.path.join(OUTPUT_DIR, "batch")
os.makedirs(batch_dir, exist_ok=True)

for i in range(3):
    # generate_synthetic_dataset returns DASdataset directly
    synth = generate_synthetic_dataset(n_dist=300, n_time=2000, fs_hz=200.0, dx_m=10.0)
    out_path = os.path.join(batch_dir, f"file_{i:03d}_fk.npz")
    (synth
     .detrend()
     .taper(alpha=0.05)
     .bandpass(10, 30)
     .fk_filter(1400, 3500, 10, 30)
     .save_npz(out_path))
    print(f"  file_{i:03d} → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MULTISPECTRAL REPRESENTATION (MSR)
# ─────────────────────────────────────────────────────────────────────────────

print("\nMultispectral Representation (MSR) ...")

from dasexplorer.core.msr import (
    multispectral_representation, MSRcube,
    msr_to_rgb, msr_to_grayscale,
    export_npz as msr_export_npz,
    export_tiff as msr_export_tiff,
)

bands = [(1.0, 5.0), (5.0, 15.0), (15.0, 40.0)]

# ── Via DASdataset method — returns MSRcube with metadata ─────────────────────
cube = ds.msr(bands=bands, percentile=95.0)
print(f"  {cube}")                          # MSRcube repr
print(f"  shape:       {cube.shape}")
print(f"  n_bands:     {cube.n_bands}")
print(f"  array range: [{cube.array.min():.3f}, {cube.array.max():.3f}]")

# ── Visualisation ─────────────────────────────────────────────────────────────
# RGB composite — 3 bands assigned to R, G, B
rgb = cube.to_rgb(r_band=0, g_band=1, b_band=2)
print(f"  to_rgb:       {rgb.shape}  {rgb.dtype}")

# Single band — grayscale (default)
gray = cube.to_grayscale(band=0, colormap='gray')
print(f"  to_grayscale: {gray.shape}  colormap=gray")

# Single band — false-colour with custom range
viridis = cube.to_grayscale(band=1, colormap='viridis', vmin=0.0, vmax=0.8)
print(f"  false-colour: {viridis.shape}  colormap=viridis  vmax=0.8")

# ── Export — method chaining ──────────────────────────────────────────────────
(cube
    .export_npz(os.path.join(OUTPUT_DIR, "msr_cube.npz"))
    .export_tiff(os.path.join(OUTPUT_DIR, "msr_cube.tiff")))
print(f"  export_npz:  {OUTPUT_DIR}/msr_cube.npz")
print(f"  export_tiff: {OUTPUT_DIR}/msr_cube.tiff")

# ── Full pipeline chain ───────────────────────────────────────────────────────
(ds
    .detrend()
    .bandpass(1, 80)
    .msr(bands=bands)
    .export_npz(os.path.join(OUTPUT_DIR, "msr_processed.npz")))
print(f"  detrend → bandpass → msr → export_npz OK")

# ── Reload and verify ─────────────────────────────────────────────────────────
data = np.load(os.path.join(OUTPUT_DIR, "msr_cube.npz"))
print(f"  reloaded keys: {list(data.keys())}")
assert data['cube'].shape == cube.shape

# ── 5-band cube for ML pipelines ──────────────────────────────────────────────
bands5 = [(1,5),(5,10),(10,20),(20,40),(40,80)]
cube5  = ds.msr(bands=bands5)
print(f"  5-band cube: {cube5.shape}  (ready for ML)")

# ── Standalone — returns ndarray directly ────────────────────────────────────
arr = multispectral_representation(ds.tr, ds.fs_hz, bands=bands)
print(f"  standalone:  {arr.shape}  {arr.dtype}  (ndarray, no metadata)")

# ── numpy interop ─────────────────────────────────────────────────────────────
arr2 = np.array(cube)
print(f"  np.array(cube): {arr2.shape}")

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
