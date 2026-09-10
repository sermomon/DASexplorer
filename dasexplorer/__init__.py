from dasexplorer.version import __version__

# ── Public API ────────────────────────────────────────────────────────────────
from dasexplorer.api import DASdataset, DASannotations
from dasexplorer.core.processing import (
    bandpass_filter, hilbert_envelope,
    downsample_signal, upsample_signal,
    detrend, taper, normalize,
)
from dasexplorer.core.fk_filter import fk_filter_design, fk_filter_apply
from dasexplorer.core.rgb import compute_rgb_composite
from dasexplorer.core.msr import (
    multispectral_representation, MSRcube,
    msr_to_rgb, msr_to_grayscale,
    export_npz as msr_export_npz, export_tiff as msr_export_tiff,
)
from dasexplorer.core.readers import read_das_file, generate_synthetic_dataset
from dasexplorer.core.annotations_model import (
    AnnotationModel, AnnType,
    BBoxAnnotation, OBBAnnotation, KeypointAnnotation, LineAnnotation,
)

__all__ = [
    "__version__",
    # High-level API
    "DASdataset",
    "DASannotations",
    # Processing
    "bandpass_filter",
    "hilbert_envelope",
    "downsample_signal",
    "upsample_signal",
    "detrend",
    "taper",
    "normalize",
    "fk_filter_design",
    "fk_filter_apply",
    "compute_rgb_composite",
    "multispectral_representation",
    "MSRcube",
    "msr_to_rgb",
    "msr_to_grayscale",
    # Readers
    "read_das_file",
    "generate_synthetic_dataset",
    # Annotation model
    "AnnotationModel",
    "AnnType",
    "BBoxAnnotation",
    "OBBAnnotation",
    "KeypointAnnotation",
    "LineAnnotation",
]
