from dasexplorer.version import __version__

# ── Public API ────────────────────────────────────────────────────────────────
from dasexplorer.api import DASdataset, DASannotations
from dasexplorer.core.data_model import DASDataset
from dasexplorer.core.processing import bandpass_filter, hilbert_envelope
from dasexplorer.core.fk_filter import fk_filter_design, fk_filter_apply
from dasexplorer.core.rgb import compute_rgb_composite
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
    # Internal dataclass
    "DASDataset",
    # Processing
    "bandpass_filter",
    "hilbert_envelope",
    "fk_filter_design",
    "fk_filter_apply",
    "compute_rgb_composite",
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
