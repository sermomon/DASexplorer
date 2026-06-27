"""
Annotation model for DAS Explorer.

Supports four annotation types:
  - BBox      : axis-aligned bounding box
  - OBB       : oriented bounding box (centre + half-axes + angle)
  - Keypoints : ordered list of (t, d) points within one annotation
  - Line      : LineString — ordered list of (t, d) vertices

Each type is saved to a separate CSV with a type-specific suffix:
  _bbox.csv  _obb.csv  _kp.csv  _lin.csv
"""

import csv
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Annotation type enum
# ---------------------------------------------------------------------------

class AnnType(str, Enum):
    BBOX = "bbox"
    OBB  = "obb"
    KP   = "kp"
    LINE = "lin"

ANN_SUFFIX = {
    AnnType.BBOX: "_bbox.csv",
    AnnType.OBB:  "_obb.csv",
    AnnType.KP:   "_kp.csv",
    AnnType.LINE: "_lin.csv",
}

ANN_LABEL = {
    AnnType.BBOX: "BBox",
    AnnType.OBB:  "OBB",
    AnnType.KP:   "Keypoints",
    AnnType.LINE: "Line",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BBoxAnnotation:
    """Axis-aligned bounding box."""
    ann_type: str      # always "bbox"
    id: str
    comment: str
    t0: float
    t1: float
    d0: float
    d1: float
    ti0: int
    ti1: int
    di0: int
    di1: int
    nt: int
    nx: int
    downsample: int
    start_datetime_utc: str
    velocity_ms: Optional[float] = None
    velocity_r2: Optional[float] = None

    @property
    def label(self) -> str:
        return f"[BBox:{self.id}]  t={self.t0:.2f}–{self.t1:.2f}s  d={self.d0:.0f}–{self.d1:.0f}m"


@dataclass
class OBBAnnotation:
    """
    Oriented bounding box.

    Stored as centre (in physical coords and pixel index), half-extents
    (w_t = half-width in time, h_d = half-height in dist) and angle_deg
    (rotation counter-clockwise from the +time axis, in degrees).
    """
    ann_type: str      # always "obb"
    id: str
    comment: str
    cx_t: float        # centre time [s]
    cy_d: float        # centre distance [m]
    w_t: float         # half-width in time [s]
    h_d: float         # half-height in distance [m]
    angle_deg: float   # rotation CCW from +time axis [deg]
    cx_ti: int         # centre time index
    cy_di: int         # centre distance index
    w_ti: int          # half-width in time bins
    h_di: int          # half-height in distance bins
    nt: int
    nx: int
    downsample: int
    start_datetime_utc: str

    @property
    def label(self) -> str:
        return (f"[OBB:{self.id}]  t={self.cx_t:.2f}s  d={self.cy_d:.0f}m  "
                f"θ={self.angle_deg:.1f}°")

    def corners_ti_di(self) -> List[Tuple[float, float]]:
        """
        Return the four corners as (ti, di) index pairs, in clockwise order
        starting from the corner in the +time/+dist quadrant.
        """
        a = np.deg2rad(self.angle_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        hw, hh = self.w_ti, self.h_di
        offsets = [( hw,  hh), ( hw, -hh), (-hw, -hh), (-hw,  hh)]
        corners = []
        for dt, dd in offsets:
            corners.append((
                self.cx_ti + cos_a * dt - sin_a * dd,
                self.cy_di + sin_a * dt + cos_a * dd,
            ))
        return corners

    def bbox_ti_di(self) -> Tuple[int, int, int, int]:
        """Axis-aligned bounding box enclosing the OBB, in index coords."""
        corners = self.corners_ti_di()
        tis = [c[0] for c in corners]
        dis = [c[1] for c in corners]
        return (int(min(tis)), int(max(tis)), int(min(dis)), int(max(dis)))


@dataclass
class KeypointAnnotation:
    """
    Multiple keypoints within one annotation event.

    kp_ti and kp_di are JSON-serialised lists of integer pixel indices.
    kp_t  and kp_d  are JSON-serialised lists of physical coords [s] / [m].
    """
    ann_type: str      # always "kp"
    id: str
    comment: str
    kp_t: str          # JSON list of floats [s]
    kp_d: str          # JSON list of floats [m]
    kp_ti: str         # JSON list of ints
    kp_di: str         # JSON list of ints
    nt: int
    nx: int
    downsample: int
    start_datetime_utc: str

    @property
    def label(self) -> str:
        n = len(json.loads(self.kp_t))
        return f"[KP:{self.id}]  {n} point(s)"

    def get_points_t_d(self) -> List[Tuple[float, float]]:
        return list(zip(json.loads(self.kp_t), json.loads(self.kp_d)))

    def get_points_ti_di(self) -> List[Tuple[int, int]]:
        return list(zip(json.loads(self.kp_ti), json.loads(self.kp_di)))


@dataclass
class LineAnnotation:
    """
    LineString annotation — an ordered sequence of (t, d) vertices.

    pts_ti and pts_di are JSON-serialised lists of integer pixel indices.
    pts_t  and pts_d  are JSON-serialised lists of physical coords.
    """
    ann_type: str      # always "lin"
    id: str
    comment: str
    pts_t: str         # JSON list of floats [s]
    pts_d: str         # JSON list of floats [m]
    pts_ti: str        # JSON list of ints
    pts_di: str        # JSON list of ints
    nt: int
    nx: int
    downsample: int
    start_datetime_utc: str

    @property
    def label(self) -> str:
        n = len(json.loads(self.pts_t))
        return f"[Line:{self.id}]  {n} vertices"

    def get_points_t_d(self) -> List[Tuple[float, float]]:
        return list(zip(json.loads(self.pts_t), json.loads(self.pts_d)))

    def get_points_ti_di(self) -> List[Tuple[int, int]]:
        return list(zip(json.loads(self.pts_ti), json.loads(self.pts_di)))


# Union type for type hints
AnyAnnotation = BBoxAnnotation | OBBAnnotation | KeypointAnnotation | LineAnnotation


# ---------------------------------------------------------------------------
# CSV field definitions per type
# ---------------------------------------------------------------------------

BBOX_FIELDS = [
    "ann_type", "id", "comment",
    "t0", "t1", "d0", "d1",
    "ti0", "ti1", "di0", "di1",
    "nt", "nx", "downsample", "start_datetime_utc",
    "velocity_ms", "velocity_r2",
]

OBB_FIELDS = [
    "ann_type", "id", "comment",
    "cx_t", "cy_d", "w_t", "h_d", "angle_deg",
    "cx_ti", "cy_di", "w_ti", "h_di",
    "nt", "nx", "downsample", "start_datetime_utc",
]

KP_FIELDS = [
    "ann_type", "id", "comment",
    "kp_t", "kp_d", "kp_ti", "kp_di",
    "nt", "nx", "downsample", "start_datetime_utc",
]

LINE_FIELDS = [
    "ann_type", "id", "comment",
    "pts_t", "pts_d", "pts_ti", "pts_di",
    "nt", "nx", "downsample", "start_datetime_utc",
]

_FIELDS_FOR_TYPE = {
    AnnType.BBOX: BBOX_FIELDS,
    AnnType.OBB:  OBB_FIELDS,
    AnnType.KP:   KP_FIELDS,
    AnnType.LINE: LINE_FIELDS,
}

_CLASS_FOR_TYPE = {
    AnnType.BBOX: BBoxAnnotation,
    AnnType.OBB:  OBBAnnotation,
    AnnType.KP:   KeypointAnnotation,
    AnnType.LINE: LineAnnotation,
}


# ---------------------------------------------------------------------------
# Per-type model
# ---------------------------------------------------------------------------

class AnnotationModel:
    """
    In-memory list of annotations of ONE type for the currently loaded file.
    Each type has its own model instance in MainWindow.
    """

    def __init__(self, ann_type: AnnType):
        self.ann_type = ann_type
        self._annotations: List[AnyAnnotation] = []
        self.dirty: bool = False

    # CRUD ------------------------------------------------------------------

    def add(self, ann: AnyAnnotation) -> None:
        self._annotations.append(ann)
        self.dirty = True

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._annotations):
            self._annotations.pop(index)
            self.dirty = True

    def update(self, index: int, **kwargs) -> None:
        if 0 <= index < len(self._annotations):
            for k, v in kwargs.items():
                setattr(self._annotations[index], k, v)
            self.dirty = True

    def clear(self) -> None:
        self._annotations.clear()
        self.dirty = False

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, index: int) -> AnyAnnotation:
        return self._annotations[index]

    def __iter__(self):
        return iter(self._annotations)

    # CSV I/O ---------------------------------------------------------------

    def save(self, path: str) -> None:
        fields = _FIELDS_FOR_TYPE[self.ann_type]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for ann in self._annotations:
                row = asdict(ann)
                writer.writerow({k: row.get(k, "") for k in fields})
        self.dirty = False

    def load(self, path: str) -> None:
        self._annotations.clear()
        if not os.path.exists(path):
            self.dirty = False
            return
        cls = _CLASS_FOR_TYPE[self.ann_type]
        fields = _FIELDS_FOR_TYPE[self.ann_type]
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kwargs = {}
                for field in fields:
                    val = row.get(field, "")
                    # Type coercions
                    if field in ("t0","t1","d0","d1","cx_t","cy_d","w_t","h_d","angle_deg"):
                        kwargs[field] = float(val) if val != "" else 0.0
                    elif field in ("ti0","ti1","di0","di1","cx_ti","cy_di","w_ti","h_di",
                                   "nt","nx","downsample"):
                        kwargs[field] = int(val) if val != "" else 0
                    elif field == "velocity_ms":
                        kwargs[field] = float(val) if val != "" else None
                    elif field == "velocity_r2":
                        kwargs[field] = float(val) if val != "" else None
                    else:
                        kwargs[field] = val
                try:
                    self._annotations.append(cls(**kwargs))
                except TypeError:
                    pass   # skip rows with missing/incompatible fields
        self.dirty = False

    # Path helpers ----------------------------------------------------------

    @staticmethod
    def csv_path_for(data_path: str, ann_type: AnnType) -> str:
        base = os.path.splitext(data_path)[0]
        return base + ANN_SUFFIX[ann_type]

    # Index helpers ---------------------------------------------------------

    @staticmethod
    def compute_indices(t0, t1, d0, d1, time_s, dist_m):
        n_time = len(time_s)
        n_dist = len(dist_m)
        ti0 = int(np.clip(np.argmin(np.abs(time_s - t0)), 0, n_time - 1))
        ti1 = int(np.clip(np.argmin(np.abs(time_s - t1)), 0, n_time - 1))
        di0 = int(np.clip(np.argmin(np.abs(dist_m - d0)), 0, n_dist - 1))
        di1 = int(np.clip(np.argmin(np.abs(dist_m - d1)), 0, n_dist - 1))
        return ti0, ti1, di0, di1

    @staticmethod
    def coord_to_index(t: float, d: float, time_s, dist_m) -> Tuple[int, int]:
        ti = int(np.clip(np.argmin(np.abs(time_s - t)), 0, len(time_s) - 1))
        di = int(np.clip(np.argmin(np.abs(dist_m - d)), 0, len(dist_m) - 1))
        return ti, di

    # Legacy compatibility: single model accessing annotations regardless of type
    @property
    def annotations(self) -> List[AnyAnnotation]:
        return self._annotations
