"""
Annotation export converters for DAS Explorer.

Three output formats:
  - YOLO      : one .txt per image (group), normalised bbox [0,1]
  - COCO JSON : single .json with all images and annotations
  - Raven CSV : Selection Table format used by Raven Pro / PAMGuard
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_csv(csv_path: str) -> List[dict]:
    """Read an annotation CSV and return a list of row dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_class_map(rows: List[dict]) -> Tuple[dict, List[str]]:
    """
    Build a label → class_id mapping from the 'id' column.

    If every id is already an integer string ("0", "1", …) the numeric value
    is used directly as class_id.  Otherwise labels are sorted and assigned
    sequential ids starting from 0.  Returns (class_map, sorted_label_list).
    """
    labels = list(dict.fromkeys(row["id"] for row in rows))  # ordered unique
    try:
        class_map = {lbl: int(lbl) for lbl in labels}
        # Re-sort by numeric value for the classes list
        labels_sorted = sorted(labels, key=lambda x: int(x))
    except ValueError:
        labels_sorted = sorted(labels)
        class_map = {lbl: idx for idx, lbl in enumerate(labels_sorted)}
    return class_map, labels_sorted


def _group_rows(rows: List[dict], group_by: str = "start_datetime_utc"):
    """
    Group rows by a column value (one YOLO/COCO image per group).
    Falls back to a single group 'all' if the column is absent.
    """
    groups: dict = {}
    for row in rows:
        key = row.get(group_by, "all") or "all"
        groups.setdefault(key, []).append(row)
    return groups


def _safe_name(name: str) -> str:
    return str(name).replace(":", "-").replace(" ", "_")


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------

def export_yolo(
    csv_path: str,
    output_dir: str,
    group_by: str = "start_datetime_utc",
) -> Tuple[int, List[str]]:
    """
    Convert a DAS annotation CSV to YOLO label files.

    Output naming:
      - Single group  → {csv_stem}.txt
      - Multiple groups → {csv_stem}_{safe_group_value}.txt
    Plus classes.txt listing all labels.

    Returns (n_files_written, list_of_errors).
    """
    rows = _read_csv(csv_path)
    if not rows:
        return 0, ["CSV is empty."]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    class_map, labels_sorted = _build_class_map(rows)
    groups = _group_rows(rows, group_by)
    stem = Path(csv_path).stem
    multi = len(groups) > 1

    errors: List[str] = []
    n_written = 0

    for name, group_rows in groups.items():
        lines: List[str] = []
        for row in group_rows:
            try:
                nt = int(row["nt"])
                nx = int(row["nx"])
                ti0, ti1 = int(row["ti0"]), int(row["ti1"])
                di0, di1 = int(row["di0"]), int(row["di1"])
                class_id = class_map[row["id"]]

                x_center = float(np.clip((ti0 + ti1) / 2.0 / nt, 0.0, 1.0))
                y_center = float(np.clip((di0 + di1) / 2.0 / nx, 0.0, 1.0))
                width    = float(np.clip((ti1 - ti0) / nt,        0.0, 1.0))
                height   = float(np.clip((di1 - di0) / nx,        0.0, 1.0))

                lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )
            except (KeyError, ValueError, ZeroDivisionError) as exc:
                errors.append(f"Row skipped ({row.get('id','?')}): {exc}")

        # Name: stem.txt (single group) or stem_groupvalue.txt (multiple)
        suffix = f"_{_safe_name(name)}" if multi else ""
        out_path = Path(output_dir) / f"{stem}{suffix}.txt"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        n_written += 1

    # classes.txt
    classes_path = Path(output_dir) / "classes.txt"
    classes_path.write_text("\n".join(labels_sorted), encoding="utf-8")

    return n_written, errors


# ---------------------------------------------------------------------------
# COCO JSON export
# ---------------------------------------------------------------------------

def export_coco(
    csv_path: str,
    output_dir: str,
    group_by: str = "start_datetime_utc",
) -> Tuple[int, List[str]]:
    """
    Convert a DAS annotation CSV to a single COCO-format JSON file.

    Output naming: {csv_stem}_coco.json
    Images are derived from groups (one group = one image). Bounding boxes
    are in absolute pixels [x, y, w, h] (COCO convention).

    Returns (1 if JSON written else 0, list_of_errors).
    """
    rows = _read_csv(csv_path)
    if not rows:
        return 0, ["CSV is empty."]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    class_map, labels_sorted = _build_class_map(rows)
    groups = _group_rows(rows, group_by)

    categories = [
        {"id": class_map[lbl], "name": lbl, "supercategory": "event"}
        for lbl in labels_sorted
    ]

    images: List[dict] = []
    annotations: List[dict] = []
    errors: List[str] = []
    ann_id = 1

    for img_id, (name, group_rows) in enumerate(groups.items(), start=1):
        # Use the first row's nt/nx as the image dimensions
        try:
            nt = int(group_rows[0]["nt"])
            nx = int(group_rows[0]["nx"])
        except (KeyError, ValueError):
            errors.append(f"Group '{name}': could not read nt/nx — skipped.")
            continue

        images.append({
            "id": img_id,
            "file_name": f"{_safe_name(name)}.png",
            "width": nt,
            "height": nx,
            "date_captured": name if name != "all" else "",
        })

        for row in group_rows:
            try:
                ti0, ti1 = int(row["ti0"]), int(row["ti1"])
                di0, di1 = int(row["di0"]), int(row["di1"])
                w_px = ti1 - ti0
                h_px = di1 - di0
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_map[row["id"]],
                    "bbox": [ti0, di0, w_px, h_px],   # [x, y, w, h] COCO format
                    "area": w_px * h_px,
                    "iscrowd": 0,
                    # Keep original physical coords as extra metadata
                    "attributes": {
                        "t0": float(row.get("t0", 0)),
                        "t1": float(row.get("t1", 0)),
                        "d0_m": float(row.get("d0", 0)),
                        "d1_m": float(row.get("d1", 0)),
                        "comment": row.get("comment", ""),
                    },
                })
                ann_id += 1
            except (KeyError, ValueError) as exc:
                errors.append(f"Row skipped ({row.get('id','?')}): {exc}")

    coco = {
        "info": {
            "description": "DAS Explorer annotation export",
            "version": "1.0",
            "source_csv": os.path.basename(csv_path),
        },
        "licenses": [],
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }

    stem = Path(csv_path).stem
    out_path = Path(output_dir) / f"{stem}_coco.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    return 1, errors


# ---------------------------------------------------------------------------
# Raven CSV (Selection Table) export
# ---------------------------------------------------------------------------

_RAVEN_FIELDS = [
    "Selection", "View", "Channel",
    "Begin Time (s)", "End Time (s)",
    "Low Freq (Hz)", "High Freq (Hz)",
    "Begin Path", "Species", "Notes",
]


def export_raven(
    csv_path: str,
    output_dir: str,
    fs_hz: float = None,
    group_by: str = "start_datetime_utc",
) -> Tuple[int, List[str]]:
    """
    Convert a DAS annotation CSV to Raven Pro Selection Table format.

    Output naming:
      - Single group  → {csv_stem}_raven.csv
      - Multiple groups → {csv_stem}_raven_{safe_group_value}.csv

    Tab-separated, compatible with Raven Pro and PAMGuard.
    Extension is .csv (not .txt) to clearly distinguish from YOLO .txt files.
    The _raven suffix prevents collision with the original DAS Explorer CSV.

    Returns (n_files_written, list_of_errors).
    """
    rows = _read_csv(csv_path)
    if not rows:
        return 0, ["CSV is empty."]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    groups = _group_rows(rows, group_by)
    stem = Path(csv_path).stem
    multi = len(groups) > 1

    errors: List[str] = []
    n_written = 0

    for name, group_rows in groups.items():
        suffix = f"_{_safe_name(name)}" if multi else ""
        out_path = Path(output_dir) / f"{stem}_raven{suffix}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_RAVEN_FIELDS,
                                    delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            for sel_idx, row in enumerate(group_rows, start=1):
                try:
                    t0 = float(row["t0"])
                    t1 = float(row["t1"])
                    lo_freq = 0.0
                    hi_freq = float(fs_hz) / 2.0 if fs_hz else 0.0
                    writer.writerow({
                        "Selection":      sel_idx,
                        "View":           "Spectrogram 1",
                        "Channel":        1,
                        "Begin Time (s)": f"{t0:.6f}",
                        "End Time (s)":   f"{t1:.6f}",
                        "Low Freq (Hz)":  f"{lo_freq:.2f}",
                        "High Freq (Hz)": f"{hi_freq:.2f}",
                        "Begin Path":     stem,
                        "Species":        row.get("id", ""),
                        "Notes":          row.get("comment", ""),
                    })
                except (KeyError, ValueError) as exc:
                    errors.append(f"Row skipped ({row.get('id','?')}): {exc}")
        n_written += 1

    return n_written, errors
