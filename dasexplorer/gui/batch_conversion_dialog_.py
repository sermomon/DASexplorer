"""
Batch Conversion dialog for DAS Explorer.

Converts a list of DAS files (one interrogator type) to NPZ or MAT format,
writing all metadata needed to reload each file as if it were the original.
Conversion runs file by file with a progress bar; each converted file is
saved to the selected output directory.
"""

import json
import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from dasexplorer.core.readers import (
    INTERROGATOR_LABELS, INTERROGATOR_TYPES, read_das_file,
)
from dasexplorer.gui import theme


# File extensions per interrogator (mirrors main_window.py)
_FILE_EXTENSIONS = {
    "hdas2.5":   [".bin"],
    "optasense": [".h5", ".hdf5"],
}


class BatchConversionDialog(QtWidgets.QDialog):
    """
    Non-modal dialog for converting a folder of DAS files to NPZ or MAT.

    Layout
    ------
    Top tab bar: Data  (space for future tabs)
    Inside Data tab:
      - Row: input directory + Interrogator + Change Dir button
      - File list with checkboxes + Select All / Deselect All
      - Row: output directory + Change button  (defaults to input dir)
      - Separator
      - Convert to NPZ  |  Convert to MAT  buttons
      - Progress bar + status label
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Conversion")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.50), int(screen.height() * 0.65))

        self._input_dir: str = ""
        self._output_dir: str = ""

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)

        # "Batch Conversion" header label
        header = QtWidgets.QLabel("Batch Conversion")
        header.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(header)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # Collapsible "Data" section using QToolButton as toggle header
        self._data_toggle = QtWidgets.QToolButton()
        self._data_toggle.setText("  Data")
        self._data_toggle.setCheckable(True)
        self._data_toggle.setChecked(True)
        self._data_toggle.setArrowType(QtCore.Qt.DownArrow)
        self._data_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._data_toggle.setStyleSheet(
            "QToolButton { font-weight: bold; font-size: 10pt; border: none; }"
        )
        self._data_toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._data_toggle.clicked.connect(self._on_data_toggle)
        layout.addWidget(self._data_toggle)

        # Container for the Data section content
        self._data_container = QtWidgets.QWidget()
        self._build_data_content(self._data_container)
        layout.addWidget(self._data_container, 1)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep2)

        # Collapsible "Annotations" section
        self._ann_toggle = QtWidgets.QToolButton()
        self._ann_toggle.setText("  Annotations")
        self._ann_toggle.setCheckable(True)
        self._ann_toggle.setChecked(False)   # collapsed by default
        self._ann_toggle.setArrowType(QtCore.Qt.RightArrow)
        self._ann_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._ann_toggle.setStyleSheet(
            "QToolButton { font-weight: bold; font-size: 10pt; border: none; }"
        )
        self._ann_toggle.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )
        self._ann_toggle.clicked.connect(self._on_ann_toggle)
        layout.addWidget(self._ann_toggle)

        self._ann_container = QtWidgets.QWidget()
        self._ann_container.setVisible(False)
        self._build_ann_content(self._ann_container)
        layout.addWidget(self._ann_container, 1)

    def _on_data_toggle(self, checked: bool) -> None:
        self._data_container.setVisible(checked)
        self._data_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )

    def _on_ann_toggle(self, checked: bool) -> None:
        self._ann_container.setVisible(checked)
        self._ann_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )

    def _build_ann_content(self, parent: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 4, 4, 4)

        # ── Input: CSV file(s) ──────────────────────────────────────────
        csv_row = QtWidgets.QHBoxLayout()
        lbl_csv = QtWidgets.QLabel("Input CSV:")
        lbl_csv.setStyleSheet("font-weight: bold;")
        lbl_csv.setFixedWidth(130)
        csv_row.addWidget(lbl_csv)
        self.lbl_ann_csv_dir = QtWidgets.QLineEdit()
        self.lbl_ann_csv_dir.setPlaceholderText("No directory selected…")
        self.lbl_ann_csv_dir.setReadOnly(True)
        csv_row.addWidget(self.lbl_ann_csv_dir, 1)
        btn_browse_csv = QtWidgets.QPushButton("Browse")
        btn_browse_csv.setMinimumWidth(100)
        btn_browse_csv.clicked.connect(self._on_browse_ann_csv_dir)
        csv_row.addWidget(btn_browse_csv)
        layout.addLayout(csv_row)

        # ── CSV file list ───────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("CSV files to convert:"))
        self.ann_file_list = QtWidgets.QListWidget()
        self.ann_file_list.setUniformItemSizes(True)
        layout.addWidget(self.ann_file_list, 1)

        ann_sel_row = QtWidgets.QHBoxLayout()
        btn_ann_sel_all = QtWidgets.QPushButton("Select All")
        btn_ann_sel_all.clicked.connect(self._on_ann_select_all)
        btn_ann_desel_all = QtWidgets.QPushButton("Deselect All")
        btn_ann_desel_all.clicked.connect(self._on_ann_deselect_all)
        self.lbl_ann_file_count = QtWidgets.QLabel("0 files")
        self.lbl_ann_file_count.setStyleSheet(
            f"color: {theme.current()['qt_text_dim']};"
        )
        ann_sel_row.addWidget(btn_ann_sel_all)
        ann_sel_row.addWidget(btn_ann_desel_all)
        ann_sel_row.addStretch()
        ann_sel_row.addWidget(self.lbl_ann_file_count)
        layout.addLayout(ann_sel_row)

        # ── Output directory ────────────────────────────────────────────
        ann_out_row = QtWidgets.QHBoxLayout()
        lbl_ann_out = QtWidgets.QLabel("Output directory:")
        lbl_ann_out.setStyleSheet("font-weight: bold;")
        lbl_ann_out.setFixedWidth(130)
        ann_out_row.addWidget(lbl_ann_out)
        self.lbl_ann_output_dir = QtWidgets.QLineEdit()
        self.lbl_ann_output_dir.setPlaceholderText("Same as input directory")
        self.lbl_ann_output_dir.setReadOnly(True)
        ann_out_row.addWidget(self.lbl_ann_output_dir, 1)
        btn_ann_out = QtWidgets.QPushButton("Browse")
        btn_ann_out.setMinimumWidth(100)
        btn_ann_out.clicked.connect(self._on_browse_ann_output)
        ann_out_row.addWidget(btn_ann_out)
        layout.addLayout(ann_out_row)

        # ── Group-by ────────────────────────────────────────────────────
        gb_row = QtWidgets.QHBoxLayout()
        lbl_gb = QtWidgets.QLabel("Group by column:")
        lbl_gb.setStyleSheet("font-weight: bold;")
        lbl_gb.setFixedWidth(130)
        gb_row.addWidget(lbl_gb)
        self.combo_group_by = QtWidgets.QComboBox()
        self.combo_group_by.addItems(["start_datetime_utc", "id", "(all in one file)"])
        self.combo_group_by.setToolTip(
            "Determines how annotations are grouped into output files.\n"
            "• start_datetime_utc — one file per DAS acquisition window (recommended)\n"
            "• id — one file per label/class\n"
            "• (all in one file) — single output file with all annotations"
        )
        gb_row.addWidget(self.combo_group_by, 1)
        layout.addLayout(gb_row)

        # ── Separator ───────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # ── Export buttons ──────────────────────────────────────────────
        lbl_fmt = QtWidgets.QLabel("Export format:")
        lbl_fmt.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_fmt)

        btn_fmt_row = QtWidgets.QHBoxLayout()
        self.btn_yolo = QtWidgets.QPushButton("Export YOLO")
        self.btn_yolo.setMinimumHeight(32)
        self.btn_yolo.setToolTip(
            "YOLO format: one .txt per image + classes.txt\n"
            "Each line: <class_id> <x_center> <y_center> <width> <height> (normalised)"
        )
        self.btn_yolo.clicked.connect(lambda: self._on_export_annotations("yolo"))

        self.btn_coco = QtWidgets.QPushButton("Export COCO JSON")
        self.btn_coco.setMinimumHeight(32)
        self.btn_coco.setToolTip(
            "COCO JSON format: single .json with all images and annotations.\n"
            "Compatible with Detectron2, MMDetection, RT-DETR."
        )
        self.btn_coco.clicked.connect(lambda: self._on_export_annotations("coco"))

        self.btn_raven = QtWidgets.QPushButton("Export Raven CSV")
        self.btn_raven.setMinimumHeight(32)
        self.btn_raven.setToolTip(
            "Raven Pro Selection Table format (tab-separated).\n"
            "Compatible with Raven Pro, PAMGuard, and BIANET-C workflows."
        )
        self.btn_raven.clicked.connect(lambda: self._on_export_annotations("raven"))

        btn_fmt_row.addWidget(self.btn_yolo)
        btn_fmt_row.addWidget(self.btn_coco)
        btn_fmt_row.addWidget(self.btn_raven)
        layout.addLayout(btn_fmt_row)

        # ── Progress ────────────────────────────────────────────────────
        self.ann_progress_bar = QtWidgets.QProgressBar()
        self.ann_progress_bar.setValue(0)
        self.ann_progress_bar.setTextVisible(True)
        self.ann_progress_bar.setFixedHeight(18)
        layout.addWidget(self.ann_progress_bar)

        self.lbl_ann_status = QtWidgets.QLabel("")
        self.lbl_ann_status.setStyleSheet(
            f"color: {theme.current()['qt_text_dim']}; font-size: 8pt;"
        )
        self.lbl_ann_status.setWordWrap(True)
        layout.addWidget(self.lbl_ann_status)

        # Internal state
        self._ann_csv_dir: str = ""
        self._ann_output_dir: str = ""

    def _build_data_content(self, parent: QtWidgets.QWidget) -> None:
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 4, 4, 4)

        # ── Input directory + interrogator ──────────────────────────────
        dir_row = QtWidgets.QHBoxLayout()
        lbl_in = QtWidgets.QLabel("Input directory:")
        lbl_in.setStyleSheet("font-weight: bold;")
        lbl_in.setFixedWidth(130)
        dir_row.addWidget(lbl_in)

        self.lbl_input_dir = QtWidgets.QLineEdit()
        self.lbl_input_dir.setPlaceholderText("No directory selected…")
        self.lbl_input_dir.setReadOnly(True)
        dir_row.addWidget(self.lbl_input_dir, 1)

        btn_change_dir = QtWidgets.QPushButton("Browse")
        btn_change_dir.setMinimumWidth(100)
        btn_change_dir.clicked.connect(self._on_change_input_dir)
        dir_row.addWidget(btn_change_dir)

        lbl_intr = QtWidgets.QLabel("Interrogator:")
        lbl_intr.setStyleSheet("font-weight: bold;")
        dir_row.addWidget(lbl_intr)

        self.combo_interrogator = QtWidgets.QComboBox()
        for label in INTERROGATOR_LABELS:
            self.combo_interrogator.addItem(label)
        self.combo_interrogator.setFixedWidth(120)
        self.combo_interrogator.currentIndexChanged.connect(self._on_interrogator_changed)
        dir_row.addWidget(self.combo_interrogator)

        layout.addLayout(dir_row)

        # ── File list ───────────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("Files to convert:"))

        self.file_list = QtWidgets.QListWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_list.setUniformItemSizes(True)
        layout.addWidget(self.file_list, 1)

        sel_row = QtWidgets.QHBoxLayout()
        btn_select_all = QtWidgets.QPushButton("Select All")
        btn_select_all.clicked.connect(self._on_select_all)
        btn_deselect_all = QtWidgets.QPushButton("Deselect All")
        btn_deselect_all.clicked.connect(self._on_deselect_all)
        self.lbl_file_count = QtWidgets.QLabel("0 files")
        self.lbl_file_count.setStyleSheet(f"color: {theme.current()['qt_text_dim']};")
        sel_row.addWidget(btn_select_all)
        sel_row.addWidget(btn_deselect_all)
        sel_row.addStretch()
        sel_row.addWidget(self.lbl_file_count)
        layout.addLayout(sel_row)

        # ── Output directory ────────────────────────────────────────────
        out_row = QtWidgets.QHBoxLayout()
        lbl_out = QtWidgets.QLabel("Output directory:")
        lbl_out.setStyleSheet("font-weight: bold;")
        lbl_out.setFixedWidth(130)
        out_row.addWidget(lbl_out)

        self.lbl_output_dir = QtWidgets.QLineEdit()
        self.lbl_output_dir.setPlaceholderText("Same as input directory")
        self.lbl_output_dir.setReadOnly(True)
        out_row.addWidget(self.lbl_output_dir, 1)

        btn_change_out = QtWidgets.QPushButton("Browse")
        btn_change_out.setMinimumWidth(100)
        btn_change_out.clicked.connect(self._on_change_output_dir)
        out_row.addWidget(btn_change_out)

        layout.addLayout(out_row)

        # ── Separator ───────────────────────────────────────────────────
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # ── Convert buttons ─────────────────────────────────────────────
        btn_convert_row = QtWidgets.QHBoxLayout()
        self.btn_npz = QtWidgets.QPushButton("Convert to NPZ")
        self.btn_npz.setMinimumHeight(32)
        self.btn_npz.clicked.connect(lambda: self._on_convert("npz"))
        self.btn_mat = QtWidgets.QPushButton("Convert to MAT")
        self.btn_mat.setMinimumHeight(32)
        self.btn_mat.clicked.connect(lambda: self._on_convert("mat"))
        btn_convert_row.addWidget(self.btn_npz)
        btn_convert_row.addWidget(self.btn_mat)
        layout.addLayout(btn_convert_row)

        # ── Progress ────────────────────────────────────────────────────
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(18)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet(f"color: {theme.current()['qt_text_dim']}; font-size: 8pt;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_change_input_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select input directory", self._input_dir or ""
        )
        if not directory:
            return
        self._input_dir = directory
        self.lbl_input_dir.setText(directory)
        # Default output dir = same as input
        if not self._output_dir:
            self._output_dir = directory
            self.lbl_output_dir.setText(directory)
        self._refresh_file_list()

    def _on_change_output_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", self._output_dir or self._input_dir or ""
        )
        if directory:
            self._output_dir = directory
            self.lbl_output_dir.setText(directory)

    def _on_interrogator_changed(self, _index: int) -> None:
        self._refresh_file_list()

    def _on_select_all(self) -> None:
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(QtCore.Qt.Checked)
        self._update_file_count()

    def _on_deselect_all(self) -> None:
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        self._update_file_count()

    # ------------------------------------------------------------------
    # File list
    # ------------------------------------------------------------------

    def _refresh_file_list(self) -> None:
        self.file_list.clear()
        if not self._input_dir or not os.path.isdir(self._input_dir):
            return
        intr_key = INTERROGATOR_TYPES[self.combo_interrogator.currentIndex()]
        exts = _FILE_EXTENSIONS.get(intr_key, [])
        files = sorted([
            f for f in os.listdir(self._input_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        for fname in files:
            item = QtWidgets.QListWidgetItem(fname)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.file_list.addItem(item)
        self.file_list.itemChanged.connect(lambda _: self._update_file_count())
        self._update_file_count()
        self._set_status(f"{len(files)} file(s) found.", dim=True)

    def _update_file_count(self) -> None:
        checked = sum(
            1 for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == QtCore.Qt.Checked
        )
        total = self.file_list.count()
        self.lbl_file_count.setText(f"{checked} / {total} selected")

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False, dim: bool = False) -> None:
        if error:
            color = "#ff4444"
        elif dim:
            color = theme.current()["qt_text_dim"]
        else:
            color = theme.current()["qt_text"]
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 8pt;")
        self.lbl_status.setText(msg)
        QtWidgets.QApplication.processEvents()

    def _on_convert(self, fmt: str) -> None:
        if not self._input_dir:
            self._set_status("Please select an input directory first.", error=True)
            return

        out_dir = self._output_dir or self._input_dir
        if not os.path.isdir(out_dir):
            self._set_status(f"Output directory does not exist: {out_dir}", error=True)
            return

        files_to_convert = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == QtCore.Qt.Checked
        ]
        if not files_to_convert:
            self._set_status("No files selected.", error=True)
            return

        intr_key = INTERROGATOR_TYPES[self.combo_interrogator.currentIndex()]
        intr_label = INTERROGATOR_LABELS[self.combo_interrogator.currentIndex()]

        # Disable buttons during conversion
        self.btn_npz.setEnabled(False)
        self.btn_mat.setEnabled(False)
        self.progress_bar.setMaximum(len(files_to_convert))
        self.progress_bar.setValue(0)

        errors = []
        for n, fname in enumerate(files_to_convert, start=1):
            in_path = os.path.join(self._input_dir, fname)
            stem = os.path.splitext(fname)[0]
            out_fname = f"{stem}.{fmt}"
            out_path = os.path.join(out_dir, out_fname)

            self._set_status(
                f"[{n}/{len(files_to_convert)}] Reading {fname}…", dim=True
            )

            try:
                ds = read_das_file(in_path, intr_key)
            except Exception as exc:
                errors.append(f"{fname}: read error — {exc}")
                self.progress_bar.setValue(n)
                continue

            self._set_status(
                f"[{n}/{len(files_to_convert)}] Writing {out_fname}…", dim=True
            )

            start_iso = (
                ds.start_datetime_utc.isoformat()
                if ds.start_datetime_utc is not None else ""
            )
            meta_json = json.dumps(ds.metadata or {})

            try:
                if fmt == "npz":
                    np.savez_compressed(
                        out_path,
                        tr=ds.tr,
                        dist_m=ds.dist_m,
                        time_s=ds.time_s,
                        fs_hz=np.float64(ds.fs_hz),
                        start_datetime_utc=start_iso,
                        filename=ds.filename or fname,
                        interrogator=ds.interrogator or intr_key,
                        downsample=np.int64(ds.downsample or 1),
                        units=ds.units or "",
                        metadata_json=meta_json,
                    )
                else:  # mat
                    import scipy.io as sio
                    sio.savemat(
                        out_path,
                        {
                            "tr": ds.tr,
                            "dist_m": ds.dist_m,
                            "time_s": ds.time_s,
                            "fs_hz": float(ds.fs_hz),
                            "start_datetime_utc": start_iso,
                            "filename": ds.filename or fname,
                            "interrogator": ds.interrogator or intr_key,
                            "downsample": int(ds.downsample or 1),
                            "units": ds.units or "",
                            "metadata_json": meta_json,
                        },
                        do_compression=True,
                    )
            except Exception as exc:
                errors.append(f"{fname}: write error — {exc}")

            self.progress_bar.setValue(n)

        # Re-enable buttons
        self.btn_npz.setEnabled(True)
        self.btn_mat.setEnabled(True)

        n_ok = len(files_to_convert) - len(errors)
        if errors:
            self._set_status(
                f"Done: {n_ok}/{len(files_to_convert)} converted. "
                f"Errors:\n" + "\n".join(errors),
                error=True,
            )
        else:
            self._set_status(
                f"Done: {n_ok} file(s) converted to {fmt.upper()} → {out_dir}",
            )

    # ------------------------------------------------------------------
    # Annotation section slots
    # ------------------------------------------------------------------

    def _on_browse_ann_csv_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select CSV directory", self._ann_csv_dir or ""
        )
        if not directory:
            return
        self._ann_csv_dir = directory
        self.lbl_ann_csv_dir.setText(directory)
        if not self._ann_output_dir:
            self._ann_output_dir = directory
            self.lbl_ann_output_dir.setText(directory)
        self._refresh_ann_file_list()

    def _on_browse_ann_output(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory",
            self._ann_output_dir or self._ann_csv_dir or ""
        )
        if directory:
            self._ann_output_dir = directory
            self.lbl_ann_output_dir.setText(directory)

    def _refresh_ann_file_list(self) -> None:
        self.ann_file_list.clear()
        if not self._ann_csv_dir or not os.path.isdir(self._ann_csv_dir):
            return
        files = sorted([
            f for f in os.listdir(self._ann_csv_dir)
            if f.lower().endswith(".csv")
        ])
        for fname in files:
            item = QtWidgets.QListWidgetItem(fname)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.ann_file_list.addItem(item)
        self.ann_file_list.itemChanged.connect(
            lambda _: self._update_ann_file_count()
        )
        self._update_ann_file_count()
        self._set_ann_status(f"{len(files)} CSV file(s) found.", dim=True)

    def _update_ann_file_count(self) -> None:
        checked = sum(
            1 for i in range(self.ann_file_list.count())
            if self.ann_file_list.item(i).checkState() == QtCore.Qt.Checked
        )
        total = self.ann_file_list.count()
        self.lbl_ann_file_count.setText(f"{checked} / {total} selected")

    def _on_ann_select_all(self) -> None:
        for i in range(self.ann_file_list.count()):
            self.ann_file_list.item(i).setCheckState(QtCore.Qt.Checked)
        self._update_ann_file_count()

    def _on_ann_deselect_all(self) -> None:
        for i in range(self.ann_file_list.count()):
            self.ann_file_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        self._update_ann_file_count()

    def _set_ann_status(self, msg: str, error: bool = False,
                        dim: bool = False) -> None:
        if error:
            color = "#ff4444"
        elif dim:
            color = theme.current()["qt_text_dim"]
        else:
            color = theme.current()["qt_text"]
        self.lbl_ann_status.setStyleSheet(
            f"color: {color}; font-size: 8pt;"
        )
        self.lbl_ann_status.setText(msg)
        QtWidgets.QApplication.processEvents()

    def _on_export_annotations(self, fmt: str) -> None:
        if not self._ann_csv_dir:
            self._set_ann_status(
                "Please select a CSV directory first.", error=True
            )
            return

        out_dir = self._ann_output_dir or self._ann_csv_dir
        if not os.path.isdir(out_dir):
            self._set_ann_status(
                f"Output directory does not exist: {out_dir}", error=True
            )
            return

        files_to_convert = [
            self.ann_file_list.item(i).text()
            for i in range(self.ann_file_list.count())
            if self.ann_file_list.item(i).checkState() == QtCore.Qt.Checked
        ]
        if not files_to_convert:
            self._set_ann_status("No files selected.", error=True)
            return

        # Resolve group_by
        gb_text = self.combo_group_by.currentText()
        group_by = "start_datetime_utc" if gb_text == "(all in one file)" else gb_text
        all_in_one = gb_text == "(all in one file)"

        from dasexplorer.core.annotation_export import export_yolo, export_coco, export_raven

        # Disable buttons during export
        for btn in (self.btn_yolo, self.btn_coco, self.btn_raven):
            btn.setEnabled(False)
        self.ann_progress_bar.setMaximum(len(files_to_convert))
        self.ann_progress_bar.setValue(0)

        all_errors: list = []
        n_total_files = 0

        for n, fname in enumerate(files_to_convert, start=1):
            csv_path = os.path.join(self._ann_csv_dir, fname)
            # Each CSV exports into a per-stem subdirectory to avoid collisions
            stem = os.path.splitext(fname)[0]
            file_out_dir = os.path.join(out_dir, stem) if len(files_to_convert) > 1 else out_dir

            self._set_ann_status(
                f"[{n}/{len(files_to_convert)}] Exporting {fname}…", dim=True
            )

            try:
                if all_in_one:
                    # Override group_by with a constant to put everything in one file
                    if fmt == "yolo":
                        n_files, errors = export_yolo(csv_path, file_out_dir,
                                                       group_by="__all__")
                    elif fmt == "coco":
                        n_files, errors = export_coco(csv_path, file_out_dir,
                                                       group_by="__all__")
                    else:
                        n_files, errors = export_raven(csv_path, file_out_dir,
                                                        group_by="__all__")
                else:
                    if fmt == "yolo":
                        n_files, errors = export_yolo(
                            csv_path, file_out_dir, group_by=group_by
                        )
                    elif fmt == "coco":
                        n_files, errors = export_coco(
                            csv_path, file_out_dir, group_by=group_by
                        )
                    else:
                        n_files, errors = export_raven(
                            csv_path, file_out_dir, group_by=group_by
                        )
                n_total_files += n_files
                all_errors.extend([f"{fname}: {e}" for e in errors])
            except Exception as exc:
                all_errors.append(f"{fname}: {exc}")

            self.ann_progress_bar.setValue(n)

        for btn in (self.btn_yolo, self.btn_coco, self.btn_raven):
            btn.setEnabled(True)

        fmt_label = {"yolo": "YOLO", "coco": "COCO JSON", "raven": "Raven CSV"}[fmt]
        n_ok = len(files_to_convert) - sum(
            1 for e in all_errors if not e.endswith("skipped")
        )
        if all_errors:
            self._set_ann_status(
                f"Done: {n_total_files} file(s) written ({fmt_label}). "
                f"Warnings/errors:\n" + "\n".join(all_errors[:5]),
                error=True,
            )
        else:
            self._set_ann_status(
                f"Done: {n_total_files} file(s) exported as {fmt_label} → {out_dir}"
            )
