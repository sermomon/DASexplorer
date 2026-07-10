"""
Batch Data Conversion dialog for DAS Explorer.
Conversion → Batch Conversion → Data
"""

import json
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets

from dasexplorer.core.readers import INTERROGATOR_LABELS, INTERROGATOR_TYPES, read_das_file
from dasexplorer.gui import theme

_FILE_EXTENSIONS = {
    "hdas2.5":   [".bin"],
    "optasense": [".h5", ".hdf5"],
}


class BatchDataDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Conversion — Data")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.50), int(screen.height() * 0.65))
        self._input_dir = ""
        self._output_dir = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── Input directory + interrogator ──────────────────────────────
        dir_row = QtWidgets.QHBoxLayout()
        lbl_in = QtWidgets.QLabel("Input directory:")
        lbl_in.setStyleSheet("font-weight: bold;")
        dir_row.addWidget(lbl_in)
        self.lbl_input_dir = QtWidgets.QLineEdit()
        self.lbl_input_dir.setPlaceholderText("No directory selected…")
        self.lbl_input_dir.setReadOnly(True)
        dir_row.addWidget(self.lbl_input_dir, 1)
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.setMinimumWidth(100)
        btn_browse.clicked.connect(self._on_browse_input)
        dir_row.addWidget(btn_browse)
        lbl_intr = QtWidgets.QLabel("Interrogator:")
        lbl_intr.setStyleSheet("font-weight: bold;")
        dir_row.addWidget(lbl_intr)
        self.combo_interrogator = QtWidgets.QComboBox()
        for label in INTERROGATOR_LABELS:
            self.combo_interrogator.addItem(label)
        self.combo_interrogator.setMinimumWidth(110)
        self.combo_interrogator.currentIndexChanged.connect(self._refresh_file_list)
        dir_row.addWidget(self.combo_interrogator)
        layout.addLayout(dir_row)

        # ── File list ───────────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("Files to convert:"))
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setUniformItemSizes(True)
        layout.addWidget(self.file_list, 1)

        sel_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select All")
        btn_all.clicked.connect(self._on_select_all)
        btn_none = QtWidgets.QPushButton("Deselect All")
        btn_none.clicked.connect(self._on_deselect_all)
        self.lbl_count = QtWidgets.QLabel("0 files")
        self.lbl_count.setStyleSheet(f"color: {theme.current()['qt_text_dim']};")
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        sel_row.addWidget(self.lbl_count)
        layout.addLayout(sel_row)

        # ── Output directory ────────────────────────────────────────────
        out_row = QtWidgets.QHBoxLayout()
        lbl_out = QtWidgets.QLabel("Output directory:")
        lbl_out.setStyleSheet("font-weight: bold;")
        out_row.addWidget(lbl_out)
        self.lbl_output_dir = QtWidgets.QLineEdit()
        self.lbl_output_dir.setPlaceholderText("Same as input directory")
        self.lbl_output_dir.setReadOnly(True)
        out_row.addWidget(self.lbl_output_dir, 1)
        btn_out = QtWidgets.QPushButton("Browse")
        btn_out.setMinimumWidth(100)
        btn_out.clicked.connect(self._on_browse_output)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # ── Convert buttons ─────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_npz = QtWidgets.QPushButton("Convert to NPZ")
        self.btn_npz.setMinimumHeight(32)
        self.btn_npz.clicked.connect(lambda: self._on_convert("npz"))
        self.btn_mat = QtWidgets.QPushButton("Convert to MAT")
        self.btn_mat.setMinimumHeight(32)
        self.btn_mat.clicked.connect(lambda: self._on_convert("mat"))
        btn_row.addWidget(self.btn_npz)
        btn_row.addWidget(self.btn_mat)
        layout.addLayout(btn_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet(
            f"color: {theme.current()['qt_text_dim']}; font-size: 8pt;"
        )
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    def _on_browse_input(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select input directory", self._input_dir or ""
        )
        if not d:
            return
        self._input_dir = d
        self.lbl_input_dir.setText(d)
        if not self._output_dir:
            self._output_dir = d
            self.lbl_output_dir.setText(d)
        self._refresh_file_list()

    def _on_browse_output(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory",
            self._output_dir or self._input_dir or ""
        )
        if d:
            self._output_dir = d
            self.lbl_output_dir.setText(d)

    def _refresh_file_list(self) -> None:
        self.file_list.clear()
        if not self._input_dir or not os.path.isdir(self._input_dir):
            return
        intr = INTERROGATOR_TYPES[self.combo_interrogator.currentIndex()]
        exts = _FILE_EXTENSIONS.get(intr, [])
        files = sorted([
            f for f in os.listdir(self._input_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        for fname in files:
            item = QtWidgets.QListWidgetItem(fname)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.file_list.addItem(item)
        self.file_list.itemChanged.connect(lambda _: self._update_count())
        self._update_count()
        self._set_status(f"{len(files)} file(s) found.", dim=True)

    def _update_count(self) -> None:
        checked = sum(
            1 for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == QtCore.Qt.Checked
        )
        self.lbl_count.setText(f"{checked} / {self.file_list.count()} selected")

    def _on_select_all(self) -> None:
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(QtCore.Qt.Checked)

    def _on_deselect_all(self) -> None:
        for i in range(self.file_list.count()):
            self.file_list.item(i).setCheckState(QtCore.Qt.Unchecked)

    def _set_status(self, msg: str, error: bool = False, dim: bool = False) -> None:
        color = "#ff4444" if error else (
            theme.current()["qt_text_dim"] if dim else theme.current()["qt_text"]
        )
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
        files = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == QtCore.Qt.Checked
        ]
        if not files:
            self._set_status("No files selected.", error=True)
            return

        intr = INTERROGATOR_TYPES[self.combo_interrogator.currentIndex()]
        self.btn_npz.setEnabled(False)
        self.btn_mat.setEnabled(False)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)
        errors = []

        for n, fname in enumerate(files, 1):
            in_path = os.path.join(self._input_dir, fname)
            stem = os.path.splitext(fname)[0]
            out_path = os.path.join(out_dir, f"{stem}.{fmt}")
            self._set_status(f"[{n}/{len(files)}] Reading {fname}…", dim=True)
            try:
                ds = read_das_file(in_path, intr)
            except Exception as exc:
                errors.append(f"{fname}: read error — {exc}")
                self.progress_bar.setValue(n)
                continue

            self._set_status(f"[{n}/{len(files)}] Writing {stem}.{fmt}…", dim=True)
            start_iso = (
                ds.start_datetime_utc.isoformat()
                if ds.start_datetime_utc is not None else ""
            )
            meta_json = json.dumps(ds.metadata or {})
            try:
                if fmt == "npz":
                    np.savez_compressed(
                        out_path,
                        tr=ds.tr, dist_m=ds.dist_m, time_s=ds.time_s,
                        fs_hz=np.float64(ds.fs_hz),
                        start_datetime_utc=start_iso,
                        filename=ds.filename or fname,
                        interrogator=ds.interrogator or intr,
                        downsample=np.int64(ds.downsample or 1),
                        units=ds.units or "",
                        metadata_json=meta_json,
                    )
                else:
                    import scipy.io as sio
                    sio.savemat(out_path, dict(
                        tr=ds.tr, dist_m=ds.dist_m, time_s=ds.time_s,
                        fs_hz=float(ds.fs_hz),
                        start_datetime_utc=start_iso,
                        filename=ds.filename or fname,
                        interrogator=ds.interrogator or intr,
                        downsample=int(ds.downsample or 1),
                        units=ds.units or "",
                        metadata_json=meta_json,
                    ), do_compression=True)
            except Exception as exc:
                errors.append(f"{fname}: write error — {exc}")
            self.progress_bar.setValue(n)

        self.btn_npz.setEnabled(True)
        self.btn_mat.setEnabled(True)
        n_ok = len(files) - len(errors)
        if errors:
            self._set_status(
                f"Done: {n_ok}/{len(files)} converted. Errors:\n" + "\n".join(errors),
                error=True,
            )
        else:
            self._set_status(
                f"Done: {n_ok} file(s) converted to {fmt.upper()} → {out_dir}"
            )
