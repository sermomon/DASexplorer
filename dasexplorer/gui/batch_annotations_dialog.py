"""
Batch Annotations Export dialog for DAS Explorer.
Conversion → Batch Conversion → Annotations
"""

import os

from PyQt5 import QtCore, QtWidgets

from dasexplorer.core.annotation_export import export_yolo, export_coco, export_raven
from dasexplorer.gui import theme


class BatchAnnotationsDialog(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Conversion — Annotations")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.50), int(screen.height() * 0.65))
        self._csv_dir = ""
        self._output_dir = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # ── Input CSV directory ─────────────────────────────────────────
        csv_row = QtWidgets.QHBoxLayout()
        lbl_csv = QtWidgets.QLabel("Input CSV:")
        lbl_csv.setStyleSheet("font-weight: bold;")
        csv_row.addWidget(lbl_csv)
        self.lbl_csv_dir = QtWidgets.QLineEdit()
        self.lbl_csv_dir.setPlaceholderText("No directory selected…")
        self.lbl_csv_dir.setReadOnly(True)
        csv_row.addWidget(self.lbl_csv_dir, 1)
        btn_browse_csv = QtWidgets.QPushButton("Browse")
        btn_browse_csv.setMinimumWidth(100)
        btn_browse_csv.clicked.connect(self._on_browse_csv)
        csv_row.addWidget(btn_browse_csv)
        layout.addLayout(csv_row)

        # ── File list ───────────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("CSV files to convert:"))
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

        # ── Group by ───────────────────────────────────────────────────
        gb_row = QtWidgets.QHBoxLayout()
        lbl_gb = QtWidgets.QLabel("Group by:")
        lbl_gb.setStyleSheet("font-weight: bold;")
        gb_row.addWidget(lbl_gb)
        self.combo_group_by = QtWidgets.QComboBox()
        self.combo_group_by.addItems([
            "start_datetime_utc",
            "id",
            "(all in one file)",
        ])
        self.combo_group_by.setToolTip(
            "start_datetime_utc — one output file per DAS acquisition window (recommended)\n"
            "id — one output file per label/class\n"
            "(all in one file) — single output file with all annotations"
        )
        gb_row.addWidget(self.combo_group_by, 1)
        layout.addLayout(gb_row)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(sep)

        # ── Export buttons ──────────────────────────────────────────────
        lbl_fmt = QtWidgets.QLabel("Export format:")
        lbl_fmt.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_fmt)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_yolo = QtWidgets.QPushButton("Export YOLO")
        self.btn_yolo.setMinimumHeight(32)
        self.btn_yolo.setToolTip(
            "One .txt per image group + classes.txt\n"
            "Each line: <class_id> <x_center> <y_center> <width> <height>  (normalised [0, 1])\n"
            "String IDs are mapped to integers automatically."
        )
        self.btn_yolo.clicked.connect(lambda: self._on_export("yolo"))

        self.btn_coco = QtWidgets.QPushButton("Export COCO JSON")
        self.btn_coco.setMinimumHeight(32)
        self.btn_coco.setToolTip(
            "Single COCO-format .json per CSV.\n"
            "Compatible with Detectron2, MMDetection, RT-DETR.\n"
            "Bounding boxes in absolute pixel coordinates [x, y, w, h]."
        )
        self.btn_coco.clicked.connect(lambda: self._on_export("coco"))

        self.btn_raven = QtWidgets.QPushButton("Export Raven")
        self.btn_raven.setMinimumHeight(32)
        self.btn_raven.setToolTip(
            "Raven Pro / PAMGuard Selection Table (tab-separated CSV).\n"
            "Output: {stem}_raven.csv — _raven suffix prevents collision\n"
            "with the original DAS Explorer annotation CSV."
        )
        self.btn_raven.clicked.connect(lambda: self._on_export("raven"))

        btn_row.addWidget(self.btn_yolo)
        btn_row.addWidget(self.btn_coco)
        btn_row.addWidget(self.btn_raven)
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

    def _on_browse_csv(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select CSV directory", self._csv_dir or ""
        )
        if not d:
            return
        self._csv_dir = d
        self.lbl_csv_dir.setText(d)
        if not self._output_dir:
            self._output_dir = d
            self.lbl_output_dir.setText(d)
        self._refresh_file_list()

    def _on_browse_output(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory",
            self._output_dir or self._csv_dir or ""
        )
        if d:
            self._output_dir = d
            self.lbl_output_dir.setText(d)

    def _refresh_file_list(self) -> None:
        self.file_list.clear()
        if not self._csv_dir or not os.path.isdir(self._csv_dir):
            return
        files = sorted(
            f for f in os.listdir(self._csv_dir)
            if f.lower().endswith(".csv")
        )
        for fname in files:
            item = QtWidgets.QListWidgetItem(fname)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.file_list.addItem(item)
        self.file_list.itemChanged.connect(lambda _: self._update_count())
        self._update_count()
        self._set_status(f"{len(files)} CSV file(s) found.", dim=True)

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

    def _set_status(self, msg: str, error: bool = False,
                    dim: bool = False) -> None:
        color = "#ff4444" if error else (
            theme.current()["qt_text_dim"] if dim else theme.current()["qt_text"]
        )
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 8pt;")
        self.lbl_status.setText(msg)
        QtWidgets.QApplication.processEvents()

    def _on_export(self, fmt: str) -> None:
        if not self._csv_dir:
            self._set_status("Please select a CSV directory first.", error=True)
            return
        out_dir = self._output_dir or self._csv_dir
        if not os.path.isdir(out_dir):
            self._set_status(
                f"Output directory does not exist: {out_dir}", error=True
            )
            return
        files = [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
            if self.file_list.item(i).checkState() == QtCore.Qt.Checked
        ]
        if not files:
            self._set_status("No files selected.", error=True)
            return

        gb_text = self.combo_group_by.currentText()
        group_by = "__all__" if gb_text == "(all in one file)" else gb_text

        for btn in (self.btn_yolo, self.btn_coco, self.btn_raven):
            btn.setEnabled(False)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)

        all_errors = []
        n_total = 0

        for n, fname in enumerate(files, 1):
            csv_path = os.path.join(self._csv_dir, fname)
            stem = os.path.splitext(fname)[0]
            file_out = os.path.join(out_dir, stem) if len(files) > 1 else out_dir
            self._set_status(f"[{n}/{len(files)}] Exporting {fname}…", dim=True)
            try:
                if fmt == "yolo":
                    n_files, errors = export_yolo(csv_path, file_out, group_by)
                elif fmt == "coco":
                    n_files, errors = export_coco(csv_path, file_out, group_by)
                else:
                    n_files, errors = export_raven(csv_path, file_out, group_by=group_by)
                n_total += n_files
                all_errors.extend(f"{fname}: {e}" for e in errors)
            except Exception as exc:
                all_errors.append(f"{fname}: {exc}")
            self.progress_bar.setValue(n)

        for btn in (self.btn_yolo, self.btn_coco, self.btn_raven):
            btn.setEnabled(True)

        fmt_label = {"yolo": "YOLO", "coco": "COCO JSON", "raven": "Raven"}[fmt]
        if all_errors:
            self._set_status(
                f"Done: {n_total} file(s) written ({fmt_label}). "
                f"Warnings:\n" + "\n".join(all_errors[:5]),
                error=True,
            )
        else:
            self._set_status(
                f"Done: {n_total} file(s) exported as {fmt_label} → {out_dir}"
            )
