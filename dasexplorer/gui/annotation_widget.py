"""
Annotation panel widget for DAS Explorer.
Four annotation modes: BBox, OBB, Keypoints, Line.
"""

import os
from PyQt5 import QtWidgets, QtCore, QtGui

from dasexplorer.core.annotations import AnnotationModel, AnnType, ANN_LABEL
from dasexplorer.gui import theme


class AnnotationWidget(QtWidgets.QWidget):
    annotation_mode_changed = QtCore.pyqtSignal(bool, str)
    annotation_selected     = QtCore.pyqtSignal(int)
    delete_requested        = QtCore.pyqtSignal(int)
    save_requested          = QtCore.pyqtSignal()
    clear_requested         = QtCore.pyqtSignal()
    export_path_changed     = QtCore.pyqtSignal(str)
    csv_file_selected       = QtCore.pyqtSignal(str)
    id_changed              = QtCore.pyqtSignal(int, str)   # (flat_idx, new_id)

    # Same colour for all annotation type buttons (yellow, same as annotation pens)
    _BTN_COLORS = {
        AnnType.BBOX: ("#e0a020", "#5a3a00"),
        AnnType.OBB:  ("#e0a020", "#5a3a00"),
        AnnType.KP:   ("#e0a020", "#5a3a00"),
        AnnType.LINE: ("#e0a020", "#5a3a00"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_type: AnnType = None
        self._export_dir: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Four annotation mode buttons ────────────────────────────────
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(3)

        self._ann_buttons = {}
        configs = [
            (AnnType.BBOX, "BBox",      "Draw axis-aligned bounding box (click × 2)",           0, 0),
            (AnnType.OBB,  "OBBox",     "Draw oriented bounding box (3 clicks: v1, v2, width)",  0, 1),
            (AnnType.KP,   "Keypoints", "Place keypoints (click × N, Enter to confirm)",          1, 0),
            (AnnType.LINE, "Line",      "Draw LineString (click × N, Enter to confirm)",           1, 1),
        ]
        for ann_type, label, tooltip, row, col in configs:
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setMinimumHeight(28)
            btn.toggled.connect(lambda checked, t=ann_type: self._on_mode_toggled(checked, t))
            self._ann_buttons[ann_type] = btn
            btn_grid.addWidget(btn, row, col)

        layout.addLayout(btn_grid)

        # ── Events table ────────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("Events:"))
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Type", "ID", "Time [s]", "Distance [m]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        # Allow double-click editing only on column 1 (ID)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.setAlternatingRowColors(False)
        self.table.setStyleSheet(self._table_stylesheet())
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(80)
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        layout.addWidget(self.table, 1)

        # ── Save / Clear ────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_save  = QtWidgets.QPushButton("Save CSV")
        self.btn_clear = QtWidgets.QPushButton("Clear All")
        self.btn_save.clicked.connect(self.save_requested)
        self.btn_clear.clicked.connect(self._on_clear_clicked)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_clear)
        layout.addLayout(btn_row)

        # ── Export path ─────────────────────────────────────────────────
        path_row = QtWidgets.QHBoxLayout()
        path_row.setSpacing(4)
        self.lbl_export_path = QtWidgets.QLabel("<i>—</i>")
        self.lbl_export_path.setStyleSheet(
            f"color: {theme.current()['qt_text']}; font-size: 8pt;"
        )
        self.lbl_export_path.setWordWrap(True)
        self.lbl_export_path.setToolTip("Export directory for CSV files")
        self.btn_change_path = QtWidgets.QPushButton("…")
        self.btn_change_path.setFixedWidth(28)
        self.btn_change_path.setToolTip("Change export directory")
        self.btn_change_path.clicked.connect(self._on_change_path)
        path_row.addWidget(self.lbl_export_path, 1)
        path_row.addWidget(self.btn_change_path, 0)
        layout.addLayout(path_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def annotation_mode(self) -> bool:
        return self._active_type is not None

    @property
    def active_ann_type(self) -> AnnType:
        return self._active_type

    def _table_stylesheet(self) -> str:
        th = theme.current()
        return (
            f"QTableWidget {{ background-color: {th['qt_bg_alt']}; "
            f"gridline-color: {th['qt_border']}; }}"
            f"QTableWidget::item {{ background-color: {th['qt_bg_alt']}; "
            f"color: {th['qt_text']}; }}"
            f"QTableWidget::item:selected {{ background-color: {th['qt_select_bg']}; "
            f"color: {th['qt_text']}; }}"
            f"QHeaderView::section {{ background-color: {th['qt_panel']}; "
            f"color: {th['qt_text_dim']}; border: 1px solid {th['qt_border']}; "
            f"padding: 2px; }}"
        )

    def apply_theme(self) -> None:
        self.table.setStyleSheet(self._table_stylesheet())
        self.lbl_export_path.setStyleSheet(
            f"color: {theme.current()['qt_text']}; font-size: 8pt;"
        )

    def set_export_dir(self, directory: str) -> None:
        self._export_dir = directory
        self.lbl_export_path.setText(directory or "<i>—</i>")
        self.lbl_export_path.setStyleSheet(
            f"color: {theme.current()['qt_text']}; font-size: 8pt;"
        )

    def refresh_csv_list(self, directory: str) -> None:
        pass   # removed from UI

    def highlight_csv(self, filename: str) -> None:
        pass   # removed from UI

    def refresh_table(self, models: dict) -> None:
        """
        Rebuild the annotation table from a dict of {AnnType: AnnotationModel}.
        """
        self.table.setRowCount(0)
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = models.get(ann_type)
            if model is None:
                continue
            for ann in model:
                row = self.table.rowCount()
                self.table.insertRow(row)

                # Column 0 (Type): not editable
                type_item = QtWidgets.QTableWidgetItem(ANN_LABEL[ann_type])
                type_item.setFlags(type_item.flags() & ~QtCore.Qt.ItemIsEditable)
                fg_color, bg_color = self._BTN_COLORS[ann_type]
                type_item.setForeground(QtGui.QColor(fg_color))
                self.table.setItem(row, 0, type_item)

                # Column 1 (ID): editable
                id_item = QtWidgets.QTableWidgetItem(ann.id)
                self.table.setItem(row, 1, id_item)

                # Time / Distance columns depend on type
                if ann_type == AnnType.BBOX:
                    time_str = f"{ann.t0:.2f} – {ann.t1:.2f}"
                    dist_str = f"{ann.d0:.0f} – {ann.d1:.0f}"
                elif ann_type == AnnType.OBB:
                    time_str = f"{ann.cx_t:.2f} ±{ann.w_t:.2f}"
                    dist_str = f"{ann.cy_d:.0f} ±{ann.h_d:.0f}"
                elif ann_type == AnnType.KP:
                    import json as _json
                    pts = list(zip(_json.loads(ann.kp_t), _json.loads(ann.kp_d)))
                    time_str = f"{len(pts)} pts"
                    dist_str = ""
                else:   # LINE
                    import json as _json
                    pts = list(zip(_json.loads(ann.pts_t), _json.loads(ann.pts_d)))
                    time_str = f"{len(pts)} verts"
                    dist_str = ""

                time_item = QtWidgets.QTableWidgetItem(time_str)
                time_item.setFlags(time_item.flags() & ~QtCore.Qt.ItemIsEditable)
                dist_item = QtWidgets.QTableWidgetItem(dist_str)
                dist_item.setFlags(dist_item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(row, 2, time_item)
                self.table.setItem(row, 3, dist_item)
                # Store (ann_type, local_idx) in the type cell for identification
                idx_in_model = list(model).index(ann)
                type_item.setData(QtCore.Qt.UserRole, (ann_type, idx_in_model))

    def set_button_active(self, ann_type: AnnType, active: bool) -> None:
        """Externally update button state (called from MainWindow)."""
        btn = self._ann_buttons.get(ann_type)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(active)
        fg, bg = self._BTN_COLORS[ann_type]
        if active:
            btn.setStyleSheet(
                f"background-color: {bg}; border: 1px solid {fg}; color: {fg};"
            )
        else:
            btn.setStyleSheet("")
        btn.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Emit id_changed when the user edits the ID cell (column 1)."""
        if item.column() != 1:
            return
        row = item.row()
        type_item = self.table.item(row, 0)
        if type_item is None:
            return
        data = type_item.data(QtCore.Qt.UserRole)
        if data is None:
            return
        ann_type, local_idx = data
        # Compute flat_idx: sum of all models before this type + local_idx
        from dasexplorer.core.annotations import AnnType as AT
        order = (AT.BBOX, AT.OBB, AT.KP, AT.LINE)
        # We don't have access to _ann_models here, so emit (ann_type, local_idx, new_id)
        # and let main_window resolve; we reuse the UserRole tuple
        self.id_changed.emit(row, item.text())   # row is the flat table index

    def _on_mode_toggled(self, checked: bool, ann_type: AnnType) -> None:
        if checked:
            # Deactivate all other buttons
            for t, btn in self._ann_buttons.items():
                if t != ann_type:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.setStyleSheet("")
                    btn.blockSignals(False)
            self._active_type = ann_type
            fg, bg = self._BTN_COLORS[ann_type]
            self._ann_buttons[ann_type].setStyleSheet(
                f"background-color: {bg}; border: 1px solid {fg}; color: {fg};"
            )
        else:
            self._active_type = None
            self._ann_buttons[ann_type].setStyleSheet("")
        self.annotation_mode_changed.emit(checked, ann_type.value)

    def _on_table_selection(self) -> None:
        rows = self.table.selectedItems()
        if rows:
            self.annotation_selected.emit(self.table.currentRow())

    def _on_table_context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        menu = QtWidgets.QMenu(self)
        delete_action = menu.addAction("Delete")
        action = menu.exec_(self.table.viewport().mapToGlobal(pos))
        if action == delete_action:
            self.delete_requested.emit(row)

    def _on_clear_clicked(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, "Clear annotations",
            "Delete ALL annotations (all types) for this file?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.clear_requested.emit()

    def _on_change_path(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select export directory", self._export_dir or ""
        )
        if directory:
            self.set_export_dir(directory)
            self.export_path_changed.emit(directory)

