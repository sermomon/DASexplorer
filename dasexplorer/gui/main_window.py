"""
Main window for DASexplorer.

Left panel: Data (file browser + controls) + View (ranges + filter) + Annotations.
Main area: tabbed Raw / FK / RGB waterfall views.
"""

import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

from dasexplorer.core.annotations import (
    AnnotationModel, AnnType, ANN_LABEL, ANN_SUFFIX,
    BBoxAnnotation, OBBAnnotation, KeypointAnnotation, LineAnnotation,
)
from dasexplorer.core.config import get_interrogator_defaults, get_ui_defaults
from dasexplorer.core.readers import (
    INTERROGATOR_TYPES,  # kept for interrogator-type dispatch
    read_das_file, generate_synthetic_dataset,
)
from dasexplorer.gui.annotation_widget import AnnotationWidget
from dasexplorer.gui.analysis_dialogs import (
    SpectrogramDialog, SpectralDialog, SignalDialog,
    SignalFreqDialog, SignalEnvelopeDialog, SignalPhaseDialog,
    VelocityDialog,
)
from dasexplorer.gui.waterfall import WaterfallWidget
from dasexplorer.gui.tab_bar import DASTabBar
from dasexplorer.gui import theme
from dasexplorer.version import __version__

# File extensions fallback (profiles in config.json override these).
_FALLBACK_EXTENSIONS = {
    "hdas2.5":   [".bin"],
    "optasense": [".h5", ".hdf5"],
    "optodas":   [".hdf5"],
}

STRIDE_VALUES = [1, 2, 3, 4, 5, 10]


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setWindowTitle(f"DASexplorer v{__version__}")
        self.setWindowTitle(f"DASexplorer")
        self.resize(1400, 800)

        # Window icon (title bar, Alt+Tab) — distinct from the taskbar icon
        # set on QApplication in main.py (icon_2.ico).
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "icons", "icon_1.ico",
        )
        if os.path.isfile(icon_path):
            self.setWindowIcon(QtGui.QIcon(icon_path))

        self.dataset      = None
        self._current_dir = None
        self._file_list   = []
        # One model per annotation type
        self._ann_models = {
            AnnType.BBOX: AnnotationModel(AnnType.BBOX),
            AnnType.OBB:  AnnotationModel(AnnType.OBB),
            AnnType.KP:   AnnotationModel(AnnType.KP),
            AnnType.LINE: AnnotationModel(AnnType.LINE),
        }
        # Keep _ann_model as alias to BBOX for backwards compat (VelocityDialog etc.)
        self._ann_model = self._ann_models[AnnType.BBOX]
        self._export_dir  = ""
        self._current_data_path: str = ""
        self._envelope: bool = False
        self._envelope_fk: bool = False
        self._tr_fk_base: np.ndarray = None
        self._rgb_array: np.ndarray = None
        self._default_view: str = "raw"
        self._last_tab_index: int = 0
        self._build_ui()

        # Apply default interrogator from config (after UI is built so
        # combo_interrogator exists). Suppress the currentIndexChanged signal
        # to avoid triggering a file-list refresh before any file is loaded.
        from dasexplorer.core.config import get_default_profile_key, get_all_profiles as _gap
        _pkeys = list(_gap().keys())
        _def   = get_default_profile_key()
        if _def in _pkeys:
            self.combo_interrogator.blockSignals(True)
            self.combo_interrogator.setCurrentIndex(_pkeys.index(_def))
            self.combo_interrogator.blockSignals(False)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_menu_bar()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        self.left_widget = self._build_left_panel()
        main_area        = self._build_main_area()

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(4)
        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(main_area)
        self.splitter.setChildrenCollapsible(False)
        self.left_widget.setMinimumWidth(220)

        root_layout.addWidget(self.splitter)

        self._build_status_bar()

        # Connect waterfall draw signals — must be done after all three
        # waterfall widgets are created (they live inside _build_main_area)
        for wf in self._waterfalls():
            wf.bbox_drawn.connect(self._on_bbox_drawn)
            wf.obb_drawn.connect(self._on_obb_drawn)
            wf.kp_drawn.connect(self._on_kp_drawn)
            wf.line_drawn.connect(self._on_line_drawn)
            wf.bbox_edited.connect(self._on_bbox_edited)
            wf.obb_edited.connect(self._on_obb_edited)
            wf.kp_edited.connect(self._on_kp_edited)
            wf.line_edited.connect(self._on_line_edited)

    def _build_menu_bar(self) -> None:
        """
        Top menu bar (File, Edit, View, Settings, Analysis, Conversion) plus
        Info pinned to the top-right corner, above the colour histogram
        column. Most items are placeholders for future functionality.
        """
        menubar = self.menuBar()
        menubar.setStyleSheet(
            "QMenuBar::item {"
            "  padding: 8px 24px;"
            "  margin: 0px;"
            "}"
        )

        menu_file = menubar.addMenu("File")
        act_open = menu_file.addAction("Open file…")
        act_open.triggered.connect(self.load_file_dialog)
        menu_file.addSeparator()
        act_save_npz = menu_file.addAction("Save as NPZ")
        act_save_npz.triggered.connect(self._on_save_as_npz)
        act_import_npz = menu_file.addAction("Import from NPZ")
        act_import_npz.triggered.connect(self._on_import_from_npz)
        menu_file.addSeparator()
        act_save_mat = menu_file.addAction("Save as MAT")
        act_save_mat.triggered.connect(self._on_save_as_mat)
        act_import_mat = menu_file.addAction("Import from MAT")
        act_import_mat.triggered.connect(self._on_import_from_mat)
        menu_file.addSeparator()
        act_refresh = menu_file.addAction("Refresh")
        act_refresh.triggered.connect(self._on_menu_refresh)
        menu_file.addSeparator()
        act_exit = menu_file.addAction("Exit")
        act_exit.triggered.connect(self.close)

        menu_edit = menubar.addMenu("Edit")
        act_undo  = menu_edit.addAction("Undo")
        act_redo  = menu_edit.addAction("Redo")
        menu_edit.addSeparator()
        act_prefs = menu_edit.addAction("Preferences")
        for a in (act_undo, act_redo, act_prefs):
            a.setEnabled(False)

        menu_view = menubar.addMenu("View")
        act_full_view = menu_view.addAction("Full View")
        act_full_view.triggered.connect(self._on_menu_full_view)
        act_refresh_layout = menu_view.addAction("Refresh Layout")
        act_refresh_layout.triggered.connect(self._on_menu_refresh)
        menu_view.addSeparator()
        menu_theme = menu_view.addMenu("Theme")
        self._theme_action_group = QtWidgets.QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        act_theme_dark = menu_theme.addAction("Dark")
        act_theme_dark.setCheckable(True)
        act_theme_light = menu_theme.addAction("Light")
        act_theme_light.setCheckable(True)
        self._theme_action_group.addAction(act_theme_dark)
        self._theme_action_group.addAction(act_theme_light)
        act_theme_dark.triggered.connect(lambda: self._on_theme_changed("dark"))
        act_theme_light.triggered.connect(lambda: self._on_theme_changed("light"))
        self._act_theme_dark = act_theme_dark
        self._act_theme_light = act_theme_light
        if theme.current()["name"] == "light":
            act_theme_light.setChecked(True)
        else:
            act_theme_dark.setChecked(True)

        menu_settings = menubar.addMenu("Settings")
        act_interrogators = menu_settings.addAction("Configuration Profile")
        act_interrogators.triggered.connect(self._on_show_config_profile)
        act_paths = menu_settings.addAction("Data Paths")
        act_paths.setEnabled(False)

        menu_analysis = menubar.addMenu("Analysis")
        act_batch     = menu_analysis.addAction("Batch Processing")
        act_pipeline  = menu_analysis.addAction("Pipeline Manager")
        for a in (act_batch, act_pipeline):
            a.setEnabled(False)

        menu_conversion = menubar.addMenu("Conversion")
        menu_batch = menu_conversion.addMenu("Batch Conversion")
        act_batch_data = menu_batch.addAction("Data")
        act_batch_data.triggered.connect(self._on_show_batch_data)
        act_batch_ann = menu_batch.addAction("Annotations")
        act_batch_ann.triggered.connect(self._on_show_batch_annotations)

        # Info pinned to the top-right corner — separated from the other
        # menus, sitting right above the colour histogram column.
        info_bar = QtWidgets.QMenuBar(menubar)
        info_bar.setStyleSheet(menubar.styleSheet())
        menu_info = info_bar.addMenu("Help")
        act_docs    = menu_info.addAction("Documentation")
        act_about   = menu_info.addAction("About DASexplorer")
        act_docs.setEnabled(False)
        act_about.triggered.connect(self._show_about_dialog)
        menubar.setCornerWidget(info_bar, QtCore.Qt.TopRightCorner)

    def _show_about_dialog(self) -> None:
        about_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "about.txt"
        )
        about_path = os.path.normpath(about_path)
        if os.path.isfile(about_path):
            try:
                with open(about_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                text = f"DASexplorer v{__version__}\n\nCould not read about.txt."
        else:
            text = (
                f"DASexplorer v{__version__}\n\n"
                "Visualization and annotation tool for Distributed Acoustic "
                "Sensing (DAS) data.\n\n"
                "IGIC-UPV — Underwater Acoustics Group\n\n"
                "(Place an about.txt file in the project root to customise this text.)"
            )
        QtWidgets.QMessageBox.about(self, "About DASexplorer", text)

    def _flat_idx(self, ann_type: AnnType, local_idx: int) -> int:
        """Flat index of an annotation in the combined table (across all models)."""
        offset = 0
        for t in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            if t == ann_type:
                return offset + local_idx
            offset += len(self._ann_models[t])
        return local_idx

    def _on_show_config_profile(self) -> None:
        from dasexplorer.gui.config_profile_dialog import ConfigurationProfileDialog
        dlg = ConfigurationProfileDialog(parent=self)
        dlg.exec_()

    def _on_show_batch_data(self) -> None:
        from dasexplorer.gui.batch_data_dialog import BatchDataDialog
        dlg = BatchDataDialog(parent=self)
        dlg.show()

    def _on_show_batch_annotations(self) -> None:
        from dasexplorer.gui.batch_annotations_dialog import BatchAnnotationsDialog
        dlg = BatchAnnotationsDialog(parent=self)
        dlg.show()

    def _on_theme_changed(self, theme_name: str) -> None:
        """
        Switch between Dark and Light themes at runtime, applying the
        change to: the global Qt stylesheet, pyqtgraph's default
        background/foreground (used by any NEW plot widget, e.g. analysis
        dialogs opened from now on), and every already-created plot
        (Raw/F-K/RGB waterfalls + the annotation table), since pyqtgraph
        colours are not controlled by the Qt stylesheet at all. The choice
        is persisted to config.json so it's restored on next launch.
        """
        th = theme.set_current(theme_name)
        pg.setConfigOption("background", th["pg_background"])
        pg.setConfigOption("foreground", th["pg_foreground"])

        app = QtWidgets.QApplication.instance()
        stylesheet = theme.build_stylesheet(th)
        app.setStyleSheet(stylesheet)

        # The corner Info menu has its own QMenuBar with a copied stylesheet
        # (see _build_menu_bar) — the global app stylesheet alone doesn't
        # reach it because it's a separate QMenuBar instance, not a child
        # styled by cascading rules from the main one in all Qt versions.
        menubar = self.menuBar()
        menubar.setStyleSheet(
            "QMenuBar::item {"
            "  padding: 8px 24px;"
            "  margin: 0px;"
            "}"
        )
        corner = menubar.cornerWidget()
        if corner is not None:
            corner.setStyleSheet(menubar.styleSheet())

        for wf in self._waterfalls():
            wf.apply_theme()
        self.ann_widget.apply_theme()

        # Re-apply current status label colour (it's set with a fixed
        # foreground colour per state — neutral/orange/red — but the
        # "neutral" state needs to track the theme's text colour).
        if not self._status_lbl.text():
            self._status_lbl.setStyleSheet(f"color: {th['qt_text']}; padding: 0 6px;")

        for lbl in [self.lbl_channels, self.lbl_spatial,
                    self.lbl_duration, self.lbl_sampling_rate, self.lbl_units]:
            lbl.setStyleSheet(f"color: {th['qt_text_dim']}; padding: 0 8px;")

        from dasexplorer.core.config import set_ui_theme
        set_ui_theme(theme_name)

        action = self._act_theme_dark if theme_name == "dark" else self._act_theme_light
        action.setChecked(True)

        self._status_done(f"Theme switched to {theme_name.capitalize()}.")

    # ------------------------------------------------------------------
    # File menu actions
    # ------------------------------------------------------------------

    def _on_menu_full_view(self) -> None:
        """
        Show the full Time/Distance extent of the currently loaded dataset
        in all three views (Raw/F-K/RGB) — i.e. undo any zoom/crop without
        touching Color/Frequency/F-K/RGB parameters (unlike Refresh, which
        resets everything to the interrogator defaults).
        """
        if self.dataset is None:
            self._status_error("No data loaded.")
            return

        ds = self.dataset
        t0, t1 = float(ds.time_s[0]), float(ds.time_s[-1])
        d0, d1 = float(ds.dist_m[0]), float(ds.dist_m[-1])

        for sb, val in [(self.spin_tmin, t0), (self.spin_tmax, t1),
                         (self.spin_dmin, d0), (self.spin_dmax, d1)]:
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

        for wf in self._waterfalls():
            if wf.dataset is not None:
                wf.apply_time_range(t0, t1)
                wf.apply_distance_range(d0, d1)

        self._status_done("Showing full Time/Distance range.")

    def _on_menu_refresh(self) -> None:
        """
        Reset all View-panel parameters to this interrogator's config.json
        defaults WITHOUT re-reading the file from disk, then re-render —
        equivalent to "as if you had just opened it", but much cheaper than
        an actual reload.
        """
        if self.dataset is None:
            self._status_error("No data loaded.")
            return

        from dasexplorer.core.config import get_all_profiles as _gap5, get_profile as _gp5
        _pkeys5  = list(_gap5().keys())
        _pidx5   = self.combo_interrogator.currentIndex()
        _pkey5   = _pkeys5[_pidx5] if _pidx5 < len(_pkeys5) else _pkeys5[0]
        _pcfg5   = _gp5(_pkey5)
        interrogator = _pcfg5.get("interrogator", INTERROGATOR_TYPES[0])
        cfg = get_interrogator_defaults(interrogator)
        self._apply_default_view_params(self.dataset, cfg)
        self._default_view = str(cfg.get("default_view", "raw")).lower()

        vmin = self.spin_vmin.value()
        vmax = self.spin_vmax.value()
        fmin = self.spin_fmin.value()
        fmax = self.spin_fmax.value()
        nyq  = self.dataset.fs_hz / 2.0
        can_filter = fmin > 0 and fmax > fmin and fmax < nyq

        self.waterfall.load_and_display(
            self.dataset, vmin=vmin, vmax=vmax,
            fmin=fmin if can_filter else None,
            fmax=fmax if can_filter else None,
            envelope=self._envelope,
        )

        # Invalidate F-K / RGB caches — parameters changed back to defaults
        self._tr_fk_base = None
        self._rgb_array = None
        self.waterfall_fk.image_item.clear()
        self.waterfall_rgb.image_item.clear()

        if self._default_view == "fk":
            self._apply_fk()
            self.tab_widget.setCurrentIndex(1)
        else:
            self.tab_widget.setCurrentIndex(0)

        self._status_done("View refreshed to interrogator defaults.")

    def _get_export_arrays(self, export_selected_view: bool):
        """Return (tr, dist_m, time_s) for export: either the full dataset
        arrays, or only the Time/Distance range currently set in the View
        panel (the current zoom/crop). Shared by the NPZ and MAT export
        paths.

        compute_indices() returns the index of the sample closest to each
        of t0/t1/d0/d1 — these are inclusive point indices, not Python
        slice bounds. Slicing with [ti0:ti1] would silently drop the last
        sample on each axis (e.g. selecting the FULL range would still
        lose one row/column). The upper bound must be ti1+1/di1+1 so the
        selected endpoint sample is actually included in the export.
        """
        ds = self.dataset
        if export_selected_view:
            t0, t1 = self.spin_tmin.value(), self.spin_tmax.value()
            d0, d1 = self.spin_dmin.value(), self.spin_dmax.value()
            ti0, ti1, di0, di1 = AnnotationModel.compute_indices(
                t0, t1, d0, d1, ds.time_s, ds.dist_m
            )
            ti1_excl = min(ti1 + 1, ds.n_time)
            di1_excl = min(di1 + 1, ds.n_dist)
            return (ds.tr[di0:di1_excl, ti0:ti1_excl],
                    ds.dist_m[di0:di1_excl],
                    ds.time_s[ti0:ti1_excl])
        return ds.tr, ds.dist_m, ds.time_s

    def _on_save_as_npz(self) -> None:
        """Export the current dataset (full array or only the View-panel
        selected Time/Distance range) plus all metadata needed to reload
        it later, as a .npz file."""
        if self.dataset is None:
            self._status_error("No data loaded.")
            return

        dlg = _SaveExportDialog(self, format_name="NPZ")
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        export_selected_view = dlg.export_selected_view()

        default_name = os.path.splitext(os.path.basename(
            self.dataset.filename or "dataset"
        ))[0] + ".npz"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save as NPZ", default_name, "NumPy archive (*.npz)"
        )
        if not path:
            return

        ds = self.dataset
        tr, dist_m, time_s = self._get_export_arrays(export_selected_view)
        start_iso = (
            ds.start_datetime_utc.isoformat()
            if ds.start_datetime_utc is not None else ""
        )

        self._status_processing("Saving NPZ…")
        QtWidgets.QApplication.processEvents()
        try:
            np.savez_compressed(
                path,
                tr=tr,
                dist_m=dist_m,
                time_s=time_s,
                fs_hz=np.float64(ds.fs_hz),
                start_datetime_utc=start_iso,
                filename=ds.filename or "",
                interrogator=ds.interrogator or "",
                downsample=np.int64(ds.downsample or 1),
                units=ds.units or "",
                metadata_json=__import__("json").dumps(ds.metadata or {}),
            )
        except Exception as exc:
            self._status_error(f"NPZ export error: {exc}")
            return

        self._status_done(f"Saved NPZ: {os.path.basename(path)}")

    def _on_save_as_mat(self) -> None:
        """Same as _on_save_as_npz, but writes MATLAB .mat format for
        interoperability with MATLAB-based workflows."""
        if self.dataset is None:
            self._status_error("No data loaded.")
            return

        dlg = _SaveExportDialog(self, format_name="MAT")
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        export_selected_view = dlg.export_selected_view()

        default_name = os.path.splitext(os.path.basename(
            self.dataset.filename or "dataset"
        ))[0] + ".mat"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save as MAT", default_name, "MATLAB file (*.mat)"
        )
        if not path:
            return

        ds = self.dataset
        tr, dist_m, time_s = self._get_export_arrays(export_selected_view)
        start_iso = (
            ds.start_datetime_utc.isoformat()
            if ds.start_datetime_utc is not None else ""
        )

        self._status_processing("Saving MAT…")
        QtWidgets.QApplication.processEvents()
        try:
            import scipy.io as sio
            sio.savemat(
                path,
                {
                    "tr": tr,
                    "dist_m": dist_m,
                    "time_s": time_s,
                    "fs_hz": float(ds.fs_hz),
                    "start_datetime_utc": start_iso,
                    "filename": ds.filename or "",
                    "interrogator": ds.interrogator or "",
                    "downsample": int(ds.downsample or 1),
                    "units": ds.units or "",
                    "metadata_json": __import__("json").dumps(ds.metadata or {}),
                },
                do_compression=True,
            )
        except Exception as exc:
            self._status_error(f"MAT export error: {exc}")
            return

        self._status_done(f"Saved MAT: {os.path.basename(path)}")

    def _on_import_from_npz(self) -> None:
        """Import a previously exported .npz file and load it like any
        other file (file list, Previous/Next, F-K, RGB, annotations all
        work normally on it)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import from NPZ", "", "NumPy archive (*.npz)"
        )
        if not path:
            return

        self._status_processing("Importing NPZ…")
        QtWidgets.QApplication.processEvents()
        try:
            from dasexplorer.core.readers import read_npz
            dataset = read_npz(path)
        except Exception as exc:
            self._status_error(f"NPZ import error: {exc}")
            return

        self._import_dataset(path, dataset)

    def _on_import_from_mat(self) -> None:
        """Import a previously exported .mat file and load it like any
        other file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import from MAT", "", "MATLAB file (*.mat)"
        )
        if not path:
            return

        self._status_processing("Importing MAT…")
        QtWidgets.QApplication.processEvents()
        try:
            from dasexplorer.core.readers import read_mat
            dataset = read_mat(path)
        except Exception as exc:
            self._status_error(f"MAT import error: {exc}")
            return

        self._import_dataset(path, dataset)

    def _import_dataset(self, path: str, dataset) -> None:
        """Shared tail of the NPZ/MAT import handlers: load the dataset as
        if it were a freshly opened file and add it to the file list."""
        self.dataset = None  # force first-load path (defaults reset)
        self._set_dataset(os.path.basename(path), dataset)
        self._current_dir = os.path.dirname(path)
        self._file_list = [os.path.basename(path)]
        self.file_list_widget.clear()
        self.file_list_widget.addItem(os.path.basename(path))
        self.file_list_widget.setCurrentRow(0)
        self._status_done(f"Imported: {os.path.basename(path)}")


    # --- Left panel ---

    def _build_left_panel(self) -> QtWidgets.QWidget:
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_data_group())
        left_layout.addWidget(self._build_view_group())
        left_layout.addWidget(self._build_annotations_group(), 1)  # stretch=1: fills remaining space
        return left_widget

    # --- Main tabbed area ---

    def _build_main_area(self) -> QtWidgets.QWidget:
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setDocumentMode(True)
        self.tab_widget.setTabBar(DASTabBar())
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Raw tab
        self.waterfall = WaterfallWidget()
        self.waterfall.cursor_info.connect(self._on_cursor_info)
        self.waterfall.roi_edit_requested.connect(self._on_annotation_selected)
        self.waterfall.roi_edit_shape_requested.connect(self._on_annotation_edit_shape)
        self.waterfall.roi_remove_requested.connect(self._on_annotation_delete)
        self.waterfall.roi_spectrogram_requested.connect(self._on_show_spectrogram)
        self.waterfall.roi_spectral_requested.connect(self._on_show_spectral)
        self.waterfall.roi_signal_requested.connect(self._on_show_signal)
        self.waterfall.roi_signal_freq_requested.connect(self._on_show_signal_freq)
        self.waterfall.roi_signal_env_requested.connect(self._on_show_signal_env)
        self.waterfall.roi_signal_phase_requested.connect(self._on_show_signal_phase)
        self.waterfall.roi_velocity_requested.connect(self._on_show_velocity)
        self.tab_widget.addTab(self.waterfall, "Raw")

        # FK tab — real WaterfallWidget sharing the same annotation model
        self.waterfall_fk = WaterfallWidget()
        self.waterfall_fk.cursor_info.connect(self._on_cursor_info)
        self.waterfall_fk.roi_edit_requested.connect(self._on_annotation_selected)
        self.waterfall_fk.roi_edit_shape_requested.connect(self._on_annotation_edit_shape)
        self.waterfall_fk.roi_remove_requested.connect(self._on_annotation_delete)
        self.waterfall_fk.roi_spectrogram_requested.connect(self._on_show_spectrogram)
        self.waterfall_fk.roi_spectral_requested.connect(self._on_show_spectral)
        self.waterfall_fk.roi_signal_requested.connect(self._on_show_signal)
        self.waterfall_fk.roi_signal_freq_requested.connect(self._on_show_signal_freq)
        self.waterfall_fk.roi_signal_env_requested.connect(self._on_show_signal_env)
        self.waterfall_fk.roi_signal_phase_requested.connect(self._on_show_signal_phase)
        self.waterfall_fk.roi_velocity_requested.connect(self._on_show_velocity)
        self.waterfall_fk.annotation_mode_changed_fk = False
        self.tab_widget.addTab(self.waterfall_fk, "F-K")

        # RGB tab — real WaterfallWidget in RGB-image mode, lazy computed
        self.waterfall_rgb = WaterfallWidget()
        self.waterfall_rgb.cursor_info.connect(self._on_cursor_info)
        self.waterfall_rgb.roi_edit_requested.connect(self._on_annotation_selected)
        self.waterfall_rgb.roi_edit_shape_requested.connect(self._on_annotation_edit_shape)
        self.waterfall_rgb.roi_remove_requested.connect(self._on_annotation_delete)
        self.waterfall_rgb.roi_spectrogram_requested.connect(self._on_show_spectrogram)
        self.waterfall_rgb.roi_spectral_requested.connect(self._on_show_spectral)
        self.waterfall_rgb.roi_signal_requested.connect(self._on_show_signal)
        self.waterfall_rgb.roi_signal_freq_requested.connect(self._on_show_signal_freq)
        self.waterfall_rgb.roi_signal_env_requested.connect(self._on_show_signal_env)
        self.waterfall_rgb.roi_signal_phase_requested.connect(self._on_show_signal_phase)
        self.waterfall_rgb.roi_velocity_requested.connect(self._on_show_velocity)
        self.tab_widget.addTab(self.waterfall_rgb, "   ")

        # Live tab (placeholder)
        live_placeholder = QtWidgets.QLabel("Live monitoring — coming soon")
        live_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.tab_widget.addTab(live_placeholder, "Live")

        return self.tab_widget

    # --- Status bar ---

    def _build_status_bar(self) -> None:
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # Coloured message label (left side of status bar)
        self._status_lbl = QtWidgets.QLabel("")
        self._status_lbl.setStyleSheet(f"color: {theme.current()['qt_text']}; padding: 0 6px;")
        self.status.addWidget(self._status_lbl, 1)

        self.lbl_channels      = QtWidgets.QLabel("Channels: —")
        self.lbl_spatial       = QtWidgets.QLabel("Spatial sampling: —")
        self.lbl_duration      = QtWidgets.QLabel("Duration: —")
        self.lbl_sampling_rate = QtWidgets.QLabel("Sampling rate: —")
        self.lbl_units         = QtWidgets.QLabel("Units: —")

        for lbl in [self.lbl_channels, self.lbl_spatial,
                    self.lbl_duration, self.lbl_sampling_rate, self.lbl_units]:
            lbl.setStyleSheet(f"color: {theme.current()['qt_text_dim']}; padding: 0 8px;")
            self.status.addPermanentWidget(lbl)

    # Coloured status helpers
    def _status_processing(self, msg: str) -> None:
        """Show an orange message (processing in progress)."""
        self._status_lbl.setStyleSheet("color: #e0a020; padding: 0 6px;")
        self._status_lbl.setText(msg)
        QtWidgets.QApplication.processEvents()

    def _status_done(self, msg: str, timeout_ms: int = 5000) -> None:
        """Show a neutral-text message (operation complete)."""
        self._status_lbl.setStyleSheet(f"color: {theme.current()['qt_text']}; padding: 0 6px;")
        self._status_lbl.setText(msg)
        if timeout_ms > 0:
            QtCore.QTimer.singleShot(timeout_ms, lambda: self._status_lbl.setText(""))

    def _status_error(self, msg: str, timeout_ms: int = 6000) -> None:
        """Show a red message (error)."""
        self._status_lbl.setStyleSheet("color: #ff4444; padding: 0 6px;")
        self._status_lbl.setText(msg)
        if timeout_ms > 0:
            QtCore.QTimer.singleShot(timeout_ms, lambda: self._status_lbl.setText(""))

    def _status_clear(self) -> None:
        self._status_lbl.setText("")

    # ------------------------------------------------------------------
    # Data group
    # ------------------------------------------------------------------

    def _build_data_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Data")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        # --- Current file label (top) ---
        self.lbl_file = QtWidgets.QLabel("<i>No file loaded</i>")
        self.lbl_file.setStyleSheet(f"color: {theme.current()['qt_text_dim']};")
        self.lbl_file.setWordWrap(True)
        layout.addWidget(self.lbl_file)

        # --- Select file button ---
        self.btn_load = QtWidgets.QPushButton("Select file...")
        self.btn_load.clicked.connect(self.load_file_dialog)
        layout.addWidget(self.btn_load)

        # --- Profile + Stride row ---
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(6)

        ctrl_row.addWidget(QtWidgets.QLabel("Profile:"))
        self.combo_interrogator = QtWidgets.QComboBox()
        from dasexplorer.core.config import get_all_profiles as _gap_ui
        for _p in _gap_ui().values():
            self.combo_interrogator.addItem(_p.get("label", "?"))
        self.combo_interrogator.currentIndexChanged.connect(self._on_interrogator_changed)
        ctrl_row.addWidget(self.combo_interrogator, 1)

        ctrl_row.addWidget(QtWidgets.QLabel("Stride:"))
        self.combo_stride = QtWidgets.QComboBox()
        for v in STRIDE_VALUES:
            self.combo_stride.addItem(str(v))
        self.combo_stride.setFixedWidth(78)
        ctrl_row.addWidget(self.combo_stride)

        layout.addLayout(ctrl_row)

        # --- File list (expands to fill available space) ---
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.file_list_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.file_list_widget.setMinimumHeight(120)
        self.file_list_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.file_list_widget.itemDoubleClicked.connect(self._on_file_double_clicked)
        layout.addWidget(self.file_list_widget, 1)

        # --- Previous / Next buttons ---
        nav_row = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("◀  Previous")
        self.btn_next = QtWidgets.QPushButton("Next  ▶")
        self.btn_prev.clicked.connect(self._load_previous)
        self.btn_next.clicked.connect(self._load_next)
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        layout.addLayout(nav_row)

        return group

    # ------------------------------------------------------------------
    # View group
    # ------------------------------------------------------------------

    def _build_view_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("View")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        layout.addLayout(self._build_range_row("Time [s]",       "spin_tmin", "spin_tmax", 0.0,   60.0,  0.0,  1e6,  1.0))
        layout.addLayout(self._build_range_row("Distance [m]",   "spin_dmin", "spin_dmax", 0.0, 50000.0,  0.0,  1e9, 100.0))
        layout.addLayout(self._build_range_row("Color",          "spin_vmin", "spin_vmax", 0.0,    12.0, -1e9,  1e9,  0.1))

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setStyleSheet(f"color: {theme.current()['qt_border']};")
        layout.addWidget(sep)

        layout.addLayout(self._build_range_row("Frequency [Hz]", "spin_fmin", "spin_fmax", 1.0,    50.0,  0.0,  1e4,  0.5))

        # Envelope (Hilbert) checkbox
        self.btn_envelope = QtWidgets.QCheckBox("Hilbert Envelope (Raw)")
        self.btn_envelope.setChecked(False)
        self.btn_envelope.setToolTip(
            "Apply Hilbert transform to show signal envelope (amplitude).\n"
            "Replicates das4whales tutorial plot style."
        )
        self.btn_envelope.toggled.connect(self._on_envelope_toggled)
        layout.addWidget(self.btn_envelope)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.HLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep2.setStyleSheet(f"color: {theme.current()['qt_border']};")
        layout.addWidget(sep2)

        # FK filter parameters — very permissive defaults so first use shows data
        layout.addLayout(self._build_range_row("F-K c [m/s]", "spin_cmin", "spin_cmax",  100.0, 100000.0, 0.0, 1e7, 100.0))
        layout.addLayout(self._build_range_row("F-K f [Hz]",  "spin_fkmin","spin_fkmax",   1.0,    249.0, 0.0, 1e4,   0.5))

        # FK envelope checkbox
        self.chk_fk_envelope = QtWidgets.QCheckBox("Hilbert Envelope (F-K)")
        self.chk_fk_envelope.setChecked(False)
        self.chk_fk_envelope.setToolTip(
            "Apply Hilbert transform to the F-K filtered signal.\n"
            "Independent from the Raw envelope toggle."
        )
        self.chk_fk_envelope.toggled.connect(self._on_fk_envelope_toggled)
        layout.addWidget(self.chk_fk_envelope)

        sep3 = QtWidgets.QFrame()
        sep3.setFrameShape(QtWidgets.QFrame.HLine)
        sep3.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep3.setStyleSheet(f"color: {theme.current()['qt_border']};")
        layout.addWidget(sep3)

        # RGB multispectral band parameters
        layout.addLayout(self._build_range_row("R [Hz]", "spin_rmin", "spin_rmax",  1.0,  5.0, 0.0, 1e4, 0.5))
        layout.addLayout(self._build_range_row("G [Hz]", "spin_gmin", "spin_gmax",  5.0, 15.0, 0.0, 1e4, 0.5))
        layout.addLayout(self._build_range_row("B [Hz]", "spin_bmin", "spin_bmax", 15.0, 40.0, 0.0, 1e4, 0.5))

        pct_row = QtWidgets.QHBoxLayout()
        lbl_pct = QtWidgets.QLabel("RGB percentile:")
        pct_row.addWidget(lbl_pct)
        self.spin_rgb_pct = QtWidgets.QDoubleSpinBox()
        self.spin_rgb_pct.setRange(1.0, 100.0)
        self.spin_rgb_pct.setDecimals(1)
        self.spin_rgb_pct.setValue(90.0)
        self.spin_rgb_pct.setFixedWidth(78)
        pct_row.addWidget(self.spin_rgb_pct)
        pct_row.addStretch()
        self._btn_apply_rgb_pct = QtWidgets.QPushButton("Apply")
        self._btn_apply_rgb_pct.setFixedWidth(90)
        pct_row.addWidget(self._btn_apply_rgb_pct)
        layout.addLayout(pct_row)

        self._btn_apply_time.clicked.connect(self._apply_time)
        self._btn_apply_dist.clicked.connect(self._apply_distance)
        self._btn_apply_color.clicked.connect(self._apply_color)
        self._btn_apply_freq.clicked.connect(self._apply_frequency)
        self._btn_apply_cmin.clicked.connect(self._apply_fk)
        self._btn_apply_fkmin.clicked.connect(self._apply_fk)
        self._btn_apply_rmin.clicked.connect(self._apply_rgb)
        self._btn_apply_gmin.clicked.connect(self._apply_rgb)
        self._btn_apply_bmin.clicked.connect(self._apply_rgb)
        self._btn_apply_rgb_pct.clicked.connect(self._apply_rgb)

        return group

    def _build_range_row(
        self,
        label: str,
        attr_min: str,
        attr_max: str,
        init_min: float,
        init_max: float,
        range_min: float,
        range_max: float,
        step: float,
    ) -> QtWidgets.QVBoxLayout:
        outer = QtWidgets.QVBoxLayout()
        outer.setSpacing(2)
        outer.addWidget(QtWidgets.QLabel(label))

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.setContentsMargins(0, 0, 0, 0)

        spin_min = self._make_spinbox(init_min, range_min, range_max, step)
        spin_max = self._make_spinbox(init_max, range_min, range_max, step)
        setattr(self, attr_min, spin_min)
        setattr(self, attr_max, spin_max)

        btn = QtWidgets.QPushButton("Apply")
        btn.setFixedWidth(90)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

        btn_attr = {
            "spin_tmin":  "_btn_apply_time",
            "spin_dmin":  "_btn_apply_dist",
            "spin_vmin":  "_btn_apply_color",
            "spin_fmin":  "_btn_apply_freq",
            "spin_cmin":  "_btn_apply_cmin",
            "spin_fkmin": "_btn_apply_fkmin",
            "spin_rmin":  "_btn_apply_rmin",
            "spin_gmin":  "_btn_apply_gmin",
            "spin_bmin":  "_btn_apply_bmin",
        }.get(attr_min)
        if btn_attr:
            setattr(self, btn_attr, btn)

        row.addWidget(QtWidgets.QLabel("Min"), 0)
        row.addWidget(spin_min, 1)
        row.addWidget(QtWidgets.QLabel("Max"), 0)
        row.addWidget(spin_max, 1)
        row.addWidget(btn, 0)

        outer.addLayout(row)
        return outer

    def _make_spinbox(self, value, min_val, max_val, step):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setDecimals(3)
        sb.setSingleStep(step)
        sb.setMinimumWidth(65)
        sb.setValue(value)
        return sb

    def _build_annotations_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Annotations")
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding,
        )
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(6, 6, 6, 6)

        self.ann_widget = AnnotationWidget()
        self.ann_widget.annotation_mode_changed.connect(self._on_annotation_mode_changed)
        self.ann_widget.annotation_selected.connect(self._on_annotation_selected)
        self.ann_widget.delete_requested.connect(self._on_annotation_delete)
        self.ann_widget.save_requested.connect(self._on_annotation_save)
        self.ann_widget.clear_requested.connect(self._on_annotation_clear)
        self.ann_widget.id_changed.connect(self._on_annotation_id_changed)
        self.ann_widget.export_path_changed.connect(self._on_export_path_changed)
        self.ann_widget.csv_file_selected.connect(self._on_csv_file_selected)

        layout.addWidget(self.ann_widget, 1)  # stretch=1: table fills remaining space
        return group

    # ------------------------------------------------------------------
    # Window events
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        total = self.splitter.width()
        if total > 0:
            self.splitter.setSizes([total // 5, total * 4 // 5])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_file_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select DAS file")
        if not path:
            return
        self._load_path(path)

    def load_synthetic_data(self) -> None:
        self.lbl_file.setText("Loading file…")
        self.lbl_file.setStyleSheet("color: #e0a020;")
        # "Loading file…" is already shown above (lbl_file); no need to
        # duplicate it in the bottom status bar.
        QtWidgets.QApplication.processEvents()
        self.dataset = None  # treat as first load so spinboxes reset
        dataset = generate_synthetic_dataset()
        self._set_dataset("synthetic data", dataset)
        self._current_dir = None
        self._file_list   = []
        self.file_list_widget.clear()

    def _load_path(self, path: str) -> None:
        # Ask to save if there are unsaved annotations
        if self._ann_model.dirty and len(self._ann_model) > 0:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved annotations",
                f"There are unsaved annotations for the current file.\n"
                f"Save before loading the next file?",
                QtWidgets.QMessageBox.Save |
                QtWidgets.QMessageBox.Discard |
                QtWidgets.QMessageBox.Cancel,
            )
            if reply == QtWidgets.QMessageBox.Cancel:
                return
            if reply == QtWidgets.QMessageBox.Save:
                self._on_annotation_save()

        from dasexplorer.core.config import get_all_profiles as _gap4, get_profile as _gp
        _pkeys4      = list(_gap4().keys())
        idx          = self.combo_interrogator.currentIndex()
        _profile_key = _pkeys4[idx] if idx < len(_pkeys4) else _pkeys4[0]
        _profile_cfg = _gp(_profile_key)
        interrogator = _profile_cfg.get("interrogator", INTERROGATOR_TYPES[0])
        stride       = STRIDE_VALUES[self.combo_stride.currentIndex()]

        self.lbl_file.setText("Loading file…")
        self.lbl_file.setStyleSheet("color: #e0a020;")
        # "Loading file…" is already shown above (lbl_file); no need to
        # duplicate it in the bottom status bar.
        QtWidgets.QApplication.processEvents()

        kwargs = {"stride": stride} if stride > 1 else {}
        if interrogator == "hdas2.5":
            kwargs["num_files"] = int(_profile_cfg.get("num_files", 1))
        try:
            dataset = read_das_file(path, interrogator, **kwargs)
        except NotImplementedError as exc:
            QtWidgets.QMessageBox.information(self, "Not implemented yet", str(exc))
            self.lbl_file.setText("<i>No file loaded</i>")
            self.lbl_file.setStyleSheet(f"color: {theme.current()['qt_text_dim']};")
            return
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error loading file", str(exc))
            self.lbl_file.setText("<i>No file loaded</i>")
            self.lbl_file.setStyleSheet(f"color: {theme.current()['qt_text_dim']};")
            return

        self._current_data_path = path
        directory = os.path.dirname(path)
        filename  = os.path.basename(path)

        if directory != self._current_dir:
            self._current_dir = directory
            self._refresh_file_list(_profile_key)

        self._highlight_file(filename)

        # Capture current viz params before any reset so they can be
        # restored after _apply_default_view_params on subsequent loads.
        if self.dataset is not None:
            from dasexplorer.gui.waterfall import COLORMAPS
            self._preserved_viz = {
                "vmin":         self.spin_vmin.value(),
                "vmax":         self.spin_vmax.value(),
                "fmin":         self.spin_fmin.value(),
                "fmax":         self.spin_fmax.value(),
                "envelope":     self._envelope,
                "cmin":         self.spin_cmin.value(),
                "cmax":         self.spin_cmax.value(),
                "fkmin":        self.spin_fkmin.value(),
                "fkmax":        self.spin_fkmax.value(),
                "envelope_fk":  self._envelope_fk,
                "rmin":         self.spin_rmin.value(),
                "rmax":         self.spin_rmax.value(),
                "gmin":         self.spin_gmin.value(),
                "gmax":         self.spin_gmax.value(),
                "bmin":         self.spin_bmin.value(),
                "bmax":         self.spin_bmax.value(),
                "rgb_pct":      self.spin_rgb_pct.value(),
                "cmap_idx":     (self.waterfall.combo_cmap.currentIndex()
                                 if hasattr(self.waterfall, "combo_cmap") else 0),
            }
        else:
            self._preserved_viz = None

        self.dataset = None  # keeps is_first_load logic in _set_dataset intact
        self._set_dataset(filename, dataset)

    def _refresh_file_list(self, profile_key_or_interrogator: str) -> None:
        """Refresh the file list for the current directory.

        Accepts either a profile key (new API) or a bare interrogator type
        (legacy fallback). Extensions are resolved from the profile config.
        """
        from dasexplorer.core.config import get_all_profiles, get_profile as _gp_rf
        profiles = get_all_profiles()
        if profile_key_or_interrogator in profiles:
            pcfg = _gp_rf(profile_key_or_interrogator)
            interrogator = pcfg.get("interrogator", "")
            exts = pcfg.get("file_extensions") or _FALLBACK_EXTENSIONS.get(interrogator, [])
        else:
            # Legacy: bare interrogator type passed
            interrogator = profile_key_or_interrogator
            exts = _FALLBACK_EXTENSIONS.get(interrogator, [])
        if self._current_dir and os.path.isdir(self._current_dir):
            self._file_list = sorted([
                f for f in os.listdir(self._current_dir)
                if os.path.splitext(f)[1].lower() in exts
            ])
        else:
            self._file_list = []

        self.file_list_widget.clear()
        for fname in self._file_list:
            self.file_list_widget.addItem(fname)

    def _highlight_file(self, filename: str) -> None:
        for i in range(self.file_list_widget.count()):
            if self.file_list_widget.item(i).text() == filename:
                self.file_list_widget.setCurrentRow(i)
                self.file_list_widget.scrollToItem(
                    self.file_list_widget.item(i),
                    QtWidgets.QAbstractItemView.PositionAtCenter,
                )
                break

    def _apply_default_view_params(self, dataset, cfg: dict) -> None:
        """
        Reset all View-panel spinboxes/checkboxes to the interrogator's
        config.json defaults. Shared by _set_dataset (first load) and
        File > Refresh. Every field visible in the View panel can be
        configured via config.json — see the file itself for documentation.
        """
        from dasexplorer.gui.waterfall import COLORMAPS

        tmin_cfg        = cfg.get("tmin_s", None)
        tmax_cfg        = cfg.get("tmax_s", None)
        dmin_cfg        = cfg.get("dmin_m", None)
        dmax_cfg        = cfg.get("dmax_m", None)
        vmin_cfg        = float(cfg.get("vmin", 0))
        vmax_cfg        = float(cfg.get("vmax", 12))
        cmap_cfg        = str(cfg.get("colormap", "Rainbow"))
        fmin_cfg        = float(cfg.get("fmin_hz", 1.0))
        fmax_off        = float(cfg.get("fmax_offset_hz", 0.01))
        fmax_hz_cfg     = cfg.get("fmax_hz", None)
        stride_cfg      = int(cfg.get("stride", 1))
        envelope_cfg    = bool(cfg.get("envelope", False))
        fk_cmin_cfg     = float(cfg.get("fk_cmin_ms", 100.0))
        fk_cmax_cfg     = float(cfg.get("fk_cmax_ms", 100000.0))
        fk_fmin_cfg     = float(cfg.get("fk_fmin_hz", 1.0))
        fk_fmax_off     = float(cfg.get("fk_fmax_offset_hz", 0.01))
        fk_fmax_hz_cfg  = cfg.get("fk_fmax_hz", None)
        fk_envelope_cfg = bool(cfg.get("fk_envelope", envelope_cfg))
        rgb_rmin_cfg    = float(cfg.get("rgb_rmin_hz", 1.0))
        rgb_rmax_cfg    = float(cfg.get("rgb_rmax_hz", 5.0))
        rgb_gmin_cfg    = float(cfg.get("rgb_gmin_hz", 5.0))
        rgb_gmax_cfg    = float(cfg.get("rgb_gmax_hz", 15.0))
        rgb_bmin_cfg    = float(cfg.get("rgb_bmin_hz", 15.0))
        rgb_bmax_cfg    = float(cfg.get("rgb_bmax_hz", 40.0))
        rgb_pct_cfg     = float(cfg.get("rgb_percentile", 90.0))
        nyq = dataset.fs_hz / 2.0

        # Time range: null means use the full extent of the loaded file.
        # Values are clamped to the actual data range so a user-set value
        # beyond the file length never crashes.
        t0 = float(tmin_cfg) if tmin_cfg is not None else float(dataset.time_s[0])
        t1 = float(tmax_cfg) if tmax_cfg is not None else float(dataset.time_s[-1])
        t0 = max(t0, float(dataset.time_s[0]))
        t1 = min(t1, float(dataset.time_s[-1]))

        # Distance range: same logic.
        d0 = float(dmin_cfg) if dmin_cfg is not None else float(dataset.dist_m[0])
        d1 = float(dmax_cfg) if dmax_cfg is not None else float(dataset.dist_m[-1])
        d0 = max(d0, float(dataset.dist_m[0]))
        d1 = min(d1, float(dataset.dist_m[-1]))

        # Block all affected spinboxes while setting values
        spinboxes = [self.spin_tmin, self.spin_tmax, self.spin_dmin, self.spin_dmax,
                     self.spin_vmin, self.spin_vmax, self.spin_fmin, self.spin_fmax]
        for sb in spinboxes:
            sb.blockSignals(True)

        self.spin_tmin.setValue(t0)
        self.spin_tmax.setValue(t1)
        self.spin_dmin.setValue(d0)
        self.spin_dmax.setValue(d1)
        self.spin_vmin.setValue(vmin_cfg)
        self.spin_vmax.setValue(vmax_cfg)
        self.spin_fmin.setValue(fmin_cfg)
        self.spin_fmax.setMaximum(nyq)
        # fmax_hz takes priority over fmax_offset_hz when explicitly set and
        # within the valid range; otherwise fall back to Nyquist - offset.
        fmax_val = float(fmax_hz_cfg) if fmax_hz_cfg is not None else round(nyq - fmax_off, 3)
        fmax_val = min(fmax_val, round(nyq - 0.001, 3))
        self.spin_fmax.setValue(fmax_val)
        self.spin_fkmax.setMaximum(nyq)
        fk_fmax_val = float(fk_fmax_hz_cfg) if fk_fmax_hz_cfg is not None else round(nyq - fk_fmax_off, 3)
        fk_fmax_val = min(fk_fmax_val, round(nyq - 0.001, 3))
        self.spin_fkmax.setValue(fk_fmax_val)
        self.spin_cmin.setValue(fk_cmin_cfg)
        self.spin_cmax.setValue(fk_cmax_cfg)
        self.spin_fkmin.setValue(fk_fmin_cfg)
        self.spin_rmin.setValue(rgb_rmin_cfg)
        self.spin_rmax.setValue(rgb_rmax_cfg)
        self.spin_gmin.setValue(rgb_gmin_cfg)
        self.spin_gmax.setValue(rgb_gmax_cfg)
        self.spin_bmin.setValue(rgb_bmin_cfg)
        self.spin_bmax.setValue(rgb_bmax_cfg)
        self.spin_rgb_pct.setValue(rgb_pct_cfg)

        for sb in spinboxes:
            sb.blockSignals(False)

        # Stride — match the configured value to the nearest available option
        stride_val = stride_cfg if stride_cfg in STRIDE_VALUES else 1
        self.combo_stride.blockSignals(True)
        self.combo_stride.setCurrentIndex(STRIDE_VALUES.index(stride_val))
        self.combo_stride.blockSignals(False)

        # Colormap — apply to all three waterfall widgets simultaneously
        cmap_names = [c[0] for c in COLORMAPS]
        if cmap_cfg in cmap_names:
            cmap_idx = cmap_names.index(cmap_cfg)
            for wf in self._waterfalls():
                if hasattr(wf, 'combo_cmap'):
                    wf.combo_cmap.blockSignals(True)
                    wf.combo_cmap.setCurrentIndex(cmap_idx)
                    wf.combo_cmap.blockSignals(False)
                    wf._apply_colormap(COLORMAPS[cmap_idx][1])

        # Raw envelope
        self._envelope = envelope_cfg
        self.btn_envelope.blockSignals(True)
        self.btn_envelope.setChecked(envelope_cfg)
        self.btn_envelope.blockSignals(False)

        # F-K envelope (independent from Raw — may differ in config)
        self._envelope_fk = fk_envelope_cfg
        self.chk_fk_envelope.blockSignals(True)
        self.chk_fk_envelope.setChecked(fk_envelope_cfg)
        self.chk_fk_envelope.blockSignals(False)

    def _set_dataset(self, label: str, dataset) -> None:
        is_first_load = self.dataset is None
        self._tr_fk_base = None   # invalidate cached FK array on new file
        self._rgb_array = None    # invalidate cached RGB composite on new file

        self.dataset = dataset
        self.lbl_file.setText(label)
        self.lbl_file.setStyleSheet("color: #4a9eff;")

        # Per-interrogator defaults from config
        interrogator = INTERROGATOR_TYPES[self.combo_interrogator.currentIndex()]
        cfg      = get_interrogator_defaults(interrogator)
        vmin_cfg = float(cfg.get("vmin", 0))
        vmax_cfg = float(cfg.get("vmax", 12))
        fmin_cfg = float(cfg.get("fmin_hz", 1.0))
        fmax_off = float(cfg.get("fmax_offset_hz", 0.01))
        fmax_hz_cfg    = cfg.get("fmax_hz", None)
        envelope_cfg   = bool(cfg.get("envelope", False))
        fk_cmin_cfg    = float(cfg.get("fk_cmin_ms", 100.0))
        fk_cmax_cfg    = float(cfg.get("fk_cmax_ms", 100000.0))
        fk_fmin_cfg    = float(cfg.get("fk_fmin_hz", 1.0))
        fk_fmax_off    = float(cfg.get("fk_fmax_offset_hz", 0.01))
        fk_fmax_hz_cfg = cfg.get("fk_fmax_hz", None)
        self._default_view = str(cfg.get("default_view", "raw")).lower()
        rgb_rmin_cfg = float(cfg.get("rgb_rmin_hz", 1.0))
        rgb_rmax_cfg = float(cfg.get("rgb_rmax_hz", 5.0))
        rgb_gmin_cfg = float(cfg.get("rgb_gmin_hz", 5.0))
        rgb_gmax_cfg = float(cfg.get("rgb_gmax_hz", 15.0))
        rgb_bmin_cfg = float(cfg.get("rgb_bmin_hz", 15.0))
        rgb_bmax_cfg = float(cfg.get("rgb_bmax_hz", 40.0))
        rgb_pct_cfg  = float(cfg.get("rgb_percentile", 90.0))
        nyq      = dataset.fs_hz / 2.0

        # _apply_default_view_params always runs (is_first_load is always True
        # here because _load_path sets self.dataset=None before calling us).
        # Preserved params are restored below when _preserved_viz is set.
        if is_first_load:
            self._apply_default_view_params(dataset, cfg)

        # Restore viz params from previous file in the session.
        # Time/distance axes are intentionally NOT restored — they are reset
        # to the new file extents by _apply_default_view_params above.
        p = getattr(self, "_preserved_viz", None)
        if p is not None:
            from dasexplorer.gui.waterfall import COLORMAPS
            # Block all affected controls
            sbs = [self.spin_vmin, self.spin_vmax, self.spin_fmin, self.spin_fmax,
                   self.spin_cmin, self.spin_cmax, self.spin_fkmin, self.spin_fkmax,
                   self.spin_rmin, self.spin_rmax, self.spin_gmin, self.spin_gmax,
                   self.spin_bmin, self.spin_bmax, self.spin_rgb_pct]
            for sb in sbs:
                sb.blockSignals(True)

            self.spin_vmin.setValue(p["vmin"])
            self.spin_vmax.setValue(p["vmax"])
            self.spin_fmin.setValue(p["fmin"])

            # Restore fmax clamped to new Nyquist
            self.spin_fmax.setMaximum(nyq)
            self.spin_fmax.setValue(min(p["fmax"], round(nyq - 0.001, 3)))

            self.spin_cmin.setValue(p["cmin"])
            self.spin_cmax.setValue(p["cmax"])
            self.spin_fkmin.setValue(p["fkmin"])

            # Restore fkmax clamped to new Nyquist
            self.spin_fkmax.setMaximum(nyq)
            self.spin_fkmax.setValue(min(p["fkmax"], round(nyq - 0.001, 3)))

            self.spin_rmin.setValue(p["rmin"])
            self.spin_rmax.setValue(p["rmax"])
            self.spin_gmin.setValue(p["gmin"])
            self.spin_gmax.setValue(p["gmax"])
            self.spin_bmin.setValue(p["bmin"])
            self.spin_bmax.setValue(p["bmax"])
            self.spin_rgb_pct.setValue(p["rgb_pct"])

            for sb in sbs:
                sb.blockSignals(False)

            # Envelope checkboxes
            self._envelope = p["envelope"]
            self.btn_envelope.blockSignals(True)
            self.btn_envelope.setChecked(p["envelope"])
            self.btn_envelope.blockSignals(False)

            self._envelope_fk = p["envelope_fk"]
            self.chk_fk_envelope.blockSignals(True)
            self.chk_fk_envelope.setChecked(p["envelope_fk"])
            self.chk_fk_envelope.blockSignals(False)

            # Colormap — restore on all waterfall widgets
            cmap_idx = p["cmap_idx"]
            if 0 <= cmap_idx < len(COLORMAPS):
                for wf in self._waterfalls():
                    if hasattr(wf, "combo_cmap"):
                        wf.combo_cmap.blockSignals(True)
                        wf.combo_cmap.setCurrentIndex(cmap_idx)
                        wf.combo_cmap.blockSignals(False)
                        wf._apply_colormap(COLORMAPS[cmap_idx][1])

        # Update status bar info
        dx = float(dataset.dist_m[1] - dataset.dist_m[0]) if dataset.n_dist > 1 else 0.0
        self.lbl_channels.setText(f"Channels: {dataset.n_dist}")
        self.lbl_spatial.setText(f"Spatial sampling: {dx:.1f} m")
        self.lbl_duration.setText(f"Duration: {dataset.time_s[-1]:.2f} s")
        self.lbl_sampling_rate.setText(f"Sampling rate: {dataset.fs_hz:.1f} Hz")
        self.lbl_units.setText(f"Units: {dataset.units or '—'}")

        fmin = self.spin_fmin.value()
        fmax = self.spin_fmax.value()
        vmin = self.spin_vmin.value()
        vmax = self.spin_vmax.value()
        can_filter = fmin > 0 and fmax > fmin and fmax < nyq

        # Show overlay (repaint only, no processEvents)
        if can_filter:
            self.waterfall.show_overlay(
                "Loading file…",
                f"Band-pass filtering  [{fmin:.3f} – {fmax:.3f} Hz]",
            )

        # Single atomic call: filter + envelope + render + axes
        self.waterfall.load_and_display(
            dataset,
            vmin=vmin,
            vmax=vmax,
            fmin=fmin if can_filter else None,
            fmax=fmax if can_filter else None,
            envelope=self._envelope,
        )

        self.waterfall.hide_overlay()

        # On subsequent loads re-apply color levels so the histogram reflects
        # the preserved vmin/vmax rather than the config.json defaults.
        if getattr(self, "_preserved_viz", None) is not None:
            self.waterfall.apply_color_levels(vmin, vmax)

        # Only auto-compute F-K if it's configured as the default view —
        # otherwise compute lazily when the user first visits the F-K tab.
        if self._default_view == "fk":
            self._apply_fk()
            self.tab_widget.setCurrentIndex(1)
        else:
            self.tab_widget.setCurrentIndex(0)

        # --- Annotations ---
        # Set export dir to data folder if not already set
        if not self._export_dir and self._current_dir:
            self._export_dir = self._current_dir
            self.ann_widget.set_export_dir(self._export_dir)

        # Load any existing annotations for this file
        self._load_annotations_for_current_file(label)

        # Refresh CSV list in annotation panel
        self.ann_widget.refresh_csv_list(self._export_dir)

    # ------------------------------------------------------------------
    # File list interactions
    # ------------------------------------------------------------------

    def _on_file_double_clicked(self, item) -> None:
        if self._current_dir is None:
            return
        path = os.path.join(self._current_dir, item.text())
        self._load_path(path)

    def _on_interrogator_changed(self, index: int) -> None:
        if self._current_dir:
            from dasexplorer.core.config import get_all_profiles as _gap2
            _pk2 = list(_gap2().keys())
            if index < len(_pk2):
                self._refresh_file_list(_pk2[index])

    def _load_previous(self) -> None:
        self._load_adjacent(-1)

    def _load_next(self) -> None:
        self._load_adjacent(+1)

    def _load_adjacent(self, delta: int) -> None:
        if not self._file_list or self._current_dir is None:
            return
        current = self.file_list_widget.currentRow()
        if current < 0:
            return
        target = current + delta
        if 0 <= target < len(self._file_list):
            path = os.path.join(self._current_dir, self._file_list[target])
            self._load_path(path)

    # ------------------------------------------------------------------
    # View Apply callbacks
    # ------------------------------------------------------------------

    def _on_envelope_toggled(self, checked: bool) -> None:
        """Apply or remove Hilbert envelope on Raw waterfall only."""
        self._envelope = checked
        if self.dataset is None:
            return

        if checked:
            self._status_processing("Computing Hilbert envelope…")
        else:
            self._status_processing("Restoring filtered signal…")

        xr = self.waterfall.plot_widget.getPlotItem().vb.viewRange()[0]
        yr = self.waterfall.plot_widget.getPlotItem().vb.viewRange()[1]

        fmin = self.spin_fmin.value()
        fmax = self.spin_fmax.value()
        nyq  = self.dataset.fs_hz / 2.0
        can_filter = fmin > 0 and fmax > fmin and fmax < nyq
        vmin = self.spin_vmin.value()
        vmax = self.spin_vmax.value()

        self.waterfall.load_and_display(
            self.dataset, vmin=vmin, vmax=vmax,
            fmin=fmin if can_filter else None,
            fmax=fmax if can_filter else None,
            envelope=self._envelope,
        )
        self.waterfall.apply_time_range(xr[0], xr[1])
        self.waterfall.apply_distance_range(yr[0], yr[1])

        if checked:
            self._status_done("Hilbert envelope applied")
        else:
            self._status_done("Signal restored (no envelope)")

    def _on_fk_envelope_toggled(self, checked: bool) -> None:
        """Apply or remove Hilbert envelope on F-K waterfall only."""
        self._envelope_fk = checked
        if self._tr_fk_base is None:
            return

        if checked:
            self._status_processing("Computing F-K Hilbert envelope…")
        else:
            self._status_processing("Restoring F-K filtered signal…")

        self._render_fk()

        if checked:
            self._status_done("F-K Hilbert envelope applied")
        else:
            self._status_done("F-K signal restored (no envelope)")

    def _update_envelope_button_style(self, active: bool) -> None:
        pass  # QCheckBox styles itself

    def _apply_time(self) -> None:
        t0, t1 = self.spin_tmin.value(), self.spin_tmax.value()
        if t0 >= t1:
            self._status_error("Time: Tmin must be less than Tmax.")
            return
        for wf in self._waterfalls():
            if wf.dataset is not None:
                wf.apply_time_range(t0, t1)

    def _apply_distance(self) -> None:
        d0, d1 = self.spin_dmin.value(), self.spin_dmax.value()
        if d0 >= d1:
            self._status_error("Distance: Dmin must be less than Dmax.")
            return
        for wf in self._waterfalls():
            if wf.dataset is not None:
                wf.apply_distance_range(d0, d1)

    def _apply_color(self) -> None:
        vmin, vmax = self.spin_vmin.value(), self.spin_vmax.value()
        if vmin >= vmax:
            self._status_error("Color: Vmin must be less than Vmax.")
            return
        self.waterfall.apply_color_levels(vmin, vmax)
        if self.waterfall_fk.get_displayed_array() is not None:
            self.waterfall_fk.apply_color_levels(vmin, vmax)

    def _apply_frequency(self) -> None:
        if self.dataset is None:
            self._status_error("Load a file before applying a filter.")
            return

        fmin = self.spin_fmin.value()
        fmax = self.spin_fmax.value()
        nyq  = self.dataset.fs_hz / 2.0

        if fmin <= 0 or fmax <= fmin:
            self._status_error("Frequency: Fmin must be > 0 and < Fmax.")
            return
        if fmax >= nyq:
            self._status_error(f"Fmax must be < Nyquist ({nyq:.1f} Hz).")
            return

        self._status_processing(f"Applying bandpass [{fmin:.3f} – {fmax:.3f} Hz]…")
        QtWidgets.QApplication.processEvents()
        self.waterfall.apply_bandpass(fmin, fmax, envelope=self._envelope)
        # Invalidate F-K cache (bandpass result changed) WITHOUT recomputing.
        # F-K is only (re)computed lazily when the user visits the F-K tab.
        self._tr_fk_base = None
        self._status_done(f"Bandpass applied: {fmin:.3f} – {fmax:.3f} Hz")

    # ------------------------------------------------------------------
    # Analysis dialogs
    # ------------------------------------------------------------------

    def _get_analysis_ann(self, index: int):
        """
        Return a BBoxAnnotation-compatible object for the annotation at the
        given flat table index (across all four type models).
        For OBB, returns a synthetic BBoxAnnotation built from bbox_ti_di().
        For KP and LINE, builds the bbox from the point cloud extent.
        Returns None if the index is invalid or no dataset is loaded.
        """
        if self.dataset is None or index < 0:
            self._status_error("No valid annotation selected.")
            return None
        ds = self.dataset
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > index:
                ann = model[index - row]
                if ann_type == AnnType.BBOX:
                    return ann
                # For non-BBox types, create a synthetic BBoxAnnotation
                # from the axis-aligned bounding box of the annotation
                import json as _json
                if ann_type == AnnType.OBB:
                    ti0, ti1, di0, di1 = ann.bbox_ti_di()
                    t0 = float(ds.time_s[max(0, ti0)])
                    t1 = float(ds.time_s[min(len(ds.time_s)-1, ti1)])
                    d0 = float(ds.dist_m[max(0, di0)])
                    d1 = float(ds.dist_m[min(len(ds.dist_m)-1, di1)])
                elif ann_type == AnnType.KP:
                    tis = _json.loads(ann.kp_ti)
                    dis = _json.loads(ann.kp_di)
                    ti0, ti1 = min(tis), max(tis)
                    di0, di1 = min(dis), max(dis)
                    t0 = float(ds.time_s[max(0, ti0)])
                    t1 = float(ds.time_s[min(len(ds.time_s)-1, ti1)])
                    d0 = float(ds.dist_m[max(0, di0)])
                    d1 = float(ds.dist_m[min(len(ds.dist_m)-1, di1)])
                else:  # LINE
                    tis = _json.loads(ann.pts_ti)
                    dis = _json.loads(ann.pts_di)
                    ti0, ti1 = min(tis), max(tis)
                    di0, di1 = min(dis), max(dis)
                    t0 = float(ds.time_s[max(0, ti0)])
                    t1 = float(ds.time_s[min(len(ds.time_s)-1, ti1)])
                    d0 = float(ds.dist_m[max(0, di0)])
                    d1 = float(ds.dist_m[min(len(ds.dist_m)-1, di1)])
                start_dt = ds.start_datetime_utc.isoformat() if ds.start_datetime_utc else ""
                return BBoxAnnotation(
                    ann_type="bbox", id=ann.id, comment=ann.comment,
                    t0=t0, t1=t1, d0=d0, d1=d1,
                    ti0=ti0, ti1=ti1, di0=di0, di1=di1,
                    nt=ds.n_time, nx=ds.n_dist,
                    downsample=ds.downsample or 1,
                    start_datetime_utc=start_dt,
                )
            row += len(model)
        self._status_error("No valid annotation selected.")
        return None

    def _current_colormap(self) -> pg.ColorMap:
        """Return the colormap currently active in the waterfall histogram."""
        return self.waterfall.histogram.gradient.colorMap()

    def _analysis_dataset(self):
        """
        Return a DASDataset whose .tr is the array currently shown in the
        Raw waterfall (bandpass-filtered, WITHOUT Hilbert envelope) — this
        is what Spectrogram / Spectral Analysis / Signal tools must use,
        regardless of whether the Envelope checkbox is active.
        Falls back to self.dataset.tr (raw) if nothing has been filtered yet.
        """
        import dataclasses
        if self.dataset is None:
            return None
        # Recompute bandpass WITHOUT envelope so analysis tools see the
        # filtered signal, never its Hilbert envelope.
        fmin = self.spin_fmin.value()
        fmax = self.spin_fmax.value()
        nyq  = self.dataset.fs_hz / 2.0
        if fmin > 0 and fmax > fmin and fmax < nyq:
            tr_filt = self.waterfall.compute_bandpass(fmin, fmax)
        else:
            tr_filt = self.dataset.tr
        return dataclasses.replace(self.dataset, tr=tr_filt)

    def _on_show_spectrogram(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        dlg = SpectrogramDialog(ann, self._analysis_dataset(),
                                self._current_colormap(), parent=self)
        dlg.show()

    def _on_show_spectral(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        dlg = SpectralDialog(ann, self._analysis_dataset(), parent=self)
        dlg.show()

    def _on_show_signal(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        dlg = SignalDialog(ann, self._analysis_dataset(), parent=self)
        dlg.show()

    def _on_show_signal_freq(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        dlg = SignalFreqDialog(ann, self._analysis_dataset(), parent=self)
        dlg.show()

    def _on_show_signal_env(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        dlg = SignalEnvelopeDialog(ann, self._analysis_dataset(), parent=self)
        dlg.show()

    def _on_show_signal_phase(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        dlg = SignalPhaseDialog(ann, self._analysis_dataset(), parent=self)
        dlg.show()

    def _on_show_velocity(self, index: int) -> None:
        ann = self._get_analysis_ann(index)
        if ann is None:
            return
        vmin, vmax = self.waterfall.get_color_levels()
        tr_display = self.waterfall.get_displayed_array()
        dlg = VelocityDialog(ann, self.dataset,
                             self._current_colormap(),
                             vmin=vmin, vmax=vmax,
                             tr_display=tr_display, parent=self)
        dlg.velocity_saved.connect(
            lambda v, r2, idx=index: self._on_velocity_saved(idx, v, r2)
        )
        dlg.show()

    def _on_velocity_saved(self, index: int, velocity_ms: float, r2: float) -> None:
        """Store velocity estimate in the annotation model and auto-save CSV."""
        if index < 0 or index >= len(self._ann_model):
            return
        self._ann_model.update(index, velocity_ms=velocity_ms, velocity_r2=r2)
        self._status_done(f"Velocity saved: {velocity_ms:.1f} m/s  R²={r2:.4f}  (annotation [{self._ann_model[index].id}])")
        # Auto-save CSV so the velocity is persisted immediately
        self._on_annotation_save()

    # ------------------------------------------------------------------
    # Cursor info
    # ------------------------------------------------------------------

    def _on_cursor_info(self, text: str) -> None:
        if text:
            self._status_lbl.setStyleSheet(f"color: {theme.current()['qt_text_dim']}; padding: 0 6px;")
            self._status_lbl.setText(text)
        else:
            self._status_clear()

    # ------------------------------------------------------------------
    # Annotation callbacks
    # ------------------------------------------------------------------

    def _on_annotation_mode_changed(self, active: bool, ann_type_str: str) -> None:
        ann_type = AnnType(ann_type_str)
        for wf in self._waterfalls():
            wf.set_annotation_mode(active, ann_type)
        hints = {
            AnnType.BBOX: "Click × 2 to draw bounding box.",
            AnnType.OBB:  "Click 1 = centre, click 2 = vertex to define OBB.",
            AnnType.KP:   "Click to add keypoints. Press Enter or double-click to confirm.",
            AnnType.LINE: "Click to add vertices. Press Enter or double-click to confirm.",
        }
        if active:
            self._status_processing(f"Annotation mode ON [{ANN_LABEL[ann_type]}] — {hints[ann_type]}")
        else:
            self._status_clear()

    def _ask_annotation_id(self) -> tuple:
        """Show dialog to ask for ID and comment. Returns (id, comment) or (None, None)."""
        dialog = _AnnotationDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return None, None
        return dialog.get_values()

    def _on_bbox_drawn(self, t0: float, t1: float, d0: float, d1: float) -> None:
        if self.dataset is None:
            return
        event_id, comment = self._ask_annotation_id()
        if event_id is None:
            return
        ds = self.dataset
        ti0, ti1, di0, di1 = AnnotationModel.compute_indices(
            t0, t1, d0, d1, ds.time_s, ds.dist_m
        )
        start_dt = ds.start_datetime_utc.isoformat() if ds.start_datetime_utc else ""
        ann = BBoxAnnotation(
            ann_type="bbox", id=event_id, comment=comment,
            t0=t0, t1=t1, d0=d0, d1=d1,
            ti0=ti0, ti1=ti1, di0=di0, di1=di1,
            nt=ds.n_time, nx=ds.n_dist,
            downsample=ds.downsample or 1,
            start_datetime_utc=start_dt,
        )
        model = self._ann_models[AnnType.BBOX]
        model.add(ann)
        idx = len(model) - 1
        for wf in self._waterfalls():
            wf.add_annotation_roi(idx, t0, t1, d0, d1, label=event_id)
        self.ann_widget.refresh_table(self._ann_models)
        self._status_done(f"BBox added: [{event_id}]  t={t0:.2f}–{t1:.2f}s  d={d0:.0f}–{d1:.0f}m")

    def _on_obb_drawn(self, cx: float, cy: float, w: float, h: float,
                      angle_deg: float) -> None:
        if self.dataset is None:
            return
        event_id, comment = self._ask_annotation_id()
        if event_id is None:
            return
        ds = self.dataset
        cx_ti, cy_di = AnnotationModel.coord_to_index(cx, cy, ds.time_s, ds.dist_m)
        w_ti = int(w / (ds.time_s[1] - ds.time_s[0])) if len(ds.time_s) > 1 else 0
        h_di = int(h / (ds.dist_m[1] - ds.dist_m[0])) if len(ds.dist_m) > 1 else 0
        start_dt = ds.start_datetime_utc.isoformat() if ds.start_datetime_utc else ""
        ann = OBBAnnotation(
            ann_type="obb", id=event_id, comment=comment,
            cx_t=cx, cy_d=cy, w_t=w, h_d=h, angle_deg=angle_deg,
            cx_ti=cx_ti, cy_di=cy_di, w_ti=w_ti, h_di=h_di,
            nt=ds.n_time, nx=ds.n_dist,
            downsample=ds.downsample or 1,
            start_datetime_utc=start_dt,
        )
        model = self._ann_models[AnnType.OBB]
        model.add(ann)
        idx = len(model) - 1
        for wf in self._waterfalls():
            wf.add_obb_roi(idx, cx, cy, w, h, angle_deg, label=event_id)
        self.ann_widget.refresh_table(self._ann_models)
        self._status_done(f"OBB added: [{event_id}]  cx={cx:.2f}s  cy={cy:.0f}m  θ={angle_deg:.1f}°")

    def _on_kp_drawn(self, pts_t: list, pts_d: list) -> None:
        if self.dataset is None:
            return
        event_id, comment = self._ask_annotation_id()
        if event_id is None:
            return
        import json
        ds = self.dataset
        pts_ti = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[0]
                  for t, d in zip(pts_t, pts_d)]
        pts_di = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[1]
                  for t, d in zip(pts_t, pts_d)]
        start_dt = ds.start_datetime_utc.isoformat() if ds.start_datetime_utc else ""
        ann = KeypointAnnotation(
            ann_type="kp", id=event_id, comment=comment,
            kp_t=json.dumps(pts_t), kp_d=json.dumps(pts_d),
            kp_ti=json.dumps(pts_ti), kp_di=json.dumps(pts_di),
            nt=ds.n_time, nx=ds.n_dist,
            downsample=ds.downsample or 1,
            start_datetime_utc=start_dt,
        )
        model = self._ann_models[AnnType.KP]
        model.add(ann)
        idx = len(model) - 1
        for wf in self._waterfalls():
            wf.add_kp_roi(idx, pts_t, pts_d, label=event_id)
        self.ann_widget.refresh_table(self._ann_models)
        self._status_done(f"Keypoints added: [{event_id}]  {len(pts_t)} point(s)")

    def _on_line_drawn(self, pts_t: list, pts_d: list) -> None:
        if self.dataset is None:
            return
        event_id, comment = self._ask_annotation_id()
        if event_id is None:
            return
        import json
        ds = self.dataset
        pts_ti = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[0]
                  for t, d in zip(pts_t, pts_d)]
        pts_di = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[1]
                  for t, d in zip(pts_t, pts_d)]
        start_dt = ds.start_datetime_utc.isoformat() if ds.start_datetime_utc else ""
        ann = LineAnnotation(
            ann_type="lin", id=event_id, comment=comment,
            pts_t=json.dumps(pts_t), pts_d=json.dumps(pts_d),
            pts_ti=json.dumps(pts_ti), pts_di=json.dumps(pts_di),
            nt=ds.n_time, nx=ds.n_dist,
            downsample=ds.downsample or 1,
            start_datetime_utc=start_dt,
        )
        model = self._ann_models[AnnType.LINE]
        model.add(ann)
        idx = len(model) - 1
        for wf in self._waterfalls():
            wf.add_line_roi(idx, pts_t, pts_d, label=event_id)
        self.ann_widget.refresh_table(self._ann_models)
        self._status_done(f"Line added: [{event_id}]  {len(pts_t)} vertices")

    def _on_annotation_selected(self, index: int) -> None:
        """Open ID/comment edit dialog for the annotation at flat index."""
        self._all_waterfalls_highlight_roi(index)
        if index < 0:
            return
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > index:
                local_idx = index - row
                ann = model[local_idx]
                dialog = _AnnotationDialog(self, event_id=ann.id, comment=ann.comment)
                if dialog.exec_() == QtWidgets.QDialog.Accepted:
                    new_id, new_comment = dialog.get_values()
                    model.update(local_idx, id=new_id, comment=new_comment)
                    self._redraw_all_annotation_rois()
                    self.ann_widget.refresh_table(self._ann_models)
                self._all_waterfalls_highlight_roi(-1)
                return
            row += len(model)

    def _on_annotation_edit_shape(self, index: int) -> None:
        """Enter shape-edit mode for the annotation at flat index."""
        if self.dataset is None or index < 0:
            return
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > index:
                local_idx = index - row
                ann = model[local_idx]
                if ann_type == AnnType.BBOX:
                    # Only activate on the currently visible waterfall tab,
                    # not all three — each would create its own floating toolbar.
                    active_wf = self._active_waterfall()
                    QtCore.QTimer.singleShot(0, lambda a=ann, idx=index, wf=active_wf:
                        wf.enter_bbox_edit_mode(idx, a.t0, a.t1, a.d0, a.d1)
                    )
                    self._status_processing(
                        f"Shape edit [{ann.id}] — drag corners to resize."
                    )
                elif ann_type == AnnType.OBB:
                    active_wf = self._active_waterfall()
                    QtCore.QTimer.singleShot(0, lambda a=ann, idx=index, wf=active_wf:
                        wf.enter_obb_edit_mode(
                            idx, a.cx_t, a.cy_d, a.w_t, a.h_d, a.angle_deg
                        )
                    )
                    self._status_processing(
                        f"Shape edit [{ann.id}] — drag handles to reshape OBB."
                    )
                elif ann_type == AnnType.KP:
                    import json as _json
                    pts_t = _json.loads(ann.kp_t)
                    pts_d = _json.loads(ann.kp_d)
                    active_wf = self._active_waterfall()
                    QtCore.QTimer.singleShot(0, lambda pts_t=pts_t, pts_d=pts_d,
                                             idx=index, wf=active_wf:
                        wf.enter_kp_edit_mode(idx, pts_t, pts_d)
                    )
                    self._status_processing(
                        f"Shape edit [{ann.id}] — drag keypoints to reposition."
                    )
                elif ann_type == AnnType.LINE:
                    import json as _json
                    pts_t = _json.loads(ann.pts_t)
                    pts_d = _json.loads(ann.pts_d)
                    active_wf = self._active_waterfall()
                    QtCore.QTimer.singleShot(0, lambda pts_t=pts_t, pts_d=pts_d,
                                             idx=index, wf=active_wf:
                        wf.enter_line_edit_mode(idx, pts_t, pts_d)
                    )
                    self._status_processing(
                        f"Shape edit [{ann.id}] — drag vertices to reshape line."
                    )
                return
            row += len(model)

    def _active_waterfall(self):
        """Return the waterfall widget currently visible in the tab."""
        idx = self.tab_widget.currentIndex()
        if idx == 0:
            return self.waterfall
        elif idx == 1:
            return self.waterfall_fk
        else:
            return self.waterfall_rgb

    def _on_annotation_delete(self, index: int) -> None:
        # Find which model this row index belongs to
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > index:
                local_idx = index - row
                model.remove(local_idx)
                self._all_waterfalls_remove_roi(index)
                self.ann_widget.refresh_table(self._ann_models)
                return
            row += len(model)

    def _on_annotation_save(self) -> None:
        if not self._export_dir:
            self._export_dir = self._current_dir or ""
        if not self._export_dir:
            self._export_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select export directory"
            )
            if not self._export_dir:
                return
            self.ann_widget.set_export_dir(self._export_dir)

        data_basename = os.path.splitext(os.path.basename(
            self._current_data_path or self.lbl_file.text()
        ))[0]

        saved = []
        errors = []
        for ann_type, model in self._ann_models.items():
            if len(model) == 0:
                continue
            csv_path = os.path.join(
                self._export_dir,
                data_basename + ANN_SUFFIX[ann_type]
            )
            try:
                model.save(csv_path)
                saved.append(os.path.basename(csv_path))
            except Exception as exc:
                errors.append(f"{ann_type.value}: {exc}")

        if errors:
            QtWidgets.QMessageBox.critical(self, "Save error", "\n".join(errors))
        if saved:
            self._status_done(f"Saved: {', '.join(saved)}")
        elif not errors:
            self._status_done("No annotations to save.")


        self._status_done(f"Annotations saved: {csv_path}")

    # ------------------------------------------------------------------
    # Tab switching — keep view range in sync across waterfalls
    # ------------------------------------------------------------------

    def _on_tab_changed(self, index: int) -> None:
        """When switching tabs, copy the current view range from whichever
        tab was active just before this switch (true bidirectional sync —
        a zoom/crop made in any tab, including RGB, propagates to the
        others). Lazily compute F-K / RGB the first time their tab is
        visited for this file/parameter combination."""
        if self.dataset is None:
            return

        prev_index = self._last_tab_index
        self._last_tab_index = index

        # Lazy F-K computation: only the first time the tab is opened.
        if index == 1 and self._tr_fk_base is None:
            self._apply_fk()
            return  # _apply_fk already handles view inheritance from Raw

        # Lazy RGB computation: only the first time the tab is opened.
        if index == 2 and self._rgb_array is None:
            self._apply_rgb()
            return  # _apply_rgb already handles view inheritance from Raw

        # Map tab index to waterfall widget
        wf_map = {0: self.waterfall, 1: self.waterfall_fk, 2: self.waterfall_rgb}
        target = wf_map.get(index)
        source = wf_map.get(prev_index)
        if target is None or source is None or source is target:
            return
        if source.dataset is None:
            return

        xr = source.plot_widget.getPlotItem().vb.viewRange()[0]
        yr = source.plot_widget.getPlotItem().vb.viewRange()[1]
        target.apply_time_range(xr[0], xr[1])
        target.apply_distance_range(yr[0], yr[1])

    # ------------------------------------------------------------------
    # Multi-waterfall ROI helpers (Raw + F-K share the same annotations)
    # ------------------------------------------------------------------

    def _waterfalls(self):
        """Return all active WaterfallWidgets."""
        return [self.waterfall, self.waterfall_fk, self.waterfall_rgb]

    def _all_waterfalls_add_roi(self, index, t0, t1, d0, d1, label=""):
        for wf in self._waterfalls():
            wf.add_annotation_roi(index, t0, t1, d0, d1, label=label)

    def _all_waterfalls_remove_roi(self, index):
        for wf in self._waterfalls():
            wf.remove_annotation_roi(index)

    def _all_waterfalls_clear_rois(self):
        for wf in self._waterfalls():
            wf.clear_annotation_rois()

    def _all_waterfalls_update_label(self, index, label):
        for wf in self._waterfalls():
            wf.update_annotation_label(index, label)

    def _all_waterfalls_highlight_roi(self, index):
        for wf in self._waterfalls():
            wf.highlight_annotation_roi(index)

    # ------------------------------------------------------------------
    # FK filter
    # ------------------------------------------------------------------

    def _apply_fk(self) -> None:
        """Compute FK-filtered array and display in the FK tab."""
        if self.dataset is None:
            self._status_error("No data loaded.")
            return

        tr_src = self.waterfall.get_displayed_array()
        if tr_src is None:
            self._status_error("Apply bandpass filter first (Raw tab).")
            return

        c_min  = self.spin_cmin.value()
        c_max  = self.spin_cmax.value()
        fk_fmin = self.spin_fkmin.value()
        fk_fmax = self.spin_fkmax.value()

        if c_min >= c_max:
            self._status_error("c_min must be < c_max.")
            return
        if fk_fmin >= fk_fmax:
            self._status_error("FK fmin must be < fmax.")
            return

        self._status_processing(f"Computing F-K filter  c=[{c_min:.0f}–{c_max:.0f}] m/s  f=[{fk_fmin:.1f}–{fk_fmax:.1f}] Hz …")
        QtWidgets.QApplication.processEvents()

        # Hide ROIs and blank the F-K image while computing: the ROIs are
        # positioned for the OLD image rect/range, so they appear at stale
        # (wrong) screen coordinates until the new image's setRect/setRange
        # repositions everything. Same pattern as the RGB tab.
        self.waterfall_fk.set_rois_visible(False)
        self.waterfall_fk.image_item.clear()
        QtWidgets.QApplication.processEvents()

        try:
            import das4whales.dsp as dsp

            ds = self.dataset
            dx = float(ds.dist_m[1] - ds.dist_m[0]) if ds.n_dist > 1 else 1.0
            n_dist, n_time = tr_src.shape
            selected_channels = [0, n_dist, 1]

            fk_params = {
                'c_min': c_min,
                'c_max': c_max,
                'fmin':  fk_fmin,
                'fmax':  fk_fmax,
            }

            fk_filter = dsp.hybrid_ninf_gs_filter_design(
                (n_dist, n_time), selected_channels, dx, ds.fs_hz,
                fk_params, display_filter=False,
            )
            tr_fk_base = dsp.fk_filter_sparsefilt(
                tr_src, fk_filter, tapering=True
            ).astype(np.float32)

            # Cache the raw FK array (before envelope) for fast re-render on toggle
            self._tr_fk_base = tr_fk_base

        except Exception as exc:
            self._status_error(f"F-K filter error: {exc}")
            self.waterfall_fk.set_rois_visible(True)
            return

        self._render_fk()
        self.waterfall_fk.set_rois_visible(True)
        self._status_done(f"F-K filter applied  c=[{c_min:.0f}–{c_max:.0f}] m/s  f=[{fk_fmin:.1f}–{fk_fmax:.1f}] Hz")

    def _render_fk(self) -> None:
        """Re-render the FK waterfall from the cached base array + current envelope state.
        Never recomputes the F-K filter — just applies/removes Hilbert on the cached result."""
        if self._tr_fk_base is None or self.dataset is None:
            return

        tr_fk = self._tr_fk_base
        if self._envelope_fk:
            tr_fk = self.waterfall_fk.compute_envelope(tr_fk)

        vmin_fk = self.spin_vmin.value()
        vmax_fk = self.spin_vmax.value()

        src_wf = (self.waterfall_fk if self.waterfall_fk.dataset is not None
                  else self.waterfall)
        xr = src_wf.plot_widget.getPlotItem().vb.viewRange()[0]
        yr = src_wf.plot_widget.getPlotItem().vb.viewRange()[1]

        self.waterfall_fk.load_and_display(
            self.dataset, vmin=vmin_fk, vmax=vmax_fk,
            tr_override=tr_fk,
        )
        self.waterfall_fk.apply_time_range(xr[0], xr[1])
        self.waterfall_fk.apply_distance_range(yr[0], yr[1])

    # ------------------------------------------------------------------
    # RGB multispectral composite
    # ------------------------------------------------------------------

    def _apply_rgb(self) -> None:
        """
        Compute the multispectral RGB composite and display it in the RGB tab.

        Each band (R, G, B) is bandpass-filtered from the RAW trace
        (self.dataset.tr), never from the already-filtered Raw display —
        filtering an already-filtered signal would distort each band's
        spectral content, since every band IS a different filtered version
        of the same raw signal.
        """
        if self.dataset is None:
            self._status_error("No data loaded.")
            return

        r_min, r_max = self.spin_rmin.value(), self.spin_rmax.value()
        g_min, g_max = self.spin_gmin.value(), self.spin_gmax.value()
        b_min, b_max = self.spin_bmin.value(), self.spin_bmax.value()
        percentile   = self.spin_rgb_pct.value()

        for name, lo, hi in [("R", r_min, r_max), ("G", g_min, g_max), ("B", b_min, b_max)]:
            if lo >= hi:
                self._status_error(f"{name}: min must be < max.")
                return

        # Hide the RGB tab's ROIs while computing: they were positioned for
        # the previous image's rect/range, so leaving them visible mid-compute
        # makes them flash at stale (visually wrong) coordinates until the
        # new image's setRect/setRange repositions everything. Blanking the
        # image too keeps the tab fully black until the new composite and its
        # correctly-positioned ROIs are ready together.
        self.waterfall_rgb.set_rois_visible(False)
        self.waterfall_rgb.image_item.clear()

        self._status_processing(
            f"Computing RGB composite  R=[{r_min:.1f}-{r_max:.1f}]  "
            f"G=[{g_min:.1f}-{g_max:.1f}]  B=[{b_min:.1f}-{b_max:.1f}] Hz "
            f"(p{percentile:.0f})…"
        )
        QtWidgets.QApplication.processEvents()

        try:
            from dasexplorer.core.rgb import compute_rgb_composite
            rgb = compute_rgb_composite(
                self.dataset.tr, self.dataset.fs_hz,
                r_band=(r_min, r_max), g_band=(g_min, g_max), b_band=(b_min, b_max),
                percentile=percentile,
            )
        except Exception as exc:
            self._status_error(f"RGB composite error: {exc}")
            self.waterfall_rgb.set_rois_visible(True)
            return

        self._rgb_array = rgb

        # Inherit the current view range from Raw (the anchor view)
        xr = self.waterfall.plot_widget.getPlotItem().vb.viewRange()[0]
        yr = self.waterfall.plot_widget.getPlotItem().vb.viewRange()[1]

        self.waterfall_rgb.display_rgb_array(rgb, self.dataset)
        self.waterfall_rgb.apply_time_range(xr[0], xr[1])
        self.waterfall_rgb.apply_distance_range(yr[0], yr[1])
        self.waterfall_rgb.set_rois_visible(True)

        self._status_done(
            f"RGB composite applied  R=[{r_min:.1f}-{r_max:.1f}]  "
            f"G=[{g_min:.1f}-{g_max:.1f}]  B=[{b_min:.1f}-{b_max:.1f}] Hz"
        )

    def _on_bbox_edited(self, flat_idx: int, t0: float, t1: float,
                        d0: float, d1: float) -> None:
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > flat_idx and ann_type == AnnType.BBOX:
                local_idx = flat_idx - row
                ds = self.dataset
                ti0, ti1, di0, di1 = AnnotationModel.compute_indices(
                    t0, t1, d0, d1, ds.time_s, ds.dist_m
                )
                model.update(local_idx, t0=t0, t1=t1, d0=d0, d1=d1,
                             ti0=ti0, ti1=ti1, di0=di0, di1=di1)
                self._redraw_all_annotation_rois()
                self.ann_widget.refresh_table(self._ann_models)
                for wf in self._waterfalls():
                    wf.cancel_edit()
                self._status_done("BBox updated.")
                return
            row += len(model)

    def _on_obb_edited(self, flat_idx: int, cx: float, cy: float,
                       w: float, h: float, angle_deg: float) -> None:
        import math
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > flat_idx and ann_type == AnnType.OBB:
                local_idx = flat_idx - row
                ds = self.dataset
                cx_ti, cy_di = AnnotationModel.coord_to_index(cx, cy, ds.time_s, ds.dist_m)
                dt = float(ds.time_s[1] - ds.time_s[0]) if len(ds.time_s) > 1 else 1
                dd = float(ds.dist_m[1] - ds.dist_m[0]) if len(ds.dist_m) > 1 else 1
                w_ti = int(w / dt)
                h_di = int(h / dd)
                model.update(local_idx, cx_t=cx, cy_d=cy, w_t=w, h_d=h,
                             angle_deg=angle_deg, cx_ti=cx_ti, cy_di=cy_di,
                             w_ti=w_ti, h_di=h_di)
                self._redraw_all_annotation_rois()
                self.ann_widget.refresh_table(self._ann_models)
                for wf in self._waterfalls():
                    wf.cancel_edit()
                self._status_done("OBBox updated.")
                return
            row += len(model)

    def _on_kp_edited(self, flat_idx: int, pts_t: list, pts_d: list) -> None:
        import json as _json
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > flat_idx and ann_type == AnnType.KP:
                local_idx = flat_idx - row
                ds = self.dataset
                pts_ti = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[0]
                          for t, d in zip(pts_t, pts_d)]
                pts_di = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[1]
                          for t, d in zip(pts_t, pts_d)]
                model.update(local_idx,
                             kp_t=_json.dumps(pts_t), kp_d=_json.dumps(pts_d),
                             kp_ti=_json.dumps(pts_ti), kp_di=_json.dumps(pts_di))
                self._redraw_all_annotation_rois()
                self.ann_widget.refresh_table(self._ann_models)
                for wf in self._waterfalls():
                    wf.cancel_edit()
                self._status_done("Keypoints updated.")
                return
            row += len(model)

    def _on_line_edited(self, flat_idx: int, pts_t: list, pts_d: list) -> None:
        import json as _json
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > flat_idx and ann_type == AnnType.LINE:
                local_idx = flat_idx - row
                ds = self.dataset
                pts_ti = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[0]
                          for t, d in zip(pts_t, pts_d)]
                pts_di = [AnnotationModel.coord_to_index(t, d, ds.time_s, ds.dist_m)[1]
                          for t, d in zip(pts_t, pts_d)]
                model.update(local_idx,
                             pts_t=_json.dumps(pts_t), pts_d=_json.dumps(pts_d),
                             pts_ti=_json.dumps(pts_ti), pts_di=_json.dumps(pts_di))
                self._redraw_all_annotation_rois()
                self.ann_widget.refresh_table(self._ann_models)
                for wf in self._waterfalls():
                    wf.cancel_edit()
                self._status_done("Line updated.")
                return
            row += len(model)

    def _on_annotation_id_changed(self, flat_idx: int, new_id: str) -> None:
        """Called when the user edits the ID directly in the events table."""
        if not new_id.strip():
            return
        row = 0
        for ann_type in (AnnType.BBOX, AnnType.OBB, AnnType.KP, AnnType.LINE):
            model = self._ann_models[ann_type]
            if row + len(model) > flat_idx:
                local_idx = flat_idx - row
                model.update(local_idx, id=new_id.strip())
                self._redraw_all_annotation_rois()
                # Don't call refresh_table here — it would trigger itemChanged again
                return
            row += len(model)

    def _on_annotation_clear(self) -> None:
        for model in self._ann_models.values():
            model.clear()
        self._all_waterfalls_clear_rois()
        self.ann_widget.refresh_table(self._ann_models)

    def _on_export_path_changed(self, directory: str) -> None:
        self._export_dir = directory

    def _on_csv_file_selected(self, path: str) -> None:
        pass  # CSV file list removed from UI

    def _load_annotations_for_current_file(self, data_label: str) -> None:
        """Auto-load all type-specific CSVs matching the current data file."""
        self._all_waterfalls_clear_rois()
        for model in self._ann_models.values():
            model.clear()

        if not self._export_dir:
            return

        base = os.path.splitext(data_label)[0]
        any_loaded = False
        for ann_type, model in self._ann_models.items():
            csv_path = os.path.join(
                self._export_dir, base + ANN_SUFFIX[ann_type]
            )
            if os.path.exists(csv_path):
                try:
                    model.load(csv_path)
                    any_loaded = True
                except Exception:
                    pass

        if any_loaded:
            self._redraw_all_annotation_rois()
        self.ann_widget.refresh_table(self._ann_models)

    def _redraw_all_annotation_rois(self) -> None:
        """Re-draw all loaded annotations on the waterfalls."""
        self._all_waterfalls_clear_rois()
        import json
        for ann_type, model in self._ann_models.items():
            for i, ann in enumerate(model):
                for wf in self._waterfalls():
                    if ann_type == AnnType.BBOX:
                        wf.add_annotation_roi(i, ann.t0, ann.t1, ann.d0, ann.d1, label=ann.id)
                    elif ann_type == AnnType.OBB:
                        fidx = self._flat_idx(ann_type, i)
                        wf.add_obb_roi(i, ann.cx_t, ann.cy_d, ann.w_t, ann.h_d,
                                       ann.angle_deg, label=ann.id, flat_idx=fidx)
                    elif ann_type == AnnType.KP:
                        pts_t = json.loads(ann.kp_t)
                        pts_d = json.loads(ann.kp_d)
                        fidx = self._flat_idx(ann_type, i)
                        wf.add_kp_roi(i, pts_t, pts_d, label=ann.id, flat_idx=fidx)
                    elif ann_type == AnnType.LINE:
                        pts_t = json.loads(ann.pts_t)
                        pts_d = json.loads(ann.pts_d)
                        fidx = self._flat_idx(ann_type, i)
                        wf.add_line_roi(i, pts_t, pts_d, label=ann.id, flat_idx=fidx)




# ---------------------------------------------------------------------------
# Helper dialog
# ---------------------------------------------------------------------------

class _SaveExportDialog(QtWidgets.QDialog):
    """Small dialog with a single checkbox: export only the currently
    selected View (Time/Distance range) instead of the full array.
    Shared by the Save as NPZ and Save as MAT actions."""

    def __init__(self, parent=None, format_name: str = "NPZ"):
        super().__init__(parent)
        self.setWindowTitle(f"Save as {format_name}")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        lbl = QtWidgets.QLabel(
            "Export the dataset and all metadata needed to reload it."
        )
        layout.addWidget(lbl)

        self.chk_selected_view = QtWidgets.QCheckBox("Export Selected View")
        self.chk_selected_view.setToolTip(
            "If checked, export only the Time/Distance range currently set "
            "in the View panel. If unchecked, export the full array."
        )
        layout.addWidget(self.chk_selected_view)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QtWidgets.QPushButton("Export…")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

    def export_selected_view(self) -> bool:
        return self.chk_selected_view.isChecked()


class _AnnotationDialog(QtWidgets.QDialog):
    """Small dialog to capture event ID and comment for a new/edited annotation."""

    def __init__(self, parent=None, event_id: str = "", comment: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Annotation")
        self.setMinimumWidth(320)

        layout = QtWidgets.QFormLayout(self)

        self.edit_id = QtWidgets.QLineEdit(event_id)
        # self.edit_id.setPlaceholderText("e.g.  fin_whale, ship, noise…")
        layout.addRow("Event ID:", self.edit_id)

        self.edit_comment = QtWidgets.QLineEdit(comment)
        # self.edit_comment.setPlaceholderText("Optional free-text comment")
        layout.addRow("Comment:", self.edit_comment)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

        self.edit_id.setFocus()

    def get_values(self):
        return self.edit_id.text().strip(), self.edit_comment.text().strip()
