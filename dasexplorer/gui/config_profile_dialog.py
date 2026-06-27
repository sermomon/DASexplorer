"""
Configuration Profile dialog for DAS Explorer.

Displays all profiles defined in config.json. Each profile is shown as a tab
with its full parameter set. Users can edit values, add new profiles by
duplicating an existing one, rename profiles, and delete profiles.
Changes are written back to config.json via the Apply button.
"""

import json
import copy
from PyQt5 import QtWidgets, QtCore, QtGui

from dasexplorer.core.config import (
    _CONFIG_PATH, get_all_profiles, get_profile,
    save_profiles, get_default_profile_key, reload as config_reload,
)
from dasexplorer.core.readers import INTERROGATOR_TYPES, INTERROGATOR_LABELS
from dasexplorer.gui import theme


# ---------------------------------------------------------------------------
# Field catalogue  —  (key, display_name, description, type_hint)
# ---------------------------------------------------------------------------
_FIELDS = [
    # ---- Identity ----
    ("label",            "Profile label",
     "Name shown in the Profile combo box.",
     "str"),
    ("interrogator",     "Interrogator",
     f"Reader function to use. Options: {', '.join(INTERROGATOR_TYPES)}.",
     "str"),
    ("file_extensions",  "File extensions",
     "Comma-separated list of file extensions to show (e.g. .bin  or  .h5,.hdf5).",
     "str_list"),
    ("num_files",        "Num files (HDAS only)",
     "Number of consecutive HDAS .bin files to load as one dataset.",
     "int"),
    # ---- Time / Distance ----
    ("tmin_s",           "Time min [s]",
     "Default start time shown on load. null = beginning of file.",
     "float_or_null"),
    ("tmax_s",           "Time max [s]",
     "Default end time shown on load. null = end of file.",
     "float_or_null"),
    ("dmin_m",           "Distance min [m]",
     "Default start distance. null = beginning of cable.",
     "float_or_null"),
    ("dmax_m",           "Distance max [m]",
     "Default end distance. null = end of cable.",
     "float_or_null"),
    # ---- Color scale ----
    ("vmin",             "Color scale min",
     "Minimum value of the colour scale (Raw waterfall).",
     "float"),
    ("vmax",             "Color scale max",
     "Maximum value of the colour scale (Raw waterfall).",
     "float"),
    ("colormap",         "Colormap",
     "Default colormap. Options: Rainbow, Turbo, Grays, Viridis, Magma, Seismic.",
     "str"),
    # ---- Bandpass filter ----
    ("fmin_hz",          "Filter f-min [Hz]",
     "High-pass corner of the Butterworth bandpass filter.",
     "float"),
    ("fmax_hz",          "Filter f-max [Hz]",
     "Low-pass corner. null = Nyquist − fmax_offset_hz.",
     "float_or_null"),
    ("fmax_offset_hz",   "Filter f-max offset [Hz]",
     "Subtracted from Nyquist to compute f-max when fmax_hz is null.",
     "float"),
    # ---- Decimation / Envelope ----
    ("stride",           "Stride",
     "Spatial decimation factor at load time (1 = no decimation).",
     "int"),
    ("envelope",         "Hilbert envelope (Raw)",
     "Apply Hilbert envelope to the Raw view on load.",
     "bool"),
    # ---- F-K filter ----
    ("fk_cmin_ms",       "F-K c-min [m/s]",
     "Minimum apparent velocity for the F-K velocity filter.",
     "float"),
    ("fk_cmax_ms",       "F-K c-max [m/s]",
     "Maximum apparent velocity for the F-K velocity filter.",
     "float"),
    ("fk_fmin_hz",       "F-K f-min [Hz]",
     "Frequency lower bound of the F-K filter.",
     "float"),
    ("fk_fmax_hz",       "F-K f-max [Hz]",
     "Frequency upper bound. null = Nyquist − fk_fmax_offset_hz.",
     "float_or_null"),
    ("fk_fmax_offset_hz","F-K f-max offset [Hz]",
     "Subtracted from Nyquist to compute F-K f-max when fk_fmax_hz is null.",
     "float"),
    ("fk_envelope",      "Hilbert envelope (F-K)",
     "Apply Hilbert envelope to the F-K view on load.",
     "bool"),
    # ---- RGB multispectral ----
    ("rgb_rmin_hz",      "RGB R-band min [Hz]", "Lower bound of the Red channel.", "float"),
    ("rgb_rmax_hz",      "RGB R-band max [Hz]", "Upper bound of the Red channel.", "float"),
    ("rgb_gmin_hz",      "RGB G-band min [Hz]", "Lower bound of the Green channel.", "float"),
    ("rgb_gmax_hz",      "RGB G-band max [Hz]", "Upper bound of the Green channel.", "float"),
    ("rgb_bmin_hz",      "RGB B-band min [Hz]", "Lower bound of the Blue channel.", "float"),
    ("rgb_bmax_hz",      "RGB B-band max [Hz]", "Upper bound of the Blue channel.", "float"),
    ("rgb_percentile",   "RGB percentile",
     "Percentile of |signal| used as per-band normalisation ceiling (typ. 90–99).",
     "float"),
    # ---- Misc ----
    ("default_view",     "Default view",
     "Which tab to show first after loading: 'raw' or 'fk'.",
     "str"),
]


def _format_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _parse_value(text: str, type_hint: str):
    text = text.strip()
    if type_hint == "float_or_null":
        if text.lower() in ("null", "none", ""):
            return None
        return float(text)
    if type_hint == "float":
        return float(text)
    if type_hint == "int":
        return int(text)
    if type_hint == "bool":
        if text.lower() in ("true", "1", "yes"):
            return True
        if text.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"Expected true/false, got '{text}'")
    if type_hint == "str_list":
        # "  .bin , .hdf5 " → [".bin", ".hdf5"]
        return [x.strip() for x in text.split(",") if x.strip()]
    return text  # str


class ConfigurationProfileDialog(QtWidgets.QDialog):
    """
    Read/edit per-profile defaults from config.json.

    Layout
    ------
    - Tab bar: one tab per profile
    - Buttons above tabs: New Profile (duplicate), Rename, Delete, Set as Default
    - Each tab: table with [Parameter, Value, Description]
    - Bottom: Apply (writes to config.json) and Close buttons
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration Profiles")
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.60), int(screen.height() * 0.80))

        self._tables: dict = {}        # profile_key → QTableWidget
        self._profile_keys: list = []  # ordered list of keys
        self._build_ui()
        self._load_all_profiles()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        info = QtWidgets.QLabel(
            "Each profile defines an interrogator + visualisation defaults. "
            "Edit values directly and click <b>Apply</b> to save to <b>config.json</b>."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Profile management buttons
        mgmt_row = QtWidgets.QHBoxLayout()
        self.btn_new = QtWidgets.QPushButton("Duplicate Profile")
        self.btn_new.setToolTip("Create a copy of the current profile with a new name")
        self.btn_new.clicked.connect(self._on_duplicate)
        mgmt_row.addWidget(self.btn_new)

        self.btn_rename = QtWidgets.QPushButton("Rename Profile")
        self.btn_rename.clicked.connect(self._on_rename)
        mgmt_row.addWidget(self.btn_rename)

        self.btn_delete = QtWidgets.QPushButton("Delete Profile")
        self.btn_delete.clicked.connect(self._on_delete)
        mgmt_row.addWidget(self.btn_delete)

        self.btn_set_default = QtWidgets.QPushButton("Set as Default")
        self.btn_set_default.setToolTip("Load this profile automatically on app start")
        self.btn_set_default.clicked.connect(self._on_set_default)
        mgmt_row.addWidget(self.btn_set_default)

        mgmt_row.addStretch()
        layout.addLayout(mgmt_row)

        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget, 1)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        self.btn_reset = QtWidgets.QPushButton("Reload from File")
        self.btn_reset.setMinimumWidth(140)
        self.btn_reset.setToolTip("Discard unsaved edits and reload from config.json")
        self.btn_reset.clicked.connect(self._load_all_profiles)
        btn_row.addWidget(self.btn_reset)

        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.setMinimumWidth(100)
        self.btn_apply.setDefault(True)
        self.btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self.btn_apply)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setMinimumWidth(80)
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    def _make_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(len(_FIELDS), 3)
        table.setHorizontalHeaderLabels(["Parameter", "Value", "Description"])
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        table.setColumnWidth(1, 160)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        table.verticalHeader().setDefaultSectionSize(22)

        for row, (key, display_name, description, type_hint) in enumerate(_FIELDS):
            name_item = QtWidgets.QTableWidgetItem(display_name)
            name_item.setFlags(QtCore.Qt.ItemIsEnabled)
            name_item.setToolTip(f"config.json key: {key}")
            bold = any(tag in display_name for tag in ("F-K", "RGB", "Profile", "Interrogator"))
            if bold:
                f = QtGui.QFont()
                f.setBold(True)
                name_item.setFont(f)
            table.setItem(row, 0, name_item)

            val_item = QtWidgets.QTableWidgetItem("")
            val_item.setData(QtCore.Qt.UserRole, (key, type_hint))
            table.setItem(row, 1, val_item)

            desc_item = QtWidgets.QTableWidgetItem(description)
            desc_item.setFlags(QtCore.Qt.ItemIsEnabled)
            desc_item.setForeground(QtGui.QColor(theme.current()["qt_text_dim"]))
            table.setItem(row, 2, desc_item)

        return table

    # ------------------------------------------------------------------
    def _load_all_profiles(self) -> None:
        """Reload all profiles from config and rebuild the tab widget."""
        config_reload()
        profiles = get_all_profiles()
        default_key = get_default_profile_key()

        self.tab_widget.blockSignals(True)
        # Remove all existing tabs
        while self.tab_widget.count():
            self.tab_widget.removeTab(0)
        self._tables.clear()
        self._profile_keys.clear()

        for pkey, pdata in profiles.items():
            table = self._make_table()
            self._tables[pkey] = table
            self._profile_keys.append(pkey)
            label = pdata.get("label", pkey)
            tab_label = f"★ {label}" if pkey == default_key else label
            self.tab_widget.addTab(table, tab_label)
            # Populate values
            full = get_profile(pkey)
            for row in range(table.rowCount()):
                val_item = table.item(row, 1)
                if val_item is None:
                    continue
                key, _ = val_item.data(QtCore.Qt.UserRole)
                val_item.setText(_format_value(full.get(key)))
                val_item.setForeground(QtGui.QColor(theme.current()["qt_text"]))

        self.tab_widget.blockSignals(False)

    def _current_profile_key(self) -> str:
        idx = self.tab_widget.currentIndex()
        if 0 <= idx < len(self._profile_keys):
            return self._profile_keys[idx]
        return ""

    # ------------------------------------------------------------------
    def _on_duplicate(self) -> None:
        src_key = self._current_profile_key()
        if not src_key:
            return
        new_label, ok = QtWidgets.QInputDialog.getText(
            self, "Duplicate Profile",
            "Name for the new profile:",
            text=f"{self._tables[src_key].item(0,1).text()} (copy)"
        )
        if not ok or not new_label.strip():
            return
        # Generate a safe key
        base = new_label.strip().lower().replace(" ", "_").replace("/", "_")
        profiles = get_all_profiles()
        new_key = base
        i = 2
        while new_key in profiles:
            new_key = f"{base}_{i}"
            i += 1
        # Copy data
        new_data = copy.deepcopy(profiles.get(src_key, {}))
        new_data["label"] = new_label.strip()
        profiles[new_key] = new_data
        save_profiles(profiles, get_default_profile_key())
        self._load_all_profiles()
        # Switch to new tab
        if new_key in self._profile_keys:
            self.tab_widget.setCurrentIndex(self._profile_keys.index(new_key))

    def _on_rename(self) -> None:
        pkey = self._current_profile_key()
        if not pkey:
            return
        profiles = get_all_profiles()
        old_label = profiles.get(pkey, {}).get("label", pkey)
        new_label, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Profile", "New label:", text=old_label
        )
        if not ok or not new_label.strip():
            return
        profiles[pkey]["label"] = new_label.strip()
        save_profiles(profiles, get_default_profile_key())
        self._load_all_profiles()

    def _on_delete(self) -> None:
        pkey = self._current_profile_key()
        if not pkey:
            return
        profiles = get_all_profiles()
        if len(profiles) <= 1:
            QtWidgets.QMessageBox.warning(self, "Cannot delete",
                                          "At least one profile must exist.")
            return
        label = profiles[pkey].get("label", pkey)
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{label}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        del profiles[pkey]
        default = get_default_profile_key()
        if default == pkey:
            default = next(iter(profiles))
        save_profiles(profiles, default)
        self._load_all_profiles()

    def _on_set_default(self) -> None:
        pkey = self._current_profile_key()
        if not pkey:
            return
        profiles = get_all_profiles()
        save_profiles(profiles, pkey)
        self._load_all_profiles()
        QtWidgets.QMessageBox.information(
            self, "Default set",
            f"Profile '{profiles[pkey].get('label', pkey)}' will be loaded on next app start."
        )

    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        errors = []
        profiles = get_all_profiles()
        updated_profiles = {}

        for pkey in self._profile_keys:
            table = self._tables.get(pkey)
            if table is None:
                continue
            p_updates = {}
            for row in range(table.rowCount()):
                val_item = table.item(row, 1)
                if val_item is None:
                    continue
                key, type_hint = val_item.data(QtCore.Qt.UserRole)
                text = val_item.text().strip()
                try:
                    parsed = _parse_value(text, type_hint)
                    p_updates[key] = parsed
                    val_item.setForeground(QtGui.QColor(theme.current()["qt_text"]))
                except (ValueError, TypeError) as exc:
                    val_item.setForeground(QtGui.QColor("#ff4444"))
                    errors.append(f"[{pkey}] {key}: {exc}")
            # Merge with existing profile data (preserve keys not in _FIELDS)
            existing = dict(profiles.get(pkey, {}))
            existing.update(p_updates)
            updated_profiles[pkey] = existing

        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Validation errors",
                "The following fields contain invalid values (shown in red):\n\n"
                + "\n".join(errors)
            )
            return

        save_profiles(updated_profiles, get_default_profile_key())
        config_reload()

        QtWidgets.QMessageBox.information(
            self, "Applied",
            "Configuration saved to config.json.\n"
            "Values will be applied the next time a file is loaded."
        )
