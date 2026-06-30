"""
Analysis dialog windows for DAS Explorer.

QDialog subclasses launched from the annotation right-click menu:
  A) SpectrogramDialog     - time-frequency spectrogram with channel scrollbar
  B) SpectralDialog        - FFT magnitude spectra of bbox channels
  C) SignalDialog          - time-domain waveform with Fix Channel
  D) SignalFreqDialog      - frequency-domain FFT magnitude with Fix Channel
  E) SignalEnvelopeDialog  - Hilbert envelope amplitude with Fix Channel
  F) SignalPhaseDialog     - unwrapped instantaneous phase with Fix Channel
  G) VelocityDialog        - apparent velocity estimation from point picks
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from dasexplorer.core.analysis import (compute_spectrogram, compute_spectrum, select_channels_for_spectral)
from dasexplorer.gui import theme

# Tab10 colours for fixed channels / individual spectra
_TAB10 = [
    (31,  119, 180), (255, 127,  14), (44,  160,  44),
    (214,  39,  40), (148, 103, 189), (140,  86,  75),
    (227, 119, 194), (127, 127, 127), (188, 189,  34), (23, 190, 207),
]


def _make_spinbox(value, min_val, max_val, step=1, decimals=0):
    if decimals == 0:
        sb = QtWidgets.QSpinBox()
        sb.setRange(int(min_val), int(max_val))
        sb.setSingleStep(int(step))
        sb.setValue(int(value))
    else:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(float(min_val), float(max_val))
        sb.setSingleStep(float(step))
        sb.setDecimals(decimals)
        sb.setValue(float(value))
    sb.setFixedWidth(90)
    return sb


def _param_row(*pairs) -> QtWidgets.QHBoxLayout:
    """Build a horizontal row of (QLabel, QWidget) pairs."""
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(6)
    for label_text, widget in pairs:
        lbl = QtWidgets.QLabel(label_text)
        lbl.setStyleSheet("font-weight: bold;")
        row.addWidget(lbl)
        row.addWidget(widget)
    row.addStretch()
    return row


# ---------------------------------------------------------------------------
# A) Spectrogram dialog  — pyqtgraph ImageItem + bilinear zoom for smooth rendering
# ---------------------------------------------------------------------------

class SpectrogramDialog(QtWidgets.QDialog):
    """
    Time-frequency spectrogram rendered with pyqtgraph ImageItem.

    Smooth appearance is achieved by upsampling the spectrogram array ×4 in
    both axes using bilinear interpolation (scipy.ndimage.zoom order=1),
    equivalent to matplotlib's shading='gouraud'.

    Navigation: QScrollBar to scroll through channels within the bbox.
    Parameters: NPERSEG, Overlap %, NFFT, Fmax, vMin/vMax → Apply.
    Dashed white lines mark t0/t1 of the bounding box.
    """

    _ZOOM = 4   # upsampling factor applied to both axes before display

    def __init__(self, ann, dataset, colormap: pg.ColorMap, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Spectrogram — Event [{ann.id}]  "
            f"t={ann.t0:.2f}–{ann.t1:.2f} s  "
            f"d={ann.d0:.0f}–{ann.d1:.0f} m"
        )
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.55))

        self.ann = ann
        self.dataset = dataset
        self.cmap = colormap
        self._cache = {} # (ch, nperseg, noverlap, nfft) → (f, t, Sxx_db)

        # Convert annotation global channel indices to local indices of ds.tr.
        # When stride > 1 the reader decimates channels but keeps the full
        # dist_m array, so ann.di0/di1 (global) must be mapped to the
        # decimated tr row index via the physical distances d0/d1.
        n_tr = dataset.tr.shape[0]
        dist_local = dataset.dist_m[:n_tr]
        di0 = int(np.clip(np.searchsorted(dist_local, ann.d0), 0, n_tr - 1))
        di1 = int(np.clip(np.searchsorted(dist_local, ann.d1), 0, n_tr - 1))
        if di1 <= di0:
            di1 = min(di0 + 1, n_tr - 1)
        self._di0 = di0
        self._di1 = di1
        self._cur_ch = (di0 + di1) // 2

        self._build_ui()
        self._plot(self._cur_ch, update_range=True)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Channel spinbox — created here so it is available to both the top
        # control bar and the scrollbar sync callbacks defined later.
        self.spin_channel = QtWidgets.QSpinBox()
        self.spin_channel.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.spin_channel.setValue(self._cur_ch)
        self.spin_channel.setFixedWidth(117)
        self.spin_channel.setAlignment(QtCore.Qt.AlignCenter)
        self.spin_channel.setToolTip("Jump to channel index")
        self.spin_channel.valueChanged.connect(self._on_channel_spinbox)

        # --- Parameter bar ---
        # nperseg=1024, overlap=75% → matches annotation_gui() quality
        self.sb_nperseg = _make_spinbox(1024, 16, 8192, 16)
        self.sb_overlap = _make_spinbox(75,   0,   99,  5)
        self.sb_nfft = _make_spinbox(1024, 16, 8192, 16)
        self.sb_fmax = _make_spinbox(
            self.dataset.fs_hz / 2, 1, self.dataset.fs_hz / 2, 1)
        self.sb_zoom = _make_spinbox(4,    1,   16,  1)
        self.sb_zoom.setToolTip(
            "Bilinear upsampling factor applied to both axes before display.\n"
            "Higher = smoother appearance, slower rendering.")
        self.sb_vmin = QtWidgets.QDoubleSpinBox()
        self.sb_vmin.setRange(-300, 300); self.sb_vmin.setDecimals(1)
        self.sb_vmin.setFixedWidth(80)
        self.sb_vmax = QtWidgets.QDoubleSpinBox()
        self.sb_vmax.setRange(-300, 300); self.sb_vmax.setDecimals(1)
        self.sb_vmax.setFixedWidth(80)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setFixedWidth(105)
        btn_apply.clicked.connect(self._on_apply)

        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setFixedWidth(90)
        btn_export.clicked.connect(self._export_png)

        param_row = _param_row(
            ("NPERSEG:", self.sb_nperseg),
            ("Overlap %:", self.sb_overlap),
            ("NFFT:", self.sb_nfft),
            ("Fmax [Hz]:", self.sb_fmax),
            ("Zoom:", self.sb_zoom),
            ("vMin:", self.sb_vmin),
            ("vMax:", self.sb_vmax),
        )
        lbl_ch_ctrl = QtWidgets.QLabel("Channel:")
        lbl_ch_ctrl.setStyleSheet("font-weight: bold;")
        param_row.addSpacing(8)
        param_row.addWidget(lbl_ch_ctrl)
        param_row.addWidget(self.spin_channel)
        param_row.addWidget(btn_apply)
        param_row.addWidget(btn_export)
        layout.addLayout(param_row)

        # --- Plot area + scrollbar ---
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setSpacing(4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time [s]")
        self.plot_widget.setLabel("left", "Frequency [Hz]")
        self.image_item = pg.ImageItem()
        self.image_item.setAutoDownsample(False)  # let zoom handle quality
        self.plot_widget.addItem(self.image_item)

        # Bbox time markers
        pen_dash = pg.mkPen(color=theme.current()["pg_line_avg"], width=1.5, style=QtCore.Qt.DashLine)
        self._vline0 = pg.InfiniteLine(pos=self.ann.t0, angle=90, pen=pen_dash)
        self._vline1 = pg.InfiniteLine(pos=self.ann.t1, angle=90, pen=pen_dash)
        self.plot_widget.addItem(self._vline0)
        self.plot_widget.addItem(self._vline1)

        # Histogram (level control) — same cmap as main waterfall
        self.histogram = pg.HistogramLUTWidget()
        self.histogram.setImageItem(self.image_item)
        self.histogram.gradient.setColorMap(self.cmap)
        self.histogram.setFixedWidth(110)

        # Scrollbar
        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.scrollbar.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.scrollbar.setValue(self._cur_ch)
        self.scrollbar.setInvertedAppearance(True)
        self.scrollbar.valueChanged.connect(self._on_scroll)


        scroll_col = QtWidgets.QVBoxLayout()
        scroll_col.setSpacing(2)
        scroll_col.addWidget(self.scrollbar, 1)

        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addLayout(scroll_col)
        plot_row.addWidget(self.histogram)
        layout.addLayout(plot_row, 1)

    def _plot(self, ch: int, update_range: bool = False):
        from scipy.ndimage import zoom as ndimage_zoom

        self._cur_ch = ch
        ds = self.dataset
        dist = float(ds.dist_m[ch]) if ch < len(ds.dist_m) else 0
        nperseg = self.sb_nperseg.value()
        overlap = self.sb_overlap.value()
        nfft = max(self.sb_nfft.value(), nperseg)
        noverlap = int(nperseg * overlap / 100)
        fmax = self.sb_fmax.value()

        key = (ch, nperseg, noverlap, nfft)
        if key not in self._cache:
            self._cache[key] = compute_spectrogram(
                ds.tr[ch, :], ds.fs_hz,
                nperseg=nperseg, noverlap=noverlap, nfft=nfft,
            )
        f, t, Sxx_db = self._cache[key]

        # Frequency mask
        fmask = f <= fmax
        f_shown = f[fmask]
        Sxx_show = Sxx_db[fmask, :]   # (n_freq, n_time)

        if update_range:
            vmin = float(np.min(Sxx_show))
            vmax = float(np.max(Sxx_show))
            self.sb_vmin.setValue(round(vmin, 1))
            self.sb_vmax.setValue(round(vmax, 1))
        else:
            vmin = self.sb_vmin.value()
            vmax = self.sb_vmax.value()

        # Bilinear upsampling — factor controlled by user (default 4)
        z = self.sb_zoom.value()
        Sxx_render = ndimage_zoom(Sxx_show, (z, z), order=1).astype(np.float32)
        n_freq_r, n_time_r = Sxx_render.shape

        # Physical coordinates of the rendered array
        t0_s = float(t[0] + ds.time_s[0])   # adjust to absolute time
        t1_s = float(t[-1] + ds.time_s[0])
        y0 = float(f_shown[0])
        y1 = float(f_shown[-1])
        dt_r = (t1_s - t0_s) / max(n_time_r - 1, 1)
        df_r = (y1 - y0) / max(n_freq_r - 1, 1)

        # --- Render ---
        self.image_item.sigImageChanged.disconnect(self.histogram.item.imageChanged)
        self.image_item.setImage(Sxx_render, autoLevels=False, levels=[vmin, vmax])
        self.image_item.setRect(QtCore.QRectF(t0_s, y0, t1_s - t0_s + dt_r, y1 - y0 + df_r))
        self.image_item.sigImageChanged.connect(self.histogram.item.imageChanged)

        self.histogram.item.imageChanged(autoLevel=False)
        self.histogram.setLevels(vmin, vmax)
        self.histogram.setHistogramRange(vmin - 5, vmax + 5, padding=0.05)

        self.plot_widget.setLimits(
          xMin=t0_s, xMax=t1_s + dt_r, yMin=y0, yMax=y1 + df_r)
        self.plot_widget.setRange(
          xRange=(t0_s, t1_s + dt_r), yRange=(y0, y1 + df_r), padding=0)

        self.setWindowTitle(
            f"Spectrogram — Event [{self.ann.id}]  "
            f"Channel {ch} ({dist:.0f} m)  "
            f"t={self.ann.t0:.2f}–{self.ann.t1:.2f} s"
        )

    def _on_scroll(self, val: int):
        # Sync spinbox without re-triggering _on_channel_spinbox
        self.spin_channel.blockSignals(True)
        self.spin_channel.setValue(val)
        self.spin_channel.blockSignals(False)
        self._plot(val, update_range=False)

    def _on_channel_spinbox(self, val: int):
        """Jump directly to a channel by typing its index."""
        val = max(self._di0, min(val, self._di1 - 1))
        self.scrollbar.setValue(val)  # triggers _on_scroll

    def _on_apply(self):
        self._cache.clear()
        self._plot(self._cur_ch, update_range=False)

    def _export_png(self) -> None:
        """Export the spectrogram plot as a high-quality PNG at 300 DPI."""
        default_name = (
            f"spectrogram_{self.ann.id}_"
            f"t{self.ann.t0:.1f}-{self.ann.t1:.1f}s"
            f"_ch{self._cur_ch}.png"
        ).replace(" ", "_")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name,
            "PNG images (*.png)"
        )
        if not path:
            return

        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi
        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width()  * scale),
            int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        img_out = scaled.toImage()
        dpm = int(300 / 0.0254)
        img_out.setDotsPerMeterX(dpm)
        img_out.setDotsPerMeterY(dpm)

        if img_out.save(path, "PNG"):
            QtWidgets.QMessageBox.information(
                self, "Export successful", f"Saved 300 DPI PNG:\n{path}"
            )
        else:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not save:\n{path}"
            )



# ---------------------------------------------------------------------------
# B) Spectral analysis dialog
# ---------------------------------------------------------------------------

class SpectralDialog(QtWidgets.QDialog):
    """
    Magnitude spectra of up to 30 channels within the annotation bbox.

    Individual spectra: thin translucent lines in Tab10 colours.
    Average spectrum: solid white thicker line.
    Parameters: NFFT, Window, Fmax, Y-scale (default Log) → Apply.
    """

    MAX_SPECTRUMS = 30

    def __init__(self, ann, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Spectral Analysis  —  Event: [{ann.id}]  "
            f"t={ann.t0:.2f}–{ann.t1:.2f}s"
        )
        # ~50% of screen, with native minimize/maximize buttons
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.55))

        self.ann = ann
        self.dataset = dataset

        di0, di1 = ann.di0, ann.di1
        if di1 <= di0:
            di1 = di0 + 1
        self._channels = select_channels_for_spectral(di0, di1, self.MAX_SPECTRUMS)
        self._ti0 = ann.ti0
        self._ti1 = ann.ti1 if ann.ti1 > ann.ti0 else ann.ti0 + 1

        n_shown = len(select_channels_for_spectral(di0, di1, self.MAX_SPECTRUMS))
        n_total = di1 - di0
        self.setWindowTitle(
            f"Spectral Analysis — Event [{ann.id}]  "
            f"Channels {di0}–{di1}  ({n_shown}/{n_total})  "
            f"Time {ann.t0:.2f}–{ann.t1:.2f} s"
        )

        self._build_ui()
        self._plot()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        self.sb_nfft = _make_spinbox(2048, 16, 65536, 256)
        self.combo_win = QtWidgets.QComboBox()
        self.combo_win.addItems(['hann', 'hamming', 'blackman', 'bartlett', 'none'])
        self.combo_win.setFixedWidth(100)
        self.sb_fmax = _make_spinbox(
            self.dataset.fs_hz / 2, 1, self.dataset.fs_hz / 2, 1)
        self.combo_scale = QtWidgets.QComboBox()
        self.combo_scale.addItems(['Log', 'Linear'])   # Log first = default
        self.combo_scale.setFixedWidth(96)
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setFixedWidth(105)
        btn_apply.clicked.connect(self._plot)

        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setFixedWidth(90)
        btn_export.clicked.connect(self._export_png)

        param_row = _param_row(
            ("NFFT:", self.sb_nfft),
            ("Window:", self.combo_win),
            ("Fmax [Hz]:", self.sb_fmax),
            ("Y-scale:", self.combo_scale),
        )
        param_row.addWidget(btn_apply)
        param_row.addWidget(btn_export)
        layout.addLayout(param_row)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Frequency [Hz]")
        self.plot_widget.setLabel("left", "Magnitude [a.u.]")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget, 1)

    def _plot(self):
        ds = self.dataset
        nfft = self.sb_nfft.value()
        win = self.combo_win.currentText()
        fmax = self.sb_fmax.value()
        log = self.combo_scale.currentText() == 'Log'

        self.plot_widget.clear()

        ti0, ti1 = self._ti0, self._ti1
        n_ch = len(self._channels)

        spectra = []
        for i, ch in enumerate(self._channels):
            sig = ds.tr[ch, ti0:ti1]
            freqs, mag = compute_spectrum(sig, ds.fs_hz, nfft=nfft, window=win)
            spectra.append(mag)

            color = _TAB10[i % len(_TAB10)]
            pen = pg.mkPen(color=(*color, 100), width=1)
            self.plot_widget.plot(freqs, mag, pen=pen)

        if spectra:
            avg = np.mean(np.array(spectra), axis=0)
            # Solid emphasis line so it stands out against the individual
            # per-channel spectra: white on dark background, black on light.
            pen_avg = pg.mkPen(color=theme.current()["pg_line_avg"], width=3)
            self.plot_widget.plot(freqs, avg, pen=pen_avg, name="Average")

        # Fmax clip
        self.plot_widget.setXRange(0, fmax, padding=0)

        if log:
            self.plot_widget.setLogMode(y=True)
        else:
            self.plot_widget.setLogMode(y=False)

    def _export_png(self) -> None:
        """Export the plot as a high-quality PNG at 300 DPI."""
        default_name = (
            f"spectral_{self.ann.id}_"
            f"t{self.ann.t0:.1f}-{self.ann.t1:.1f}s.png"
        ).replace(" ", "_")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name,
            "PNG images (*.png)"
        )
        if not path:
            return

        # Grab the plot widget at screen resolution, then scale up to 300 DPI.
        # Assuming 96 DPI screen → 300/96 ≈ 3.125×
        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi

        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width()  * scale),
            int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )

        # Embed 300 DPI metadata (dots per metre = dpi / 0.0254)
        img = scaled.toImage()
        dpm = int(300 / 0.0254)
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)

        if img.save(path, "PNG"):
            QtWidgets.QMessageBox.information(
                self, "Export successful",
                f"Saved 300 DPI PNG:\n{path}"
            )
        else:
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Could not save file:\n{path}"
            )


# ---------------------------------------------------------------------------
# C) Time-domain signal dialog
# ---------------------------------------------------------------------------

class SignalDialog(QtWidgets.QDialog):
    """
    Time-domain waveform for channels within the annotation bbox.

    Navigation: QScrollBar to scroll through channels.
    Fix Channel: pins the current channel in a Tab10 colour, drawn on top
                 of any subsequent active channel so comparisons are easy.
    Clear Fixed: removes all pinned channels.
    Vertical dashed red lines mark t0/t1 of the bounding box.
    yMin/yMax + Apply + Export at the right of the control bar.
    """

    def __init__(self, ann, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Signal (time domain) — Event [{ann.id}]  "
            f"d={ann.d0:.0f}–{ann.d1:.0f} m  "
            f"t={ann.t0:.2f}–{ann.t1:.2f} s"
        )
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.45))

        self.ann     = ann
        self.dataset = dataset

        n_tr = dataset.tr.shape[0]
        dist_local = dataset.dist_m[:n_tr]
        di0 = int(np.clip(np.searchsorted(dist_local, ann.d0), 0, n_tr - 1))
        di1 = int(np.clip(np.searchsorted(dist_local, ann.d1), 0, n_tr - 1))
        if di1 <= di0:
            di1 = min(di0 + 1, n_tr - 1)
        self._di0     = di0
        self._di1     = di1
        self._cur_ch  = (di0 + di1) // 2
        # Fixed channels: list of (channel_index, Tab10_color)
        self._fixed_chs: list = []

        self._build_ui()
        self._plot(self._cur_ch)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Channel spinbox — created here so it is available to both the top
        # control bar and the scrollbar sync callbacks defined later.
        self.spin_channel = QtWidgets.QSpinBox()
        self.spin_channel.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.spin_channel.setValue(self._cur_ch)
        self.spin_channel.setFixedWidth(117)
        self.spin_channel.setAlignment(QtCore.Qt.AlignCenter)
        self.spin_channel.setToolTip("Jump to channel index")
        self.spin_channel.valueChanged.connect(self._on_channel_spinbox)

        # --- Control bar ---
        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(8)

        # yMin / yMax
        self.sb_ymin = QtWidgets.QDoubleSpinBox()
        self.sb_ymin.setRange(-1e9, 1e9); self.sb_ymin.setDecimals(4)
        self.sb_ymin.setMinimumWidth(110)
        self.sb_ymax = QtWidgets.QDoubleSpinBox()
        self.sb_ymax.setRange(-1e9, 1e9); self.sb_ymax.setDecimals(4)
        self.sb_ymax.setMinimumWidth(110)

        for lbl_text, sb in [("yMin:", self.sb_ymin), ("yMax:", self.sb_ymax)]:
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet("font-weight: bold;")
            ctrl_row.addWidget(lbl)
            ctrl_row.addWidget(sb)

        ctrl_row.addSpacing(8)

        # Fix Channel / Clear Fixed
        self.btn_fix = QtWidgets.QPushButton("Fix Channel")
        self.btn_fix.setMinimumWidth(105)
        self.btn_fix.clicked.connect(self._on_fix)
        self.btn_clear = QtWidgets.QPushButton("Clear Fixed")
        self.btn_clear.setMinimumWidth(105)
        self.btn_clear.clicked.connect(self._on_clear)
        ctrl_row.addWidget(self.btn_fix)
        ctrl_row.addWidget(self.btn_clear)

        # Info label
        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        ctrl_row.addWidget(self.lbl_info)

        lbl_ch_ctrl = QtWidgets.QLabel("Channel:")
        lbl_ch_ctrl.setStyleSheet("font-weight: bold;")
        ctrl_row.addWidget(lbl_ch_ctrl)
        ctrl_row.addWidget(self.spin_channel)

        ctrl_row.addStretch()

        # Apply / Export at the right
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setMinimumWidth(105)
        btn_apply.clicked.connect(self._on_apply_y)
        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setMinimumWidth(90)
        btn_export.clicked.connect(self._export_png)
        ctrl_row.addWidget(btn_apply)
        ctrl_row.addWidget(btn_export)

        layout.addLayout(ctrl_row)

        # --- Plot area + scrollbar ---
        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setSpacing(4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time [s]")
        self.plot_widget.setLabel("left",   "Amplitude [a.u.]")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Bbox time markers (red dashed)
        pen_box = pg.mkPen(color=(220, 50, 50), width=1.5,
                            style=QtCore.Qt.DashLine)
        self._vline0 = pg.InfiniteLine(pos=self.ann.t0, angle=90, pen=pen_box)
        self._vline1 = pg.InfiniteLine(pos=self.ann.t1, angle=90, pen=pen_box)
        self.plot_widget.addItem(self._vline0)
        self.plot_widget.addItem(self._vline1)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.scrollbar.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.scrollbar.setValue(self._cur_ch)
        self.scrollbar.setInvertedAppearance(True)
        self.scrollbar.valueChanged.connect(self._on_scroll)


        scroll_col = QtWidgets.QVBoxLayout()
        scroll_col.setSpacing(2)
        scroll_col.addWidget(self.scrollbar, 1)

        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addLayout(scroll_col)
        layout.addLayout(plot_row, 1)

    def _plot(self, ch: int):
        self._cur_ch = ch
        ds   = self.dataset
        dist = float(ds.dist_m[ch]) if ch < len(ds.dist_m) else 0
        time = ds.time_s

        # Remove all data curves (keep InfiniteLines)
        for item in list(self.plot_widget.listDataItems()):
            self.plot_widget.removeItem(item)

        # Active channel first (drawn below fixed channels)
        pen_active = pg.mkPen(color=theme.current()["pg_line_main"], width=1.5)
        self.plot_widget.plot(time, ds.tr[ch, :], pen=pen_active)

        # Fixed channels drawn ON TOP with their assigned Tab10 colour
        # (fully opaque, slightly thicker — they are the reference channels)
        for fch, color in self._fixed_chs:
            pen = pg.mkPen(color=(*color, 255), width=2.0)
            self.plot_widget.plot(time, ds.tr[fch, :], pen=pen)

        # Y range
        sig       = ds.tr[ch, :]
        amp_range = sig.max() - sig.min()
        margin    = amp_range * 0.05 if amp_range > 0 else 1e-12
        auto_min  = float(sig.min() - margin)
        auto_max  = float(sig.max() + margin)

        ymin = self.sb_ymin.value()
        ymax = self.sb_ymax.value()
        if ymin == 0.0 and ymax == 0.0:   # uninitialised
            self.sb_ymin.setValue(auto_min)
            self.sb_ymax.setValue(auto_max)
            ymin, ymax = auto_min, auto_max

        self.plot_widget.setYRange(ymin, ymax, padding=0)

        fixed_info = (
            "  |  Fixed: " + ", ".join(
                f"CH{fch}" for fch, _ in self._fixed_chs
            ) if self._fixed_chs else ""
        )
        self.lbl_info.setText(f"Channel: {ch}  ({dist:.0f} m){fixed_info}")
        self.setWindowTitle(
            f"Signal (time domain) — Event [{self.ann.id}]  "
            f"Channel {ch}  ({dist:.0f} m)"
        )

    def _on_scroll(self, val: int):
        self.spin_channel.blockSignals(True)
        self.spin_channel.setValue(val)
        self.spin_channel.blockSignals(False)
        self._plot(val)

    def _on_channel_spinbox(self, val: int):
        """Jump directly to a channel by typing its index."""
        val = max(self._di0, min(val, self._di1 - 1))
        self.scrollbar.setValue(val)

    def _on_apply_y(self):
        self.plot_widget.setYRange(
            self.sb_ymin.value(), self.sb_ymax.value(), padding=0
        )

    def _on_fix(self):
        ch = self._cur_ch
        # Only add if not already fixed
        if not any(fch == ch for fch, _ in self._fixed_chs):
            color = _TAB10[len(self._fixed_chs) % len(_TAB10)]
            self._fixed_chs.append((ch, color))
        self._plot(ch)

    def _on_clear(self):
        self._fixed_chs.clear()
        self._plot(self._cur_ch)

    def _export_png(self) -> None:
        """Export the signal plot as a high-quality PNG at 300 DPI."""
        default_name = (
            f"signal_{self.ann.id}_"
            f"ch{self._cur_ch}.png"
        ).replace(" ", "_")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name,
            "PNG images (*.png)"
        )
        if not path:
            return

        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi
        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width()  * scale),
            int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        img = scaled.toImage()
        dpm = int(300 / 0.0254)
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)

        if img.save(path, "PNG"):
            QtWidgets.QMessageBox.information(
                self, "Export successful", f"Saved 300 DPI PNG:\n{path}"
            )
        else:
            QtWidgets.QMessageBox.critical(
                self, "Export failed", f"Could not save:\n{path}"
            )


# ---------------------------------------------------------------------------
# D) Signal (frequency domain) - FFT magnitude spectrum
# ---------------------------------------------------------------------------

_FFT_WINDOWS = ["Hann", "Hamming", "Blackman", "Rectangular"]


class SignalFreqDialog(QtWidgets.QDialog):
    """
    FFT magnitude spectrum of a single channel within the annotation bbox.

    Same navigation/fix-channel pattern as SignalDialog:
      - QScrollBar to scroll through channels
      - Fix Channel / Clear Fixed to overlay reference channels
      - yMin/yMax + Apply + Export
    Adds FFT-specific controls: window function and magnitude scale (dB/linear).
    """

    def __init__(self, ann, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Signal (frequency domain) - Event [{ann.id}]  "
            f"d={ann.d0:.0f}-{ann.d1:.0f} m  t={ann.t0:.2f}-{ann.t1:.2f} s"
        )
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.45))

        self.ann     = ann
        self.dataset = dataset

        n_tr = dataset.tr.shape[0]
        dist_local = dataset.dist_m[:n_tr]
        di0 = int(np.clip(np.searchsorted(dist_local, ann.d0), 0, n_tr - 1))
        di1 = int(np.clip(np.searchsorted(dist_local, ann.d1), 0, n_tr - 1))
        if di1 <= di0:
            di1 = min(di0 + 1, n_tr - 1)
        self._di0     = di0
        self._di1     = di1
        self._cur_ch  = (di0 + di1) // 2
        self._fixed_chs: list = []
        self._last_scale: str = "dB (log)"

        self._build_ui()
        self._plot(self._cur_ch)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Channel spinbox — created here so it is available to both the top
        # control bar and the scrollbar sync callbacks defined later.
        self.spin_channel = QtWidgets.QSpinBox()
        self.spin_channel.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.spin_channel.setValue(self._cur_ch)
        self.spin_channel.setFixedWidth(117)
        self.spin_channel.setAlignment(QtCore.Qt.AlignCenter)
        self.spin_channel.setToolTip("Jump to channel index")
        self.spin_channel.valueChanged.connect(self._on_channel_spinbox)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(8)

        lbl_win = QtWidgets.QLabel("Window:")
        lbl_win.setStyleSheet("font-weight: bold;")
        ctrl_row.addWidget(lbl_win)
        self.combo_window = QtWidgets.QComboBox()
        self.combo_window.addItems(_FFT_WINDOWS)
        self.combo_window.setCurrentText("Hann")
        self.combo_window.setFixedWidth(100)
        self.combo_window.currentIndexChanged.connect(lambda _: self._plot(self._cur_ch))
        ctrl_row.addWidget(self.combo_window)

        ctrl_row.addSpacing(8)

        lbl_scale = QtWidgets.QLabel("Scale:")
        lbl_scale.setStyleSheet("font-weight: bold;")
        ctrl_row.addWidget(lbl_scale)
        self.combo_scale = QtWidgets.QComboBox()
        self.combo_scale.addItems(["dB (log)", "Linear"])
        self.combo_scale.setFixedWidth(135)
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        ctrl_row.addWidget(self.combo_scale)

        ctrl_row.addSpacing(8)

        self.sb_ymin = QtWidgets.QDoubleSpinBox()
        self.sb_ymin.setRange(-1e9, 1e9); self.sb_ymin.setDecimals(4)
        self.sb_ymin.setMinimumWidth(110)
        self.sb_ymax = QtWidgets.QDoubleSpinBox()
        self.sb_ymax.setRange(-1e9, 1e9); self.sb_ymax.setDecimals(4)
        self.sb_ymax.setMinimumWidth(110)
        for lbl_text, sb in [("yMin:", self.sb_ymin), ("yMax:", self.sb_ymax)]:
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet("font-weight: bold;")
            ctrl_row.addWidget(lbl)
            ctrl_row.addWidget(sb)

        ctrl_row.addSpacing(8)

        self.btn_fix = QtWidgets.QPushButton("Fix Channel")
        self.btn_fix.setMinimumWidth(105)
        self.btn_fix.clicked.connect(self._on_fix)
        self.btn_clear = QtWidgets.QPushButton("Clear Fixed")
        self.btn_clear.setMinimumWidth(105)
        self.btn_clear.clicked.connect(self._on_clear)
        ctrl_row.addWidget(self.btn_fix)
        ctrl_row.addWidget(self.btn_clear)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        ctrl_row.addWidget(self.lbl_info)

        lbl_ch_ctrl = QtWidgets.QLabel("Channel:")
        lbl_ch_ctrl.setStyleSheet("font-weight: bold;")
        ctrl_row.addWidget(lbl_ch_ctrl)
        ctrl_row.addWidget(self.spin_channel)

        ctrl_row.addStretch()

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setMinimumWidth(105)
        btn_apply.clicked.connect(self._on_apply_y)
        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setMinimumWidth(90)
        btn_export.clicked.connect(self._export_png)
        ctrl_row.addWidget(btn_apply)
        ctrl_row.addWidget(btn_export)

        layout.addLayout(ctrl_row)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setSpacing(4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Frequency [Hz]")
        self.plot_widget.setLabel("left",   "Magnitude")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.scrollbar.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.scrollbar.setValue(self._cur_ch)
        self.scrollbar.setInvertedAppearance(True)
        self.scrollbar.valueChanged.connect(self._on_scroll)


        scroll_col = QtWidgets.QVBoxLayout()
        scroll_col.setSpacing(2)
        scroll_col.addWidget(self.scrollbar, 1)

        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addLayout(scroll_col)
        layout.addLayout(plot_row, 1)

    def _window_array(self, n: int) -> np.ndarray:
        name = self.combo_window.currentText()
        if name == "Hann":
            return np.hanning(n)
        elif name == "Hamming":
            return np.hamming(n)
        elif name == "Blackman":
            return np.blackman(n)
        else:
            return np.ones(n)

    def _compute_fft(self, ch: int):
        ds  = self.dataset
        sig = ds.tr[ch, :].astype(np.float64)
        n   = len(sig)
        win = self._window_array(n)
        sig_w = sig * win

        spec = np.fft.rfft(sig_w)
        freqs = np.fft.rfftfreq(n, d=1.0 / ds.fs_hz)
        mag = np.abs(spec)

        if self.combo_scale.currentText().startswith("dB"):
            mag = 20.0 * np.log10(np.maximum(mag, 1e-12))

        return freqs, mag

    def _plot(self, ch: int):
        self._cur_ch = ch
        ds = self.dataset
        dist = float(ds.dist_m[ch]) if ch < len(ds.dist_m) else 0

        for item in list(self.plot_widget.listDataItems()):
            self.plot_widget.removeItem(item)

        pen_active = pg.mkPen(color=theme.current()["pg_line_main"], width=1.5)
        freqs, mag = self._compute_fft(ch)
        self.plot_widget.plot(freqs, mag, pen=pen_active)

        for fch, color in self._fixed_chs:
            f_fch, m_fch = self._compute_fft(fch)
            pen = pg.mkPen(color=(*color, 255), width=2.0)
            self.plot_widget.plot(f_fch, m_fch, pen=pen)

        cur_scale = self.combo_scale.currentText()
        scale_label = "Magnitude [dB]" if cur_scale.startswith("dB") else "Magnitude"
        self.plot_widget.setLabel("left", scale_label)

        # Collect all currently-plotted magnitudes (active + fixed channels)
        # so the auto-range and 0-baseline (Linear) cover everything shown.
        all_mags = [mag] + [self._compute_fft(fch)[1] for fch, _ in self._fixed_chs]
        mag_min = min(float(m.min()) for m in all_mags)
        mag_max = max(float(m.max()) for m in all_mags)
        amp_range = mag_max - mag_min
        margin = amp_range * 0.05 if amp_range > 0 else 1e-12

        if cur_scale.startswith("dB"):
            auto_min = mag_min - margin
            auto_max = mag_max + margin
        else:
            # Linear magnitude is always >= 0 — keep 0 pinned at the bottom
            # so the plot isn't clipped when switching from dB to Linear.
            auto_min = 0.0
            auto_max = mag_max + margin

        # Force a fresh auto-range whenever the scale (dB <-> Linear) changes,
        # since the two have completely different numeric ranges.
        scale_changed = cur_scale != self._last_scale
        self._last_scale = cur_scale

        ymin = self.sb_ymin.value()
        ymax = self.sb_ymax.value()
        if scale_changed or (ymin == 0.0 and ymax == 0.0):
            self.sb_ymin.setValue(auto_min)
            self.sb_ymax.setValue(auto_max)
            ymin, ymax = auto_min, auto_max
        self.plot_widget.setYRange(ymin, ymax, padding=0)

        fixed_info = (
            "  |  Fixed: " + ", ".join(f"CH{fch}" for fch, _ in self._fixed_chs)
            if self._fixed_chs else ""
        )
        self.lbl_info.setText(f"Channel: {ch}  ({dist:.0f} m){fixed_info}")
        self.setWindowTitle(
            f"Signal (frequency domain) - Event [{self.ann.id}]  "
            f"Channel {ch}  ({dist:.0f} m)"
        )

    def _on_scroll(self, val: int):
        self.spin_channel.blockSignals(True)
        self.spin_channel.setValue(val)
        self.spin_channel.blockSignals(False)
        self._plot(val)

    def _on_channel_spinbox(self, val: int):
        """Jump directly to a channel by typing its index."""
        val = max(self._di0, min(val, self._di1 - 1))
        self.scrollbar.setValue(val)

    def _on_scale_changed(self, _index: int):
        self._plot(self._cur_ch)

    def _on_apply_y(self):
        self.plot_widget.setYRange(self.sb_ymin.value(), self.sb_ymax.value(), padding=0)

    def _on_fix(self):
        ch = self._cur_ch
        if not any(fch == ch for fch, _ in self._fixed_chs):
            color = _TAB10[len(self._fixed_chs) % len(_TAB10)]
            self._fixed_chs.append((ch, color))
        self._plot(ch)

    def _on_clear(self):
        self._fixed_chs.clear()
        self._plot(self._cur_ch)

    def _export_png(self) -> None:
        default_name = f"signal_freq_{self.ann.id}_ch{self._cur_ch}.png".replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name, "PNG images (*.png)"
        )
        if not path:
            return
        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi
        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width() * scale), int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
        )
        img = scaled.toImage()
        dpm = int(300 / 0.0254)
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if img.save(path, "PNG"):
            QtWidgets.QMessageBox.information(self, "Export successful", f"Saved 300 DPI PNG:\n{path}")
        else:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not save:\n{path}")


# ---------------------------------------------------------------------------
# E) Signal (envelope) - Hilbert envelope amplitude
# ---------------------------------------------------------------------------

class SignalEnvelopeDialog(QtWidgets.QDialog):
    """
    Hilbert envelope |hilbert(signal)| of a single channel within the bbox.
    Same navigation/fix-channel pattern as SignalDialog.
    """

    def __init__(self, ann, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Signal (envelope) - Event [{ann.id}]  "
            f"d={ann.d0:.0f}-{ann.d1:.0f} m  t={ann.t0:.2f}-{ann.t1:.2f} s"
        )
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.45))

        self.ann     = ann
        self.dataset = dataset

        n_tr = dataset.tr.shape[0]
        dist_local = dataset.dist_m[:n_tr]
        di0 = int(np.clip(np.searchsorted(dist_local, ann.d0), 0, n_tr - 1))
        di1 = int(np.clip(np.searchsorted(dist_local, ann.d1), 0, n_tr - 1))
        if di1 <= di0:
            di1 = min(di0 + 1, n_tr - 1)
        self._di0     = di0
        self._di1     = di1
        self._cur_ch  = (di0 + di1) // 2
        self._fixed_chs: list = []

        self._build_ui()
        self._plot(self._cur_ch)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Channel spinbox — created here so it is available to both the top
        # control bar and the scrollbar sync callbacks defined later.
        self.spin_channel = QtWidgets.QSpinBox()
        self.spin_channel.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.spin_channel.setValue(self._cur_ch)
        self.spin_channel.setFixedWidth(117)
        self.spin_channel.setAlignment(QtCore.Qt.AlignCenter)
        self.spin_channel.setToolTip("Jump to channel index")
        self.spin_channel.valueChanged.connect(self._on_channel_spinbox)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.sb_ymin = QtWidgets.QDoubleSpinBox()
        self.sb_ymin.setRange(-1e9, 1e9); self.sb_ymin.setDecimals(4)
        self.sb_ymin.setMinimumWidth(110)
        self.sb_ymax = QtWidgets.QDoubleSpinBox()
        self.sb_ymax.setRange(-1e9, 1e9); self.sb_ymax.setDecimals(4)
        self.sb_ymax.setMinimumWidth(110)
        for lbl_text, sb in [("yMin:", self.sb_ymin), ("yMax:", self.sb_ymax)]:
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet("font-weight: bold;")
            ctrl_row.addWidget(lbl)
            ctrl_row.addWidget(sb)

        ctrl_row.addSpacing(8)

        self.btn_fix = QtWidgets.QPushButton("Fix Channel")
        self.btn_fix.setMinimumWidth(105)
        self.btn_fix.clicked.connect(self._on_fix)
        self.btn_clear = QtWidgets.QPushButton("Clear Fixed")
        self.btn_clear.setMinimumWidth(105)
        self.btn_clear.clicked.connect(self._on_clear)
        ctrl_row.addWidget(self.btn_fix)
        ctrl_row.addWidget(self.btn_clear)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        ctrl_row.addWidget(self.lbl_info)

        lbl_ch_ctrl = QtWidgets.QLabel("Channel:")
        lbl_ch_ctrl.setStyleSheet("font-weight: bold;")
        ctrl_row.addWidget(lbl_ch_ctrl)
        ctrl_row.addWidget(self.spin_channel)

        ctrl_row.addStretch()

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setMinimumWidth(105)
        btn_apply.clicked.connect(self._on_apply_y)
        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setMinimumWidth(90)
        btn_export.clicked.connect(self._export_png)
        ctrl_row.addWidget(btn_apply)
        ctrl_row.addWidget(btn_export)

        layout.addLayout(ctrl_row)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setSpacing(4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time [s]")
        self.plot_widget.setLabel("left",   "Envelope amplitude [a.u.]")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        pen_box = pg.mkPen(color=(220, 50, 50), width=1.5, style=QtCore.Qt.DashLine)
        self._vline0 = pg.InfiniteLine(pos=self.ann.t0, angle=90, pen=pen_box)
        self._vline1 = pg.InfiniteLine(pos=self.ann.t1, angle=90, pen=pen_box)
        self.plot_widget.addItem(self._vline0)
        self.plot_widget.addItem(self._vline1)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.scrollbar.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.scrollbar.setValue(self._cur_ch)
        self.scrollbar.setInvertedAppearance(True)
        self.scrollbar.valueChanged.connect(self._on_scroll)


        scroll_col = QtWidgets.QVBoxLayout()
        scroll_col.setSpacing(2)
        scroll_col.addWidget(self.scrollbar, 1)

        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addLayout(scroll_col)
        layout.addLayout(plot_row, 1)

    def _compute_envelope(self, ch: int) -> np.ndarray:
        import scipy.signal as sp
        sig = self.dataset.tr[ch, :].astype(np.float64)
        return np.abs(sp.hilbert(sig))

    def _plot(self, ch: int):
        self._cur_ch = ch
        ds   = self.dataset
        dist = float(ds.dist_m[ch]) if ch < len(ds.dist_m) else 0
        time = ds.time_s

        for item in list(self.plot_widget.listDataItems()):
            self.plot_widget.removeItem(item)

        pen_active = pg.mkPen(color=theme.current()["pg_line_main"], width=1.5)
        env = self._compute_envelope(ch)
        self.plot_widget.plot(time, env, pen=pen_active)

        for fch, color in self._fixed_chs:
            env_fch = self._compute_envelope(fch)
            pen = pg.mkPen(color=(*color, 255), width=2.0)
            self.plot_widget.plot(time, env_fch, pen=pen)

        amp_range = env.max() - env.min()
        margin = amp_range * 0.05 if amp_range > 0 else 1e-12
        auto_min = float(max(env.min() - margin, 0.0))
        auto_max = float(env.max() + margin)

        ymin = self.sb_ymin.value()
        ymax = self.sb_ymax.value()
        if ymin == 0.0 and ymax == 0.0:
            self.sb_ymin.setValue(auto_min)
            self.sb_ymax.setValue(auto_max)
            ymin, ymax = auto_min, auto_max
        self.plot_widget.setYRange(ymin, ymax, padding=0)

        fixed_info = (
            "  |  Fixed: " + ", ".join(f"CH{fch}" for fch, _ in self._fixed_chs)
            if self._fixed_chs else ""
        )
        self.lbl_info.setText(f"Channel: {ch}  ({dist:.0f} m){fixed_info}")
        self.setWindowTitle(
            f"Signal (envelope) - Event [{self.ann.id}]  "
            f"Channel {ch}  ({dist:.0f} m)"
        )

    def _on_scroll(self, val: int):
        self.spin_channel.blockSignals(True)
        self.spin_channel.setValue(val)
        self.spin_channel.blockSignals(False)
        self._plot(val)

    def _on_channel_spinbox(self, val: int):
        """Jump directly to a channel by typing its index."""
        val = max(self._di0, min(val, self._di1 - 1))
        self.scrollbar.setValue(val)

    def _on_apply_y(self):
        self.plot_widget.setYRange(self.sb_ymin.value(), self.sb_ymax.value(), padding=0)

    def _on_fix(self):
        ch = self._cur_ch
        if not any(fch == ch for fch, _ in self._fixed_chs):
            color = _TAB10[len(self._fixed_chs) % len(_TAB10)]
            self._fixed_chs.append((ch, color))
        self._plot(ch)

    def _on_clear(self):
        self._fixed_chs.clear()
        self._plot(self._cur_ch)

    def _export_png(self) -> None:
        default_name = f"signal_envelope_{self.ann.id}_ch{self._cur_ch}.png".replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name, "PNG images (*.png)"
        )
        if not path:
            return
        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi
        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width() * scale), int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
        )
        img = scaled.toImage()
        dpm = int(300 / 0.0254)
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if img.save(path, "PNG"):
            QtWidgets.QMessageBox.information(self, "Export successful", f"Saved 300 DPI PNG:\n{path}")
        else:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not save:\n{path}")


# ---------------------------------------------------------------------------
# F) Signal (phase) - unwrapped instantaneous phase
# ---------------------------------------------------------------------------

class SignalPhaseDialog(QtWidgets.QDialog):
    """
    Unwrapped instantaneous phase, np.unwrap(np.angle(hilbert(signal))), of
    a single channel within the bbox. Unwrapped (not wrapped -pi..pi) phase
    is used because a wrapped phase plot of an oscillating 10-30 Hz signal
    is dominated by sawtooth jumps and carries no readable information;
    the unwrapped phase grows smoothly for a stable tone and any deviation
    from that linear trend (curvature, step) directly reveals frequency
    changes or phase discontinuities relevant to source tracking.
    Same navigation/fix-channel pattern as SignalDialog.
    """

    def __init__(self, ann, dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Signal (phase) - Event [{ann.id}]  "
            f"d={ann.d0:.0f}-{ann.d1:.0f} m  t={ann.t0:.2f}-{ann.t1:.2f} s"
        )
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.55), int(screen.height() * 0.45))

        self.ann     = ann
        self.dataset = dataset

        n_tr = dataset.tr.shape[0]
        dist_local = dataset.dist_m[:n_tr]
        di0 = int(np.clip(np.searchsorted(dist_local, ann.d0), 0, n_tr - 1))
        di1 = int(np.clip(np.searchsorted(dist_local, ann.d1), 0, n_tr - 1))
        if di1 <= di0:
            di1 = min(di0 + 1, n_tr - 1)
        self._di0     = di0
        self._di1     = di1
        self._cur_ch  = (di0 + di1) // 2
        self._fixed_chs: list = []

        self._build_ui()
        self._plot(self._cur_ch)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # Channel spinbox — created here so it is available to both the top
        # control bar and the scrollbar sync callbacks defined later.
        self.spin_channel = QtWidgets.QSpinBox()
        self.spin_channel.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.spin_channel.setValue(self._cur_ch)
        self.spin_channel.setFixedWidth(117)
        self.spin_channel.setAlignment(QtCore.Qt.AlignCenter)
        self.spin_channel.setToolTip("Jump to channel index")
        self.spin_channel.valueChanged.connect(self._on_channel_spinbox)

        ctrl_row = QtWidgets.QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.sb_ymin = QtWidgets.QDoubleSpinBox()
        self.sb_ymin.setRange(-1e9, 1e9); self.sb_ymin.setDecimals(4)
        self.sb_ymin.setMinimumWidth(110)
        self.sb_ymax = QtWidgets.QDoubleSpinBox()
        self.sb_ymax.setRange(-1e9, 1e9); self.sb_ymax.setDecimals(4)
        self.sb_ymax.setMinimumWidth(110)
        for lbl_text, sb in [("yMin:", self.sb_ymin), ("yMax:", self.sb_ymax)]:
            lbl = QtWidgets.QLabel(lbl_text)
            lbl.setStyleSheet("font-weight: bold;")
            ctrl_row.addWidget(lbl)
            ctrl_row.addWidget(sb)

        ctrl_row.addSpacing(8)

        self.btn_fix = QtWidgets.QPushButton("Fix Channel")
        self.btn_fix.setMinimumWidth(105)
        self.btn_fix.clicked.connect(self._on_fix)
        self.btn_clear = QtWidgets.QPushButton("Clear Fixed")
        self.btn_clear.setMinimumWidth(105)
        self.btn_clear.clicked.connect(self._on_clear)
        ctrl_row.addWidget(self.btn_fix)
        ctrl_row.addWidget(self.btn_clear)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 8pt;")
        ctrl_row.addWidget(self.lbl_info)

        lbl_ch_ctrl = QtWidgets.QLabel("Channel:")
        lbl_ch_ctrl.setStyleSheet("font-weight: bold;")
        ctrl_row.addWidget(lbl_ch_ctrl)
        ctrl_row.addWidget(self.spin_channel)

        ctrl_row.addStretch()

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setMinimumWidth(105)
        btn_apply.clicked.connect(self._on_apply_y)
        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setMinimumWidth(90)
        btn_export.clicked.connect(self._export_png)
        ctrl_row.addWidget(btn_apply)
        ctrl_row.addWidget(btn_export)

        layout.addLayout(ctrl_row)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.setSpacing(4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time [s]")
        self.plot_widget.setLabel("left",   "Unwrapped phase [rad]")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        pen_box = pg.mkPen(color=(220, 50, 50), width=1.5, style=QtCore.Qt.DashLine)
        self._vline0 = pg.InfiniteLine(pos=self.ann.t0, angle=90, pen=pen_box)
        self._vline1 = pg.InfiniteLine(pos=self.ann.t1, angle=90, pen=pen_box)
        self.plot_widget.addItem(self._vline0)
        self.plot_widget.addItem(self._vline1)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Vertical)
        self.scrollbar.setRange(self._di0, max(self._di1 - 1, self._di0))
        self.scrollbar.setValue(self._cur_ch)
        self.scrollbar.setInvertedAppearance(True)
        self.scrollbar.valueChanged.connect(self._on_scroll)


        scroll_col = QtWidgets.QVBoxLayout()
        scroll_col.setSpacing(2)
        scroll_col.addWidget(self.scrollbar, 1)

        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addLayout(scroll_col)
        layout.addLayout(plot_row, 1)

    def _compute_phase(self, ch: int) -> np.ndarray:
        import scipy.signal as sp
        sig = self.dataset.tr[ch, :].astype(np.float64)
        return np.unwrap(np.angle(sp.hilbert(sig)))

    def _plot(self, ch: int):
        self._cur_ch = ch
        ds   = self.dataset
        dist = float(ds.dist_m[ch]) if ch < len(ds.dist_m) else 0
        time = ds.time_s

        for item in list(self.plot_widget.listDataItems()):
            self.plot_widget.removeItem(item)

        pen_active = pg.mkPen(color=theme.current()["pg_line_main"], width=1.5)
        phase = self._compute_phase(ch)
        self.plot_widget.plot(time, phase, pen=pen_active)

        for fch, color in self._fixed_chs:
            phase_fch = self._compute_phase(fch)
            pen = pg.mkPen(color=(*color, 255), width=2.0)
            self.plot_widget.plot(time, phase_fch, pen=pen)

        amp_range = phase.max() - phase.min()
        margin = amp_range * 0.05 if amp_range > 0 else 1e-12
        auto_min = float(phase.min() - margin)
        auto_max = float(phase.max() + margin)

        ymin = self.sb_ymin.value()
        ymax = self.sb_ymax.value()
        if ymin == 0.0 and ymax == 0.0:
            self.sb_ymin.setValue(auto_min)
            self.sb_ymax.setValue(auto_max)
            ymin, ymax = auto_min, auto_max
        self.plot_widget.setYRange(ymin, ymax, padding=0)

        fixed_info = (
            "  |  Fixed: " + ", ".join(f"CH{fch}" for fch, _ in self._fixed_chs)
            if self._fixed_chs else ""
        )
        self.lbl_info.setText(f"Channel: {ch}  ({dist:.0f} m){fixed_info}")
        self.setWindowTitle(
            f"Signal (phase) - Event [{self.ann.id}]  "
            f"Channel {ch}  ({dist:.0f} m)"
        )

    def _on_scroll(self, val: int):
        self.spin_channel.blockSignals(True)
        self.spin_channel.setValue(val)
        self.spin_channel.blockSignals(False)
        self._plot(val)

    def _on_channel_spinbox(self, val: int):
        """Jump directly to a channel by typing its index."""
        val = max(self._di0, min(val, self._di1 - 1))
        self.scrollbar.setValue(val)

    def _on_apply_y(self):
        self.plot_widget.setYRange(self.sb_ymin.value(), self.sb_ymax.value(), padding=0)

    def _on_fix(self):
        ch = self._cur_ch
        if not any(fch == ch for fch, _ in self._fixed_chs):
            color = _TAB10[len(self._fixed_chs) % len(_TAB10)]
            self._fixed_chs.append((ch, color))
        self._plot(ch)

    def _on_clear(self):
        self._fixed_chs.clear()
        self._plot(self._cur_ch)

    def _export_png(self) -> None:
        default_name = f"signal_phase_{self.ann.id}_ch{self._cur_ch}.png".replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name, "PNG images (*.png)"
        )
        if not path:
            return
        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi
        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width() * scale), int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
        )
        img = scaled.toImage()
        dpm = int(300 / 0.0254)
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        if img.save(path, "PNG"):
            QtWidgets.QMessageBox.information(self, "Export successful", f"Saved 300 DPI PNG:\n{path}")
        else:
            QtWidgets.QMessageBox.critical(self, "Export failed", f"Could not save:\n{path}")


# ---------------------------------------------------------------------------
# G) Velocity estimation dialog
# ---------------------------------------------------------------------------

class VelocityDialog(QtWidgets.QDialog):
    """
    Apparent velocity estimator from manual point picks on a zoomed waterfall.

    The dialog shows the bbox region of the waterfall (with a configurable
    margin) using the same colormap as the main view.  The user clicks
    "Pick Points" to enter pick mode, then clicks on the image to place
    (t, d) pairs.  With ≥ 2 points a linear regression d = v·t + c is
    computed in real time and the best-fit line is drawn.

    The estimated velocity (m/s, with sign) and R² can be saved back to the
    annotation CSV via the "Save to annotation" button.
    """

    # Signal emitted when the user saves velocity to the annotation
    velocity_saved = QtCore.pyqtSignal(float, float)   # (velocity_ms, r2)

    def __init__(self, ann, dataset, colormap: pg.ColorMap,
                 vmin: float = None, vmax: float = None,
                 tr_display: np.ndarray = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Estimate Velocity — Event [{ann.id}]  "
            f"t={ann.t0:.2f}–{ann.t1:.2f} s  "
            f"d={ann.d0:.0f}–{ann.d1:.0f} m"
        )
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
        )
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.60), int(screen.height() * 0.55))

        self.ann     = ann
        self.dataset = dataset
        self.cmap    = colormap
        self._ext_vmin    = vmin
        self._ext_vmax    = vmax
        # Use the filtered array currently shown in the main waterfall if provided
        self._tr_display  = tr_display  # shape (n_dist, n_time), same as dataset.tr

        self._pick_mode = False
        self._picks     = []   # list of (t, d) tuples

        # Zoom margin: extend bbox by 50% on each side
        _margin_t = (ann.t1 - ann.t0) * 0.5
        _margin_d = (ann.d1 - ann.d0) * 0.5
        self._t0_view = max(float(dataset.time_s[0]),  ann.t0 - _margin_t)
        self._t1_view = min(float(dataset.time_s[-1]), ann.t1 + _margin_t)
        self._d0_view = max(float(dataset.dist_m[0]),  ann.d0 - _margin_d)
        self._d1_view = min(float(dataset.dist_m[-1]), ann.d1 + _margin_d)

        self._build_ui()
        self._render_waterfall()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- Control bar ---
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.setSpacing(8)

        self.btn_pick = QtWidgets.QPushButton("Pick Points")
        self.btn_pick.setCheckable(True)
        self.btn_pick.setMinimumWidth(110)
        self.btn_pick.toggled.connect(self._on_pick_toggled)

        self.btn_clear = QtWidgets.QPushButton("Clear Points")
        self.btn_clear.setMinimumWidth(110)
        self.btn_clear.clicked.connect(self._on_clear)

        self.lbl_result = QtWidgets.QLabel("Pick ≥ 2 points to estimate velocity")
        self.lbl_result.setStyleSheet("color: #e0e0e0; font-size: 10pt;")

        self.btn_save = QtWidgets.QPushButton("Save to annotation")
        self.btn_save.setMinimumWidth(150)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)

        btn_export = QtWidgets.QPushButton("Export")
        btn_export.setMinimumWidth(90)
        btn_export.clicked.connect(self._export_png)

        ctrl.addWidget(self.btn_pick)
        ctrl.addWidget(self.btn_clear)
        ctrl.addSpacing(12)
        ctrl.addWidget(self.lbl_result)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_save)
        ctrl.addWidget(btn_export)
        layout.addLayout(ctrl)

        # --- Waterfall plot ---
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time [s]")
        self.plot_widget.setLabel("left",   "Distance [m]")
        self.plot_widget.invertY(False)

        self.image_item = pg.ImageItem()
        self.image_item.setAutoDownsample(True)
        self.plot_widget.addItem(self.image_item)

        # Histogram
        self.histogram = pg.HistogramLUTWidget()
        self.histogram.setImageItem(self.image_item)
        self.histogram.gradient.setColorMap(self.cmap)
        self.histogram.setFixedWidth(110)

        # Bbox boundary (yellow dashed rectangle lines)
        pen_box = pg.mkPen(color=(255, 220, 0), width=1.5,
                            style=QtCore.Qt.DashLine)
        for pos, angle in [(self.ann.t0, 90), (self.ann.t1, 90)]:
            vl = pg.InfiniteLine(pos=pos, angle=angle, pen=pen_box)
            self.plot_widget.addItem(vl)
        for pos in [self.ann.d0, self.ann.d1]:
            hl = pg.InfiniteLine(pos=pos, angle=0, pen=pen_box)
            self.plot_widget.addItem(hl)

        # Pick points scatter
        self._scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(color=(255, 80, 0), width=2),
            brush=pg.mkBrush(255, 80, 0, 180),
        )
        self.plot_widget.addItem(self._scatter)

        # Regression line — thick orange dashed
        self._reg_line = self.plot_widget.plot(
            [], [],
            pen=pg.mkPen(color=(255, 80, 0), width=4,
                          style=QtCore.Qt.DashLine),
        )

        # Mouse click handler
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

        plot_row = QtWidgets.QHBoxLayout()
        plot_row.addWidget(self.plot_widget, 1)
        plot_row.addWidget(self.histogram)
        layout.addLayout(plot_row, 1)

    # ------------------------------------------------------------------
    # Waterfall rendering
    # ------------------------------------------------------------------

    def _render_waterfall(self):
        ds = self.dataset

        ti0 = int(np.argmin(np.abs(ds.time_s - self._t0_view)))
        ti1 = int(np.argmin(np.abs(ds.time_s - self._t1_view))) + 1
        di0 = int(np.argmin(np.abs(ds.dist_m - self._d0_view)))
        di1 = int(np.argmin(np.abs(ds.dist_m - self._d1_view))) + 1
        ti1 = min(ti1, ds.n_time)
        di1 = min(di1, ds.n_dist)

        # Use the filtered array shown in the main waterfall if available,
        # otherwise fall back to the raw dataset array
        tr_src = self._tr_display if self._tr_display is not None else ds.tr
        tr_sub = tr_src[di0:di1, ti0:ti1]   # (n_dist_sub, n_time_sub)

        x0 = float(ds.time_s[ti0])
        x1 = float(ds.time_s[min(ti1 - 1, ds.n_time - 1)])
        y0 = float(ds.dist_m[di0])
        y1 = float(ds.dist_m[min(di1 - 1, ds.n_dist - 1)])

        # Use levels from the main waterfall if provided; otherwise auto from local data
        if self._ext_vmin is not None and self._ext_vmax is not None:
            vmin = float(self._ext_vmin)
            vmax = float(self._ext_vmax)
        else:
            p99  = float(np.percentile(np.abs(tr_sub), 99))
            vmin, vmax = -p99, p99
        if vmin >= vmax:
            vmax = vmin + 1

        self.image_item.sigImageChanged.disconnect(self.histogram.item.imageChanged)
        self.image_item.setImage(tr_sub, autoLevels=False, levels=[vmin, vmax])
        self.image_item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.image_item.sigImageChanged.connect(self.histogram.item.imageChanged)
        self.histogram.item.imageChanged(autoLevel=False)
        self.histogram.setLevels(vmin, vmax)

        self.plot_widget.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1)
        self.plot_widget.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.02)

    # ------------------------------------------------------------------
    # Pick mode
    # ------------------------------------------------------------------

    def _on_pick_toggled(self, checked: bool):
        self._pick_mode = checked
        if checked:
            self.btn_pick.setText("Pick Points  [ON]")
            self.btn_pick.setStyleSheet(
                "background-color: #5a3a00; border: 1px solid #e0a020; color: #e0a020;"
            )
            self.plot_widget.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.btn_pick.setText("Pick Points")
            self.btn_pick.setStyleSheet("")
            self.plot_widget.setCursor(QtCore.Qt.ArrowCursor)

    def _on_scene_clicked(self, event):
        if not self._pick_mode:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return

        vb  = self.plot_widget.getPlotItem().vb
        pos = vb.mapSceneToView(event.scenePos())
        t, d = pos.x(), pos.y()

        # Validate within view bounds
        if not (self._t0_view <= t <= self._t1_view and
                self._d0_view <= d <= self._d1_view):
            return

        self._picks.append((t, d))
        self._update_picks()

    def _on_clear(self):
        self._picks.clear()
        self._scatter.setData([])
        self._reg_line.setData([], [])
        self.lbl_result.setText("Pick ≥ 2 points to estimate velocity")
        self.btn_save.setEnabled(False)
        self._velocity_ms = None
        self._r2 = None

    def _update_picks(self):
        if not self._picks:
            return

        pts = [{'pos': (t, d)} for t, d in self._picks]
        self._scatter.setData(pts)

        n = len(self._picks)
        if n < 2:
            self.lbl_result.setText(f"{n} point picked — need ≥ 2")
            self._reg_line.setData([], [])
            self.btn_save.setEnabled(False)
            return

        # Linear regression: d = v * t + c
        from scipy.stats import linregress
        t_arr = np.array([p[0] for p in self._picks])
        d_arr = np.array([p[1] for p in self._picks])
        slope, intercept, r, _, se = linregress(t_arr, d_arr)
        r2 = r ** 2
        self._velocity_ms = abs(slope)
        self._r2 = r2

        # Draw regression line across view t range
        t_fit = np.array([self._t0_view, self._t1_view])
        d_fit = slope * t_fit + intercept
        self._reg_line.setData(t_fit, d_fit)

        sign = "+" if slope >= 0 else ""
        self.lbl_result.setText(
            f"v = {sign}{slope:.1f} m/s     R² = {r2:.4f}     n = {n} pts"
        )
        self.lbl_result.setStyleSheet(
            "color: #ff5000; font-size: 10pt; font-weight: bold;"
            if r2 < 0.8 else
            "color: #44ff88; font-size: 10pt; font-weight: bold;"
        )
        self.btn_save.setEnabled(True)
        self.setWindowTitle(
            f"Estimate Velocity — Event [{self.ann.id}]  "
            f"v = {sign}{slope:.1f} m/s  R² = {r2:.4f}"
        )

    def _on_save(self):
        if self._velocity_ms is None:
            return
        self.velocity_saved.emit(self._velocity_ms, self._r2)
        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Velocity saved to annotation:\n"
            f"v = {self._velocity_ms:.1f} m/s,  R² = {self._r2:.4f}"
        )

    def _export_png(self):
        default_name = f"velocity_{self.ann.id}.png".replace(" ", "_")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PNG (300 DPI)", default_name, "PNG images (*.png)"
        )
        if not path:
            return
        screen_dpi = QtWidgets.QApplication.primaryScreen().logicalDotsPerInch()
        scale = 300.0 / screen_dpi
        pix = self.plot_widget.grab()
        scaled = pix.scaled(
            int(pix.width() * scale), int(pix.height() * scale),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation,
        )
        img = scaled.toImage()
        dpm = int(300 / 0.0254)
        img.setDotsPerMeterX(dpm); img.setDotsPerMeterY(dpm)
        if img.save(path, "PNG"):
            QtWidgets.QMessageBox.information(self, "Export successful",
                                               f"Saved 300 DPI PNG:\n{path}")
        else:
            QtWidgets.QMessageBox.critical(self, "Export failed",
                                            f"Could not save:\n{path}")
