"""
Waterfall display widget for DAS Explorer.
"""

import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

from dasexplorer.core.data_model import DASDataset
from dasexplorer.core.annotations import AnnType
from dasexplorer.gui import theme

COLORMAPS = [
    ("Rainbow",  "nipy_spectral"),
    ("Turbo",    "turbo"),
    ("Grays",    "gray"),
    ("Viridis",  "viridis"),
    ("Magma",    "magma"),
    ("Seismic",  "seismic"),
]


def _bbox_pen():
    return pg.mkPen(color=theme.current()["pg_bbox"], width=2)

def _bbox_pen_sel():
    return pg.mkPen(color=theme.current()["pg_bbox_sel"], width=2, style=QtCore.Qt.DashLine)

def _bbox_brush():
    c = theme.current()["pg_bbox"]
    return pg.mkBrush(color=(*c, 30))

# Single colour for all annotation types — same yellow as BBox for consistency
def _ann_pen():
    return pg.mkPen(color=theme.current()["pg_bbox"], width=2)

def _ann_pen_sel():
    return pg.mkPen(color=theme.current()["pg_bbox_sel"], width=2, style=QtCore.Qt.DashLine)

# Keep individual aliases for backwards compat
def _obb_pen():   return _ann_pen()
def _kp_pen():    return _ann_pen()
def _line_pen():  return _ann_pen()

_ANN_TYPE_PENS = {
    AnnType.BBOX: _ann_pen,
    AnnType.OBB:  _ann_pen,
    AnnType.KP:   _ann_pen,
    AnnType.LINE: _ann_pen,
}

class AnnotationROI(pg.ROI):
    """Plain ROI that emits sigRightClicked instead of showing pyqtgraph's context menu.
    
    Stores its own `ann_index` attribute so the index can be updated in-place
    when other annotations are removed (avoids stale lambda captures).
    """
    sigRightClicked = QtCore.pyqtSignal(object)

    def __init__(self, *args, ann_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ann_index = ann_index

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.sigRightClicked.emit(self)
        else:
            super().mouseClickEvent(ev)


class OBBCurveItem(pg.PlotCurveItem):
    """PlotCurveItem subclass for OBB polygons.

    Intercepts right-click at the QGraphicsItem level (mouseClickEvent)
    and emits sigRightClicked — identical pattern to AnnotationROI for BBox.
    Stores ann_index so the index can be updated in-place when annotations
    are removed, without stale lambda captures.
    """
    sigRightClicked = QtCore.pyqtSignal(object)

    def __init__(self, *args, ann_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ann_index = ann_index
        # PlotCurveItem does not accept mouse events by default — enable it.
        self.setAcceptedMouseButtons(QtCore.Qt.RightButton)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.sigRightClicked.emit(self)
        else:
            super().mouseClickEvent(ev)


class PolylineItem(pg.PlotCurveItem):
    """PlotCurveItem subclass for LINE and KP skeleton connectors.

    Intercepts right-click and emits sigRightClicked with ann_index,
    following the same pattern as AnnotationROI (BBox) and OBBCurveItem.
    """
    sigRightClicked = QtCore.pyqtSignal(object)

    def __init__(self, *args, ann_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ann_index = ann_index
        self.setAcceptedMouseButtons(QtCore.Qt.RightButton)

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.sigRightClicked.emit(self)
        else:
            super().mouseClickEvent(ev)


class ScatterAnnotItem(pg.ScatterPlotItem):
    """ScatterPlotItem subclass for KP and LINE vertex dots.

    Intercepts right-click and emits sigRightClicked with ann_index.
    ScatterPlotItem already accepts mouse events, but we need to intercept
    right-click before pyqtgraph shows its own menu.
    """
    sigRightClicked = QtCore.pyqtSignal(object)

    def __init__(self, *args, ann_index: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ann_index = ann_index

    def mouseClickEvent(self, ev) -> None:
        if ev.button() == QtCore.Qt.RightButton:
            ev.accept()
            self.sigRightClicked.emit(self)
        else:
            super().mouseClickEvent(ev)


class _EditKeyFilter(QtCore.QObject):
    """
    Application-level event filter that intercepts Enter and Esc
    while a WaterfallWidget is in vertex-edit mode.
    Installed on QApplication so it works regardless of keyboard focus.
    The _active guard prevents re-entrant calls if Qt delivers the
    same key event to multiple objects in the widget hierarchy.
    """
    def __init__(self, waterfall):
        super().__init__()
        self._wf = waterfall
        self._active = False   # re-entrancy guard

    def eventFilter(self, obj, event):
        if self._active:
            return False
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if self._wf._edit_mode:
                    self._active = True
                    try:
                        t = self._wf._edit_ann_type
                        if t == AnnType.OBB:
                            self._wf._commit_obb_edit()
                        elif t == AnnType.KP:
                            self._wf._commit_kp_edit()
                        elif t == AnnType.LINE:
                            self._wf._commit_line_edit()
                        else:
                            self._wf._commit_bbox_edit()
                    finally:
                        self._active = False
                    return True
            elif key == QtCore.Qt.Key_Escape:
                if self._wf._edit_mode:
                    self._active = True
                    try:
                        self._wf.cancel_edit()
                    finally:
                        self._active = False
                    return True
        return False


class WaterfallWidget(QtWidgets.QWidget):
    """Interactive waterfall view of a DAS dataset."""

    cursor_info          = QtCore.pyqtSignal(str)
    bbox_drawn           = QtCore.pyqtSignal(float, float, float, float)
    obb_drawn            = QtCore.pyqtSignal(float, float, float, float, float)
    kp_drawn             = QtCore.pyqtSignal(list, list)
    line_drawn           = QtCore.pyqtSignal(list, list)
    roi_edit_requested         = QtCore.pyqtSignal(int)   # open ID/comment dialog
    roi_edit_shape_requested   = QtCore.pyqtSignal(int)   # enter vertex-edit mode
    roi_remove_requested       = QtCore.pyqtSignal(int)
    # Emitted when the user finishes editing vertices
    bbox_edited          = QtCore.pyqtSignal(int, float, float, float, float)        # idx,t0,t1,d0,d1
    obb_edited           = QtCore.pyqtSignal(int, float, float, float, float, float) # idx,cx,cy,w,h,angle
    kp_edited            = QtCore.pyqtSignal(int, list, list)   # idx, pts_t, pts_d
    line_edited          = QtCore.pyqtSignal(int, list, list)   # idx, pts_t, pts_d
    roi_spectrogram_requested  = QtCore.pyqtSignal(int)
    roi_spectral_requested     = QtCore.pyqtSignal(int)
    roi_signal_requested       = QtCore.pyqtSignal(int)
    roi_signal_freq_requested  = QtCore.pyqtSignal(int)
    roi_signal_env_requested   = QtCore.pyqtSignal(int)
    roi_signal_phase_requested = QtCore.pyqtSignal(int)
    roi_velocity_requested     = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Annotation draw state
        self._ann_type         = AnnType.BBOX
        self._annotation_mode  = False
        self._drag_start       = None
        self._drag_roi         = None
        self._ann_pts_t: list  = []
        self._ann_pts_d: list  = []
        self._live_items: list = []
        self._annotation_rois  = []
        self._ann_geoms: list  = []

        # Vertex-edit state
        self._edit_mode        = False
        self._edit_flat_idx    = -1
        self._edit_ann_type    = None
        self._edit_pts_t: list = []
        self._edit_pts_d: list = []
        self._edit_items: list = []
        self._edit_drag_vertex = -1
        self._edit_toolbar     = None
        self._edit_key_filter  = None

        self._tr_display       = None
        self._tr_envelope      = None
        self._tr_envelope_src  = None
        self._rgb_signal_disconnected = False

        # Crosshair lines (visible only in annotation mode)
        _cross_pen = pg.mkPen(color=theme.current()["pg_crosshair"], width=1,
                              style=QtCore.Qt.DashLine)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=_cross_pen)
        self._hline = pg.InfiniteLine(angle=0,  movable=False, pen=_cross_pen)
        self._vline.setVisible(False)
        self._hline.setVisible(False)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        plot_container = QtWidgets.QWidget()
        stack = QtWidgets.QStackedLayout(plot_container)
        stack.setStackingMode(QtWidgets.QStackedLayout.StackAll)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Time [s]")
        self.plot_widget.setLabel("left", "Distance [m]")
        self.plot_widget.invertY(False)
        self.plot_widget.showGrid(x=False, y=False)

        self.image_item = pg.ImageItem()
        self.image_item.setAutoDownsample(True)
        self.plot_widget.addItem(self.image_item)
        # Crosshair — added after image so they render on top
        self.plot_widget.addItem(self._vline)
        self.plot_widget.addItem(self._hline)
        # Overlay shown while processing
        self._overlay = QtWidgets.QWidget()
        self._overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        ov_layout = QtWidgets.QVBoxLayout(self._overlay)
        ov_layout.setAlignment(QtCore.Qt.AlignCenter)
        self._overlay_label = QtWidgets.QLabel("")
        self._overlay_label.setAlignment(QtCore.Qt.AlignCenter)
        self._overlay_label.setStyleSheet(
            "color: #ffffff; font-size: 13pt;"
            "background: rgba(0,0,0,170);"
            "padding: 24px 32px; border-radius: 10px;"
        )
        ov_layout.addWidget(self._overlay_label)
        self._overlay.setVisible(False)

        stack.addWidget(self.plot_widget)
        stack.addWidget(self._overlay)

        self.histogram = pg.HistogramLUTWidget()
        self.histogram.setImageItem(self.image_item)
        self._apply_colormap("nipy_spectral")

        self.combo_cmap = QtWidgets.QComboBox()
        for label, _ in COLORMAPS:
            self.combo_cmap.addItem(label)
        self.combo_cmap.currentIndexChanged.connect(self._on_cmap_changed)

        histogram_column = QtWidgets.QVBoxLayout()
        histogram_column.setSpacing(4)
        histogram_column.addWidget(self.histogram, 1)
        histogram_column.addWidget(self.combo_cmap, 0)

        outer.addWidget(plot_container, 1)
        outer.addLayout(histogram_column, 0)

        self.dataset = None
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)
        self._configure_context_menu()

    # ------------------------------------------------------------------
    # Annotation mode
    # ------------------------------------------------------------------

    def set_annotation_mode(self, active: bool,
                             ann_type: AnnType = AnnType.BBOX) -> None:
        """Enable or disable annotation drawing mode for the given type."""
        self._annotation_mode = active
        self._ann_type = ann_type
        vb = self.plot_widget.getPlotItem().vb
        if active:
            vb.setMouseEnabled(x=False, y=False)
            self.plot_widget.setCursor(QtCore.Qt.CrossCursor)
            self._vline.setVisible(True)
            self._hline.setVisible(True)
        else:
            vb.setMouseEnabled(x=True, y=True)
            self.plot_widget.setCursor(QtCore.Qt.ArrowCursor)
            self._vline.setVisible(False)
            self._hline.setVisible(False)
            self._cancel_annotation()

    def _cancel_annotation(self) -> None:
        """Discard the current in-progress annotation and clean up live preview."""
        # Remove BBox/OBB live ROI
        if self._drag_roi is not None:
            self.plot_widget.removeItem(self._drag_roi)
            self._drag_roi = None
        # Remove KP/LINE live items
        for item in self._live_items:
            self.plot_widget.removeItem(item)
        self._live_items.clear()
        self._ann_pts_t.clear()
        self._ann_pts_d.clear()
        # Disconnect mouse-move signal
        try:
            self.plot_widget.scene().sigMouseMoved.disconnect(self._on_drag_move)
        except TypeError:
            pass
        self._drag_start = None

    # keep old name as alias for code that still calls _cancel_drag
    def _cancel_drag(self) -> None:
        self._cancel_annotation()

    # ------------------------------------------------------------------
    # Key event — Enter/Escape to finalise or cancel KP / LINE
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if self._annotation_mode:
            if self._ann_pts_t and self._ann_type != AnnType.OBB:
                if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                    self._finalise_multipoint()
                    return
            if event.key() == QtCore.Qt.Key_Escape:
                self._cancel_annotation()
                return
        super().keyPressEvent(event)

    def _finalise_multipoint(self) -> None:
        """Emit the signal for KP or LINE once the user presses Enter."""
        if self._ann_type == AnnType.KP and len(self._ann_pts_t) >= 1:
            self.kp_drawn.emit(list(self._ann_pts_t), list(self._ann_pts_d))
        elif self._ann_type == AnnType.LINE and len(self._ann_pts_t) >= 2:
            self.line_drawn.emit(list(self._ann_pts_t), list(self._ann_pts_d))
        self._cancel_annotation()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def _on_scene_clicked(self, event) -> None:
        # Right-click outside annotation mode → context menu for nearest non-BBox ROI
        if event.button() == QtCore.Qt.RightButton and not self._annotation_mode:
            if self._ann_geoms and self.dataset is not None:
                vb  = self.plot_widget.getPlotItem().vb
                pos = vb.mapSceneToView(event.scenePos())
                t_c, d_c = pos.x(), pos.y()

                # Compute distances in screen pixels for reliable thresholding
                best_idx, best_dist_px = -1, float('inf')
                for flat_idx, pts_t, pts_d, at in self._ann_geoms:
                    for pt, pd in zip(pts_t, pts_d):
                        sp = vb.mapViewToScene(pg.Point(pt, pd))
                        sc = event.scenePos()
                        dx = sp.x() - sc.x()
                        dy = sp.y() - sc.y()
                        dist_px = (dx**2 + dy**2) ** 0.5
                        if dist_px < best_dist_px:
                            best_dist_px = dist_px
                            best_idx = flat_idx

                if best_idx >= 0 and best_dist_px < 40:   # 40px threshold
                    best_type = None
                    for fi, _, _, at in self._ann_geoms:
                        if fi == best_idx:
                            best_type = at
                            break
                    self._show_context_menu(best_idx, best_type)
            return

        if not self._annotation_mode or self.dataset is None:
            return
        vb  = self.plot_widget.getPlotItem().vb
        pos = vb.mapSceneToView(event.scenePos())
        t, d = pos.x(), pos.y()

        ds = self.dataset
        if not (ds.time_s[0] <= t <= ds.time_s[-1] and
                ds.dist_m[0] <= d <= ds.dist_m[-1]):
            return

        if self._ann_type == AnnType.BBOX:
            self._handle_click_bbox(t, d)
        elif self._ann_type == AnnType.OBB:
            self._handle_click_obb(t, d)
        elif self._ann_type in (AnnType.KP, AnnType.LINE):
            self._handle_click_multipoint(t, d, event)

    # ── BBox ──────────────────────────────────────────────────────────

    def _handle_click_bbox(self, t: float, d: float) -> None:
        if self._drag_start is None:
            self._drag_start = (t, d)
            self._drag_roi = pg.RectROI(
                [t, d], [0.001, 0.001],
                pen=_bbox_pen(), movable=False, resizable=False,
            )
            self._drag_roi.removeHandle(0)
            self.plot_widget.addItem(self._drag_roi)
            self.plot_widget.scene().sigMouseMoved.connect(self._on_drag_move)
        else:
            t0, d0 = self._drag_start
            t1, d1 = t, d
            self._cancel_annotation()
            if abs(t1 - t0) > 0.01 and abs(d1 - d0) > 1:
                self.bbox_drawn.emit(
                    min(t0, t1), max(t0, t1),
                    min(d0, d1), max(d0, d1),
                )

    # ── OBB ───────────────────────────────────────────────────────────
    # Three-click workflow (Labelme style):
    #   Click 1 → first vertex of the long axis (v1)
    #   Click 2 → second vertex of the long axis (v2)
    #   Click 3 → width: perpendicular offset from the v1-v2 axis
    # Centre = midpoint(v1, v2). All state lives in _ann_pts_t/d.

    def _handle_click_obb(self, t: float, d: float) -> None:
        import math
        n_pts = len(self._ann_pts_t)

        if n_pts == 0:
            # Click 1: first axis vertex — draw a dot, start tracking mouse
            self._ann_pts_t.append(t)
            self._ann_pts_d.append(d)
            dot = pg.ScatterPlotItem(
                [t], [d], pen=_ann_pen(),
                brush=pg.mkBrush(color=(*_ann_pen().color().getRgb()[:3], 220)),
                size=10, symbol='o',
            )
            self.plot_widget.addItem(dot)
            self._live_items.append(dot)   # index 0 = v1 dot
            self.plot_widget.scene().sigMouseMoved.connect(self._on_drag_move)

        elif n_pts == 1:
            # Click 2: second axis vertex — draw dot, enter width phase
            x1, y1 = self._ann_pts_t[0], self._ann_pts_d[0]
            dx, dy = t - x1, d - y1
            if math.hypot(dx, dy) < 0.5:
                return
            self._ann_pts_t.append(t)
            self._ann_pts_d.append(d)
            dot2 = pg.ScatterPlotItem(
                [t], [d], pen=_ann_pen(),
                brush=pg.mkBrush(color=(*_ann_pen().color().getRgb()[:3], 220)),
                size=10, symbol='o',
            )
            self.plot_widget.addItem(dot2)
            self._live_items.append(dot2)  # index 1 = v2 dot

        elif n_pts == 2:
            # Click 3: width click — compute OBB and emit
            x1, y1 = self._ann_pts_t[0], self._ann_pts_d[0]
            x2, y2 = self._ann_pts_t[1], self._ann_pts_d[1]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # Normalise to viewport pixel coords so t and d are comparable
            vb = self.plot_widget.getPlotItem().vb
            p1 = vb.mapViewToScene(pg.Point(x1, y1))
            p2 = vb.mapViewToScene(pg.Point(x2, y2))
            pc = vb.mapViewToScene(pg.Point(t, d))
            # Vector along axis in screen space
            ax = p2.x() - p1.x()
            ay = p2.y() - p1.y()
            axis_len = math.hypot(ax, ay)
            if axis_len < 1.0:
                self._cancel_annotation()
                return
            # Perpendicular distance in screen pixels
            h_px = abs((pc.x() - p1.x()) * (-ay / axis_len) +
                       (pc.y() - p1.y()) * (ax / axis_len))
            # Convert h back to data units using the d-axis scale
            # (h is measured perpendicular to the axis, so we use the
            #  ratio of data-to-pixel for both axes weighted by the
            #  perpendicular direction)
            vr = vb.viewRange()
            t_range = vr[0][1] - vr[0][0]
            d_range = vr[1][1] - vr[1][0]
            scene_rect = vb.sceneBoundingRect()
            t_per_px = t_range / scene_rect.width() if scene_rect.width() > 0 else 1
            d_per_px = d_range / scene_rect.height() if scene_rect.height() > 0 else 1
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            # Perpendicular direction in data space, with proper scale
            h = h_px * math.hypot(cos_a * d_per_px, sin_a * t_per_px)
            h = max(h, min(t_per_px, d_per_px))   # at least 1 pixel worth
            w = math.hypot(x2 - x1, y2 - y1) / 2.0
            self._cancel_annotation()
            self.obb_drawn.emit(cx, cy, w, h, math.degrees(angle_rad))

    # ── Keypoints / LineString ─────────────────────────────────────────

    def _handle_click_multipoint(self, t: float, d: float, event) -> None:
        is_double = (event.double() if hasattr(event, 'double') else False)

        self._ann_pts_t.append(t)
        self._ann_pts_d.append(d)

        # Draw a dot at this point
        dot = pg.ScatterPlotItem(
            [t], [d],
            pen=_ANN_TYPE_PENS[self._ann_type](),
            brush=pg.mkBrush(color=(*_ANN_TYPE_PENS[self._ann_type]().color().getRgb()[:3], 180)),
            size=8, symbol='o',
        )
        self.plot_widget.addItem(dot)
        self._live_items.append(dot)

        # Draw a connecting line if more than one point
        if len(self._ann_pts_t) > 1:
            seg = pg.PlotCurveItem(
                self._ann_pts_t, self._ann_pts_d,
                pen=_ANN_TYPE_PENS[self._ann_type](),
            )
            self.plot_widget.addItem(seg)
            self._live_items.append(seg)

        # Double-click or Enter → finalise
        if is_double:
            self._finalise_multipoint()

    # ── Live preview while dragging ─────────────────────────────────────

    def _on_drag_move(self, scene_pos) -> None:
        vb  = self.plot_widget.getPlotItem().vb
        pos = vb.mapSceneToView(scene_pos)
        t1, d1 = pos.x(), pos.y()

        if self._ann_type == AnnType.BBOX:
            if self._drag_start is None or self._drag_roi is None:
                return
            t0, d0 = self._drag_start
            self._drag_roi.setPos([min(t0, t1), min(d0, d1)])
            self._drag_roi.setSize([abs(t1 - t0), abs(d1 - d0)])

        elif self._ann_type == AnnType.OBB:
            import math
            n_pts = len(self._ann_pts_t)
            if n_pts == 0:
                return

            # Keep only the fixed vertex dots (indices 0..n_pts-1), remove preview
            while len(self._live_items) > n_pts:
                self.plot_widget.removeItem(self._live_items.pop())

            if n_pts == 1:
                # Phase 1: rubber-band line from v1 to cursor
                x1, y1 = self._ann_pts_t[0], self._ann_pts_d[0]
                line = pg.PlotCurveItem([x1, t1], [y1, d1], pen=_ann_pen())
                self.plot_widget.addItem(line)
                self._live_items.append(line)

            elif n_pts == 2:
                # Phase 2: full rotated rectangle preview with correct h
                x1, y1 = self._ann_pts_t[0], self._ann_pts_d[0]
                x2, y2 = self._ann_pts_t[1], self._ann_pts_d[1]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                w = math.hypot(x2 - x1, y2 - y1) / 2.0
                # Compute h in screen-normalised space (same logic as click 3)
                vb = self.plot_widget.getPlotItem().vb
                p1 = vb.mapViewToScene(pg.Point(x1, y1))
                p2 = vb.mapViewToScene(pg.Point(x2, y2))
                pc = vb.mapViewToScene(pg.Point(t1, d1))
                ax = p2.x() - p1.x()
                ay = p2.y() - p1.y()
                axis_len = math.hypot(ax, ay)
                if axis_len < 1.0:
                    return
                h_px = abs((pc.x() - p1.x()) * (-ay / axis_len) +
                           (pc.y() - p1.y()) * (ax / axis_len))
                vr = vb.viewRange()
                t_range = vr[0][1] - vr[0][0]
                d_range = vr[1][1] - vr[1][0]
                scene_rect = vb.sceneBoundingRect()
                t_per_px = t_range / scene_rect.width() if scene_rect.width() > 0 else 1
                d_per_px = d_range / scene_rect.height() if scene_rect.height() > 0 else 1
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                h = h_px * math.hypot(cos_a * d_per_px, sin_a * t_per_px)
                h = max(h, min(t_per_px, d_per_px))
                offsets = [(w, h), (w, -h), (-w, -h), (-w, h), (w, h)]
                xs = [cx + cos_a * ox - sin_a * oy for ox, oy in offsets]
                ys = [cy + sin_a * ox + cos_a * oy for ox, oy in offsets]
                poly = pg.PlotCurveItem(xs, ys, pen=_ann_pen())
                self.plot_widget.addItem(poly)
                self._live_items.append(poly)

    def add_annotation_roi(self, index: int, t0: float, t1: float,
                            d0: float, d1: float, label: str = "") -> None:
        """Draw a BBox annotation on the waterfall."""
        roi = AnnotationROI(
            [t0, d0], [t1 - t0, d1 - d0],
            ann_index=index,
            pen=_bbox_pen(), movable=False, resizable=False,
        )
        roi.sigRightClicked.connect(self._on_roi_right_clicked)
        self.plot_widget.addItem(roi)

        # Label at top-left corner of the box
        text = pg.TextItem(label, color=(255, 220, 0), anchor=(0, 1))
        from PyQt5.QtGui import QFont
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        text.setFont(f)
        text.setPos(t0, d1)
        self.plot_widget.addItem(text)

        self._annotation_rois.append((roi, text, index))

    def update_annotation_label(self, index: int, label: str) -> None:
        """Update the text label of the ROI at the given index."""
        for roi, text, idx in self._annotation_rois:
            if idx == index:
                text.setText(label)
                break

    def remove_annotation_roi(self, index: int) -> None:
        """Remove the ROI and label for annotation at given index,
        and decrement the stored index on all entries with higher indices.
        Works for all ROI types (AnnotationROI, PlotCurveItem, ScatterPlotItem).
        """
        remaining = []
        for roi, text, idx in self._annotation_rois:
            if idx == index:
                self.plot_widget.removeItem(roi)
                self.plot_widget.removeItem(text)
            else:
                new_idx = idx - 1 if idx > index else idx
                # Keep AnnotationROI.ann_index in sync for BBox hit-detection
                if hasattr(roi, 'ann_index'):
                    roi.ann_index = new_idx
                remaining.append((roi, text, new_idx))
        self._annotation_rois = remaining
        # Also update flat_idx in _ann_geoms
        self._ann_geoms = [
            (fi - 1 if fi > index else fi, pts_t, pts_d, at)
            for fi, pts_t, pts_d, at in self._ann_geoms
            if fi != index
        ]

    def clear_annotation_rois(self) -> None:
        """Remove all annotation ROIs from the plot."""
        for roi, text, _ in self._annotation_rois:
            self.plot_widget.removeItem(roi)
            self.plot_widget.removeItem(text)
        self._annotation_rois = []
        self._ann_geoms = []

    def set_rois_visible(self, visible: bool) -> None:
        for roi, text, _ in self._annotation_rois:
            roi.setVisible(visible)
            text.setVisible(visible)

    def add_obb_roi(self, index: int, cx_t: float, cy_d: float,
                    w_t: float, h_d: float, angle_deg: float,
                    label: str = "", flat_idx: int = -1) -> None:
        """Draw an OBB annotation as a rotated polygon on the waterfall."""
        import math
        from PyQt5.QtGui import QFont
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        # Vertex order: 0=(+w,+h), 1=(+w,-h), 2=(-w,-h), 3=(-w,+h), 4=(+w,+h) close
        offsets = [(w_t, h_d), (w_t, -h_d), (-w_t, -h_d), (-w_t, h_d), (w_t, h_d)]
        xs = [cx_t + cos_a * ox - sin_a * oy for ox, oy in offsets]
        ys = [cy_d + sin_a * ox + cos_a * oy for ox, oy in offsets]

        poly = OBBCurveItem(xs, ys, pen=_ann_pen(), ann_index=flat_idx if flat_idx >= 0 else index)
        poly.sigRightClicked.connect(self._on_obb_right_clicked)
        self.plot_widget.addItem(poly)

        # Label at the top-right corner of the OBB (vertex 0 = +w,+h in local frame).
        # anchor=(0, 1): text origin at bottom-left of the label, so it sits just
        # outside the corner without overlapping the polygon edge.
        text = pg.TextItem(label, anchor=(0, 1), color=_ann_pen().color())
        f = QFont()
        f.setPointSize(8)
        f.setBold(True)
        text.setFont(f)
        text.setPos(xs[0], ys[0])
        self.plot_widget.addItem(text)
        self._annotation_rois.append((poly, text, index))

        if flat_idx >= 0:
            # Store the 4 corners + centre so the nearest-point hit-test in
            # _on_scene_clicked works for clicks both near edges and inside.
            self._ann_geoms.append((
                flat_idx,
                xs[:4] + [cx_t],
                ys[:4] + [cy_d],
                AnnType.OBB,
            ))

    def add_kp_roi(self, index: int, pts_t: list, pts_d: list,
                   label: str = "", flat_idx: int = -1) -> None:
        """Draw a Keypoints annotation: scatter dots + dashed skeleton connectors."""
        from PyQt5.QtGui import QFont
        pen = _ann_pen()
        c = pen.color().getRgb()[:3]
        fi = flat_idx if flat_idx >= 0 else index

        # Dashed connector line between points (skeleton style).
        # Drawn first so dots appear on top.
        if len(pts_t) > 1:
            dash_pen = pg.mkPen(color=c, width=1, style=QtCore.Qt.DashLine)
            connector = PolylineItem(pts_t, pts_d, pen=dash_pen, ann_index=fi)
            connector.sigRightClicked.connect(self._on_kp_right_clicked)
            self.plot_widget.addItem(connector)
        else:
            connector = None

        # Scatter dots — right-click on dots also triggers the menu.
        scatter = ScatterAnnotItem(
            pts_t, pts_d,
            pen=pen,
            brush=pg.mkBrush(color=(*c, 180)),
            size=10, symbol='o',
            ann_index=fi,
        )
        scatter.sigRightClicked.connect(self._on_kp_right_clicked)
        self.plot_widget.addItem(scatter)

        # Label at first point.
        text = pg.TextItem(label, anchor=(0, 1), color=pen.color())
        f = QFont(); f.setPointSize(8); f.setBold(True)
        text.setFont(f)
        if pts_t:
            text.setPos(pts_t[0], pts_d[0])
        self.plot_widget.addItem(text)

        # Single _annotation_rois entry keyed on scatter (primary hit item).
        # connector stored separately so remove_annotation_roi can clean it up.
        self._annotation_rois.append((scatter, text, index))
        if connector is not None:
            self._annotation_rois.append((connector, text, index))

        if flat_idx >= 0:
            self._ann_geoms.append((flat_idx, pts_t, pts_d, AnnType.KP))

    def add_line_roi(self, index: int, pts_t: list, pts_d: list,
                     label: str = "", flat_idx: int = -1) -> None:
        """Draw a LineString annotation: solid line + vertex dots."""
        from PyQt5.QtGui import QFont
        pen = _ann_pen()
        c = pen.color().getRgb()[:3]
        fi = flat_idx if flat_idx >= 0 else index

        # Solid line — primary hit target for right-click.
        line = PolylineItem(pts_t, pts_d, pen=pen, ann_index=fi)
        line.sigRightClicked.connect(self._on_line_right_clicked)
        self.plot_widget.addItem(line)

        # Vertex dots — also respond to right-click.
        scatter = ScatterAnnotItem(
            pts_t, pts_d,
            pen=pen,
            brush=pg.mkBrush(color=(*c, 160)),
            size=7, symbol='o',
            ann_index=fi,
        )
        scatter.sigRightClicked.connect(self._on_line_right_clicked)
        self.plot_widget.addItem(scatter)

        # Label at first vertex.
        text = pg.TextItem(label, anchor=(0, 1), color=pen.color())
        f = QFont(); f.setPointSize(8); f.setBold(True)
        text.setFont(f)
        if pts_t:
            text.setPos(pts_t[0], pts_d[0])
        self.plot_widget.addItem(text)

        # Two entries: line (primary) + scatter (auxiliary), both tracked for removal.
        self._annotation_rois.append((line, text, index))
        self._annotation_rois.append((scatter, text, index))

        if flat_idx >= 0:
            self._ann_geoms.append((flat_idx, pts_t, pts_d, AnnType.LINE))

    def highlight_annotation_roi(self, index: int) -> None:
        """Visually highlight the selected annotation ROI."""
        for roi, text, idx in self._annotation_rois:
            roi.setPen(_bbox_pen_sel() if idx == index else _bbox_pen())

    # ------------------------------------------------------------------
    # BBox shape editing
    # ------------------------------------------------------------------

    def enter_bbox_edit_mode(self, flat_idx: int,
                              t0: float, t1: float,
                              d0: float, d1: float) -> None:
        """Activate vertex-edit mode for the BBox at flat_idx."""
        self.cancel_edit()
        self._edit_mode        = True
        self._edit_flat_idx    = flat_idx
        self._edit_ann_type    = AnnType.BBOX
        self._edit_pts_t       = [t0, t1]
        self._edit_pts_d       = [d0, d1]
        self._edit_drag_vertex = -1

        self._draw_bbox_edit_handles()

        # Floating Save / Cancel toolbar
        self._edit_toolbar = QtWidgets.QWidget(
            self.window(),
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint,
        )
        self._edit_toolbar.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        tb_layout = QtWidgets.QHBoxLayout(self._edit_toolbar)
        tb_layout.setContentsMargins(8, 6, 8, 6)
        tb_layout.setSpacing(8)

        lbl = QtWidgets.QLabel("Edit Shape:")
        lbl.setStyleSheet("color: #e0a020; font-weight: bold; font-size: 9pt;")
        tb_layout.addWidget(lbl)

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setMinimumWidth(90)
        btn_save.setStyleSheet(
            "QPushButton { background: #2a5a2a; color: #80ff80; "
            "border: 1px solid #60c060; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #3a7a3a; }"
        )
        btn_save.clicked.connect(self._commit_bbox_edit)
        tb_layout.addWidget(btn_save)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setMinimumWidth(90)
        btn_cancel.setStyleSheet(
            "QPushButton { background: #5a2a2a; color: #ff8080; "
            "border: 1px solid #c06060; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #7a3a3a; }"
        )
        btn_cancel.clicked.connect(self.cancel_edit)
        tb_layout.addWidget(btn_cancel)

        self._edit_toolbar.adjustSize()

        # Position at top-right of the plot area (above the colour histogram)
        plot_global = self.plot_widget.mapToGlobal(
            self.plot_widget.rect().topRight()
        )
        tb = self._edit_toolbar
        tb.move(plot_global.x() - tb.width() - 12,
                plot_global.y() + 8)
        tb.show()

    def _draw_bbox_edit_handles(self) -> None:
        """Draw the BBox edit overlay: dashed orange rect + 4 draggable TargetItems."""
        for item in self._edit_items:
            if hasattr(item, 'sigPositionChanged'):
                try:
                    item.sigPositionChanged.disconnect()
                except TypeError:
                    pass
            self.plot_widget.removeItem(item)
        self._edit_items.clear()

        t0, t1 = self._edit_pts_t
        d0, d1 = self._edit_pts_d
        orange_pen = pg.mkPen(color=(255, 140, 0), width=2,
                               style=QtCore.Qt.DashLine)

        # Dashed rectangle outline
        rect = pg.PlotCurveItem(
            [t0, t1, t1, t0, t0],
            [d0, d0, d1, d1, d0],
            pen=orange_pen,
        )
        self.plot_widget.addItem(rect)
        self._edit_items.append(rect)

        # Four draggable TargetItems at corners: 0=TL, 1=TR, 2=BR, 3=BL
        corner_coords = [(t0, d0), (t1, d0), (t1, d1), (t0, d1)]
        for v_idx, (ct, cd) in enumerate(corner_coords):
            target = pg.TargetItem(
                pos=(ct, cd),
                size=14,
                symbol='o',
                pen=pg.mkPen(color=(255, 255, 255), width=1),
                brush=pg.mkBrush(color=(255, 140, 0, 230)),
                movable=True,
            )
            # Capture v_idx in closure
            def make_moved(vi):
                def on_moved():
                    pos = self._edit_items[1 + vi].pos()
                    t, d = pos.x(), pos.y()
                    t0_, t1_ = self._edit_pts_t
                    d0_, d1_ = self._edit_pts_d
                    # Update the relevant edges
                    if vi in (0, 3):  # left
                        t0_ = min(t, t1_ - 0.001)
                    if vi in (1, 2):  # right
                        t1_ = max(t, t0_ + 0.001)
                    if vi in (0, 1):  # bottom
                        d0_ = min(d, d1_ - 0.001)
                    if vi in (2, 3):  # top
                        d1_ = max(d, d0_ + 0.001)
                    self._edit_pts_t = [t0_, t1_]
                    self._edit_pts_d = [d0_, d1_]
                    # Redraw outline and reposition all corners
                    self._update_bbox_edit_outline()
                return on_moved
            target.sigPositionChanged.connect(make_moved(v_idx))
            self.plot_widget.addItem(target)
            self._edit_items.append(target)  # indices 1..4

    def _update_bbox_edit_outline(self) -> None:
        """Update only the outline rect without recreating the TargetItems."""
        if not self._edit_items:
            return
        t0, t1 = self._edit_pts_t
        d0, d1 = self._edit_pts_d
        # item[0] = outline rect
        self._edit_items[0].setData(
            [t0, t1, t1, t0, t0],
            [d0, d0, d1, d1, d0],
        )
        # items[1..4] = TargetItems — reposition them
        corner_coords = [(t0, d0), (t1, d0), (t1, d1), (t0, d1)]
        for i, (ct, cd) in enumerate(corner_coords):
            if 1 + i < len(self._edit_items):
                target = self._edit_items[1 + i]
                if hasattr(target, 'setPos'):
                    # block signals to avoid recursion
                    target.sigPositionChanged.disconnect()
                    target.setPos(ct, cd)
                    def make_moved(vi=i):
                        def on_moved():
                            pos = self._edit_items[1 + vi].pos()
                            t, d = pos.x(), pos.y()
                            t0_, t1_ = self._edit_pts_t
                            d0_, d1_ = self._edit_pts_d
                            if vi in (0, 3): t0_ = min(t, t1_ - 0.001)
                            if vi in (1, 2): t1_ = max(t, t0_ + 0.001)
                            if vi in (0, 1): d0_ = min(d, d1_ - 0.001)
                            if vi in (2, 3): d1_ = max(d, d0_ + 0.001)
                            self._edit_pts_t = [t0_, t1_]
                            self._edit_pts_d = [d0_, d1_]
                            self._update_bbox_edit_outline()
                        return on_moved
                    target.sigPositionChanged.connect(make_moved(i))

    def cancel_edit(self) -> None:
        """Leave edit mode without saving."""
        if not self._edit_mode:
            return
        self._edit_mode = False
        self._edit_flat_idx = -1
        self._edit_ann_type = None
        self._edit_drag_vertex = -1
        for item in self._edit_items:
            if hasattr(item, 'sigPositionChanged'):
                try:
                    item.sigPositionChanged.disconnect()
                except TypeError:
                    pass
            self.plot_widget.removeItem(item)
        self._edit_items.clear()
        self.plot_widget.setCursor(QtCore.Qt.ArrowCursor)
        # Close floating toolbar
        tb = getattr(self, '_edit_toolbar', None)
        if tb is not None:
            tb.hide()
            tb.deleteLater()
            self._edit_toolbar = None
        # Remove event filter if somehow still installed
        ef = getattr(self, '_edit_key_filter', None)
        if ef is not None:
            QtWidgets.QApplication.instance().removeEventFilter(ef)
            self._edit_key_filter = None
        # Restore normal scene click handler
        try:
            self.plot_widget.scene().sigMouseClicked.disconnect(self._on_scene_clicked)
        except TypeError:
            pass
        self.plot_widget.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def _on_edit_clicked(self, event) -> None:
        if not self._edit_mode:
            return
        if event.button() == QtCore.Qt.RightButton:
            self.cancel_edit()
            return
        vb  = self.plot_widget.getPlotItem().vb
        pos = vb.mapSceneToView(event.scenePos())
        t, d = pos.x(), pos.y()

        if self._edit_drag_vertex >= 0:
            # Drop the vertex being dragged
            self._edit_drag_vertex = -1
            return

        if event.double():
            self._commit_bbox_edit()
            return

        # Find nearest corner in screen pixels (threshold 30px)
        t0, t1 = self._edit_pts_t
        d0, d1 = self._edit_pts_d
        corners_t = [t0, t1, t1, t0]
        corners_d = [d0, d0, d1, d1]
        best_v, best_px = -1, float('inf')
        sc = event.scenePos()
        for i, (ct, cd) in enumerate(zip(corners_t, corners_d)):
            sp = vb.mapViewToScene(pg.Point(ct, cd))
            dist_px = ((sp.x()-sc.x())**2 + (sp.y()-sc.y())**2) ** 0.5
            if dist_px < best_px:
                best_px = dist_px
                best_v = i
        if best_v >= 0 and best_px < 30:
            self._edit_drag_vertex = best_v

    def _on_edit_move(self, scene_pos) -> None:
        if not self._edit_mode or self._edit_drag_vertex < 0:
            return
        vb  = self.plot_widget.getPlotItem().vb
        pos = vb.mapSceneToView(scene_pos)
        t, d = pos.x(), pos.y()

        # Corners: 0=TL, 1=TR, 2=BR, 3=BL
        # Moving corner v moves both its t and d edges independently
        t0, t1 = self._edit_pts_t
        d0, d1 = self._edit_pts_d
        v = self._edit_drag_vertex

        # Each corner controls: TL→(t0,d0), TR→(t1,d0), BR→(t1,d1), BL→(t0,d1)
        if v in (0, 3):   # left side
            t0 = min(t, t1 - 0.01)
        if v in (1, 2):   # right side
            t1 = max(t, t0 + 0.01)
        if v in (0, 1):   # bottom side (d0)
            d0 = min(d, d1 - 1.0)
        if v in (2, 3):   # top side (d1)
            d1 = max(d, d0 + 1.0)

        self._edit_pts_t = [t0, t1]
        self._edit_pts_d = [d0, d1]
        self._draw_bbox_edit_handles()

    def _commit_bbox_edit(self) -> None:
        """Emit bbox_edited and leave edit mode."""
        flat_idx = self._edit_flat_idx
        t0, t1 = self._edit_pts_t
        d0, d1 = self._edit_pts_d
        self.cancel_edit()
        self.bbox_edited.emit(flat_idx, t0, t1, d0, d1)

    # ------------------------------------------------------------------
    # OBB shape editing
    # ------------------------------------------------------------------
    # Three handles:
    #   handle 0 — centre    (cx_t, cy_d)          → moves the whole OBB
    #   handle 1 — axis tip  (cx + cos_a*w, cy + sin_a*w) → controls length + angle
    #   handle 2 — width tip (cx - sin_a*h, cy + cos_a*h) → controls half-width
    #
    # All OBB parameters are stored compactly in two lists that mirror the
    # BBox convention so cancel_edit() / cancel_edit() can stay type-agnostic:
    #   _edit_pts_t = [cx_t, w_t, h_d]
    #   _edit_pts_d = [cy_d, angle_deg]   (angle stored here for convenience)

    def enter_obb_edit_mode(self, flat_idx: int,
                             cx_t: float, cy_d: float,
                             w_t: float, h_d: float,
                             angle_deg: float) -> None:
        """Activate vertex-edit mode for the OBB at flat_idx."""
        self.cancel_edit()
        self._edit_mode        = True
        self._edit_flat_idx    = flat_idx
        self._edit_ann_type    = AnnType.OBB
        # Pack OBB params into the two list slots
        self._edit_pts_t       = [cx_t, w_t, h_d]
        self._edit_pts_d       = [cy_d, angle_deg]
        self._edit_drag_vertex = -1

        self._draw_obb_edit_handles()

        # Floating Save / Cancel toolbar (identical style to BBox)
        self._edit_toolbar = QtWidgets.QWidget(
            self.window(),
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint,
        )
        self._edit_toolbar.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        tb_layout = QtWidgets.QHBoxLayout(self._edit_toolbar)
        tb_layout.setContentsMargins(8, 6, 8, 6)
        tb_layout.setSpacing(8)

        lbl = QtWidgets.QLabel("Edit Shape:")
        lbl.setStyleSheet("color: #e0a020; font-weight: bold; font-size: 9pt;")
        tb_layout.addWidget(lbl)

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setMinimumWidth(90)
        btn_save.setStyleSheet(
            "QPushButton { background: #2a5a2a; color: #80ff80; "
            "border: 1px solid #60c060; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #3a7a3a; }"
        )
        btn_save.clicked.connect(self._commit_obb_edit)
        tb_layout.addWidget(btn_save)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setMinimumWidth(90)
        btn_cancel.setStyleSheet(
            "QPushButton { background: #5a2a2a; color: #ff8080; "
            "border: 1px solid #c06060; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #7a3a3a; }"
        )
        btn_cancel.clicked.connect(self.cancel_edit)
        tb_layout.addWidget(btn_cancel)

        self._edit_toolbar.adjustSize()
        plot_global = self.plot_widget.mapToGlobal(
            self.plot_widget.rect().topRight()
        )
        tb = self._edit_toolbar
        tb.move(plot_global.x() - tb.width() - 12,
                plot_global.y() + 8)
        tb.show()

    def _obb_handle_positions(self):
        """Return the three handle positions (cx, axis_tip, width_tip) from current state."""
        import math
        cx_t, w_t, h_d    = self._edit_pts_t
        cy_d, angle_deg   = self._edit_pts_d
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        return [
            (cx_t,                    cy_d),                    # 0: centre
            (cx_t + cos_a * w_t,      cy_d + sin_a * w_t),     # 1: axis tip
            (cx_t - sin_a * h_d,      cy_d + cos_a * h_d),     # 2: width tip
        ]

    def _obb_polygon_pts(self):
        """Return the 5 closed-polygon points (xs, ys) for the current OBB."""
        import math
        cx_t, w_t, h_d  = self._edit_pts_t
        cy_d, angle_deg = self._edit_pts_d
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        offsets = [(w_t, h_d), (w_t, -h_d), (-w_t, -h_d), (-w_t, h_d), (w_t, h_d)]
        xs = [cx_t + cos_a * ox - sin_a * oy for ox, oy in offsets]
        ys = [cy_d + sin_a * ox + cos_a * oy for ox, oy in offsets]
        return xs, ys

    def _draw_obb_edit_handles(self) -> None:
        """Draw the OBB edit overlay: dashed orange polygon + 3 draggable TargetItems."""
        import math
        for item in self._edit_items:
            if hasattr(item, 'sigPositionChanged'):
                try:
                    item.sigPositionChanged.disconnect()
                except TypeError:
                    pass
            self.plot_widget.removeItem(item)
        self._edit_items.clear()

        orange_pen = pg.mkPen(color=(255, 140, 0), width=2,
                               style=QtCore.Qt.DashLine)

        # Dashed polygon outline (index 0)
        xs, ys = self._obb_polygon_pts()
        poly = pg.PlotCurveItem(xs, ys, pen=orange_pen)
        self.plot_widget.addItem(poly)
        self._edit_items.append(poly)

        # Three draggable TargetItems (indices 1, 2, 3)
        handle_styles = [
            # centre: square handle
            {'symbol': 's', 'size': 14},
            # axis tip: circle
            {'symbol': 'o', 'size': 14},
            # width tip: diamond
            {'symbol': 'd', 'size': 14},
        ]
        handle_positions = self._obb_handle_positions()

        for v_idx, ((ht, hd), style) in enumerate(zip(handle_positions, handle_styles)):
            target = pg.TargetItem(
                pos=(ht, hd),
                size=style['size'],
                symbol=style['symbol'],
                pen=pg.mkPen(color=(255, 255, 255), width=1),
                brush=pg.mkBrush(color=(255, 140, 0, 230)),
                movable=True,
            )

            def make_moved(vi):
                def on_moved():
                    pos = self._edit_items[1 + vi].pos()
                    pt, pd = pos.x(), pos.y()
                    cx_t, w_t, h_d  = self._edit_pts_t
                    cy_d, angle_deg = self._edit_pts_d
                    a = math.radians(angle_deg)
                    cos_a, sin_a = math.cos(a), math.sin(a)

                    if vi == 0:
                        # Centre moved — translate whole OBB
                        cx_t, cy_d = pt, pd
                    elif vi == 1:
                        # Axis tip moved — recompute w and angle from centre
                        dx = pt - cx_t
                        dy = pd - cy_d
                        new_w = math.hypot(dx, dy)
                        if new_w > 1e-6:
                            w_t = new_w
                            angle_deg = math.degrees(math.atan2(dy, dx))
                    elif vi == 2:
                        # Width tip moved — recompute h from perpendicular distance to axis
                        # Project (pt-cx, pd-cy) onto the perpendicular direction (-sin, cos)
                        dx = pt - cx_t
                        dy = pd - cy_d
                        new_h = abs(-sin_a * dx + cos_a * dy)  # dot with perp unit vector
                        if new_h > 1e-6:
                            h_d = new_h

                    self._edit_pts_t = [cx_t, w_t, h_d]
                    self._edit_pts_d = [cy_d, angle_deg]
                    self._update_obb_edit_outline()
                return on_moved

            target.sigPositionChanged.connect(make_moved(v_idx))
            self.plot_widget.addItem(target)
            self._edit_items.append(target)

    def _update_obb_edit_outline(self) -> None:
        """Redraw outline and reposition handles without recreating TargetItems."""
        import math
        if not self._edit_items:
            return

        # Update polygon (item 0)
        xs, ys = self._obb_polygon_pts()
        self._edit_items[0].setData(xs, ys)

        # Reposition the 3 TargetItems (items 1..3) without triggering on_moved
        handle_positions = self._obb_handle_positions()
        cx_t, w_t, h_d  = self._edit_pts_t
        cy_d, angle_deg = self._edit_pts_d
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)

        for v_idx, (ht, hd) in enumerate(handle_positions):
            target = self._edit_items[1 + v_idx]
            target.sigPositionChanged.disconnect()
            target.setPos(ht, hd)

            def make_moved(vi):
                def on_moved():
                    pos = self._edit_items[1 + vi].pos()
                    pt, pd = pos.x(), pos.y()
                    cx_t_, w_t_, h_d_  = self._edit_pts_t
                    cy_d_, angle_deg_  = self._edit_pts_d
                    a_ = math.radians(angle_deg_)
                    cos_a_ = math.cos(a_)
                    sin_a_ = math.sin(a_)
                    if vi == 0:
                        cx_t_, cy_d_ = pt, pd
                    elif vi == 1:
                        dx = pt - cx_t_
                        dy = pd - cy_d_
                        new_w = math.hypot(dx, dy)
                        if new_w > 1e-6:
                            w_t_ = new_w
                            angle_deg_ = math.degrees(math.atan2(dy, dx))
                    elif vi == 2:
                        dx = pt - cx_t_
                        dy = pd - cy_d_
                        new_h = abs(-sin_a_ * dx + cos_a_ * dy)
                        if new_h > 1e-6:
                            h_d_ = new_h
                    self._edit_pts_t = [cx_t_, w_t_, h_d_]
                    self._edit_pts_d = [cy_d_, angle_deg_]
                    self._update_obb_edit_outline()
                return on_moved

            target.sigPositionChanged.connect(make_moved(v_idx))

    def _commit_obb_edit(self) -> None:
        """Emit obb_edited and leave edit mode."""
        flat_idx         = self._edit_flat_idx
        cx_t, w_t, h_d  = self._edit_pts_t
        cy_d, angle_deg  = self._edit_pts_d
        self.cancel_edit()
        self.obb_edited.emit(flat_idx, cx_t, cy_d, w_t, h_d, angle_deg)

    # ------------------------------------------------------------------
    # KP / LINE shared shape-edit implementation
    # ------------------------------------------------------------------
    # Both types share the same vertex-drag mechanism.  The only
    # difference is the visual (dashed skeleton vs solid line) and the
    # signal emitted on commit.
    #
    # State layout (reuses existing _edit_pts_t / _edit_pts_d lists):
    #   _edit_pts_t  = list of vertex t-coordinates (mutable)
    #   _edit_pts_d  = list of vertex d-coordinates (mutable)
    # _edit_ann_type tells commit which signal to fire.

    def _enter_multipoint_edit_mode(self, flat_idx: int,
                                     ann_type: AnnType,
                                     pts_t: list, pts_d: list) -> None:
        """Shared entry point for KP and LINE shape editing."""
        self.cancel_edit()
        self._edit_mode        = True
        self._edit_flat_idx    = flat_idx
        self._edit_ann_type    = ann_type
        self._edit_pts_t       = list(pts_t)
        self._edit_pts_d       = list(pts_d)
        self._edit_drag_vertex = -1

        self._draw_multipoint_edit_handles()

        # Floating toolbar
        self._edit_toolbar = QtWidgets.QWidget(
            self.window(),
            QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint,
        )
        self._edit_toolbar.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)
        tb_layout = QtWidgets.QHBoxLayout(self._edit_toolbar)
        tb_layout.setContentsMargins(8, 6, 8, 6)
        tb_layout.setSpacing(8)

        type_label = "Keypoints" if ann_type == AnnType.KP else "Line"
        lbl = QtWidgets.QLabel(f"Edit {type_label}:")
        lbl.setStyleSheet("color: #e0a020; font-weight: bold; font-size: 9pt;")
        tb_layout.addWidget(lbl)

        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setMinimumWidth(90)
        btn_save.setStyleSheet(
            "QPushButton { background: #2a5a2a; color: #80ff80; "
            "border: 1px solid #60c060; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #3a7a3a; }"
        )
        commit_fn = self._commit_kp_edit if ann_type == AnnType.KP else self._commit_line_edit
        btn_save.clicked.connect(commit_fn)
        tb_layout.addWidget(btn_save)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setMinimumWidth(90)
        btn_cancel.setStyleSheet(
            "QPushButton { background: #5a2a2a; color: #ff8080; "
            "border: 1px solid #c06060; border-radius: 3px; padding: 4px 12px; font-size: 9pt; }"
            "QPushButton:hover { background: #7a3a3a; }"
        )
        btn_cancel.clicked.connect(self.cancel_edit)
        tb_layout.addWidget(btn_cancel)

        self._edit_toolbar.adjustSize()
        plot_global = self.plot_widget.mapToGlobal(
            self.plot_widget.rect().topRight()
        )
        tb = self._edit_toolbar
        tb.move(plot_global.x() - tb.width() - 12,
                plot_global.y() + 8)
        tb.show()

    def enter_kp_edit_mode(self, flat_idx: int,
                            pts_t: list, pts_d: list) -> None:
        """Activate vertex-edit mode for a Keypoints annotation."""
        self._enter_multipoint_edit_mode(flat_idx, AnnType.KP, pts_t, pts_d)

    def enter_line_edit_mode(self, flat_idx: int,
                              pts_t: list, pts_d: list) -> None:
        """Activate vertex-edit mode for a Line annotation."""
        self._enter_multipoint_edit_mode(flat_idx, AnnType.LINE, pts_t, pts_d)

    def _draw_multipoint_edit_handles(self) -> None:
        """Draw edit overlay: connector preview + one TargetItem per vertex."""
        for item in self._edit_items:
            if hasattr(item, 'sigPositionChanged'):
                try:
                    item.sigPositionChanged.disconnect()
                except TypeError:
                    pass
            self.plot_widget.removeItem(item)
        self._edit_items.clear()

        pts_t = self._edit_pts_t
        pts_d = self._edit_pts_d
        orange_pen = pg.mkPen(color=(255, 140, 0), width=2,
                               style=QtCore.Qt.DashLine)

        # Connector line / skeleton (index 0).
        connector = pg.PlotCurveItem(pts_t, pts_d, pen=orange_pen)
        self.plot_widget.addItem(connector)
        self._edit_items.append(connector)

        # One TargetItem per vertex (indices 1..N).
        for v_idx in range(len(pts_t)):
            target = pg.TargetItem(
                pos=(pts_t[v_idx], pts_d[v_idx]),
                size=14,
                symbol='o',
                pen=pg.mkPen(color=(255, 255, 255), width=1),
                brush=pg.mkBrush(color=(255, 140, 0, 230)),
                movable=True,
            )

            def make_moved(vi):
                def on_moved():
                    pos = self._edit_items[1 + vi].pos()
                    self._edit_pts_t[vi] = pos.x()
                    self._edit_pts_d[vi] = pos.y()
                    self._update_multipoint_edit_connector()
                return on_moved

            target.sigPositionChanged.connect(make_moved(v_idx))
            self.plot_widget.addItem(target)
            self._edit_items.append(target)

    def _update_multipoint_edit_connector(self) -> None:
        """Redraw the dashed connector without recreating TargetItems."""
        if not self._edit_items:
            return
        # item 0 = connector line
        self._edit_items[0].setData(self._edit_pts_t, self._edit_pts_d)
        # Reposition TargetItems (items 1..N) without triggering on_moved
        for vi in range(len(self._edit_pts_t)):
            target = self._edit_items[1 + vi]
            target.sigPositionChanged.disconnect()
            target.setPos(self._edit_pts_t[vi], self._edit_pts_d[vi])

            def make_moved(v):
                def on_moved():
                    pos = self._edit_items[1 + v].pos()
                    self._edit_pts_t[v] = pos.x()
                    self._edit_pts_d[v] = pos.y()
                    self._update_multipoint_edit_connector()
                return on_moved

            target.sigPositionChanged.connect(make_moved(vi))

    def _commit_kp_edit(self) -> None:
        """Emit kp_edited and leave edit mode."""
        flat_idx = self._edit_flat_idx
        pts_t    = list(self._edit_pts_t)
        pts_d    = list(self._edit_pts_d)
        self.cancel_edit()
        self.kp_edited.emit(flat_idx, pts_t, pts_d)

    def _commit_line_edit(self) -> None:
        """Emit line_edited and leave edit mode."""
        flat_idx = self._edit_flat_idx
        pts_t    = list(self._edit_pts_t)
        pts_d    = list(self._edit_pts_d)
        self.cancel_edit()
        self.line_edited.emit(flat_idx, pts_t, pts_d)

    def _on_roi_right_clicked(self, roi: AnnotationROI) -> None:
        """Right-click on a BBox ROI — show context menu."""
        self._show_context_menu(roi.ann_index, AnnType.BBOX)

    def _on_obb_right_clicked(self, item: OBBCurveItem) -> None:
        """Right-click on an OBB polygon — show context menu."""
        self._show_context_menu(item.ann_index, AnnType.OBB)

    def _on_kp_right_clicked(self, item) -> None:
        """Right-click on a KP dot or connector — show context menu."""
        self._show_context_menu(item.ann_index, AnnType.KP)

    def _on_line_right_clicked(self, item) -> None:
        """Right-click on a LINE polyline or dot — show context menu."""
        self._show_context_menu(item.ann_index, AnnType.LINE)

    def _show_context_menu(self, idx: int, ann_type=None) -> None:
        """Context menu for any annotation type."""
        from PyQt5.QtGui import QCursor
        menu = QtWidgets.QMenu()
        edit_event_action  = menu.addAction("Edit Event")        # ID / comment
        edit_shape_action  = menu.addAction("Edit Shape")        # vertex editing
        remove_action      = menu.addAction("Remove Event")
        menu.addSeparator()
        spec_action        = menu.addAction("Spectrogram")
        spectral_action    = menu.addAction("Spectral Analysis")
        signal_action      = menu.addAction("Signal (time domain)")
        sig_freq_action    = menu.addAction("Signal (frequency domain)")
        sig_env_action     = menu.addAction("Signal (envelope)")
        sig_phase_action   = menu.addAction("Signal (phase)")
        velocity_action    = menu.addAction("Estimate Velocity")
        action = menu.exec_(QCursor.pos())
        if action == edit_event_action:
            self.roi_edit_requested.emit(idx)
        elif action == edit_shape_action:
            self.roi_edit_shape_requested.emit(idx)
        elif action == remove_action:
            self.roi_remove_requested.emit(idx)
        elif action == spec_action:
            self.roi_spectrogram_requested.emit(idx)
        elif action == spectral_action:
            self.roi_spectral_requested.emit(idx)
        elif action == signal_action:
            self.roi_signal_requested.emit(idx)
        elif action == sig_freq_action:
            self.roi_signal_freq_requested.emit(idx)
        elif action == sig_env_action:
            self.roi_signal_env_requested.emit(idx)
        elif action == sig_phase_action:
            self.roi_signal_phase_requested.emit(idx)
        elif action == velocity_action:
            self.roi_velocity_requested.emit(idx)

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def show_overlay(self, *lines: str) -> None:
        self._overlay_label.setText("\n".join(lines))
        self._overlay.setVisible(True)
        # Use repaint() only — never processEvents() here
        self._overlay.repaint()

    def hide_overlay(self) -> None:
        self._overlay.setVisible(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_display(
        self,
        dataset: DASDataset,
        vmin: float,
        vmax: float,
        fmin: float = None,
        fmax: float = None,
        tr_override: np.ndarray = None,
        envelope: bool = False,
    ) -> None:
        """
        Single atomic operation: store dataset, optionally filter, render.

        Parameters
        ----------
        dataset : DASDataset
        vmin, vmax : float  colormap levels
        fmin, fmax : float, optional  bandpass cutoffs; skip filter if None
        tr_override : np.ndarray, optional
            If provided, display this array directly (e.g. FK-filtered data)
            instead of computing a new filter.
        envelope : bool
            If True apply Hilbert transform (amplitude envelope) before display.
        """
        self.dataset = dataset

        if tr_override is not None:
            tr = tr_override.astype(np.float32)
        elif fmin is not None and fmax is not None:
            nyq = dataset.fs_hz / 2.0
            if fmin > 0 and fmax > fmin and fmax < nyq:
                import scipy.signal as sp
                sos = sp.butter(5, [fmin / nyq, fmax / nyq],
                                btype="bandpass", output="sos")
                tr = sp.sosfiltfilt(sos, dataset.tr, axis=1).astype(np.float32)
            else:
                tr = dataset.tr
        else:
            tr = dataset.tr

        if envelope:
            tr = self.compute_envelope(tr)

        # Update axes
        x0, x1 = float(dataset.time_s[0]), float(dataset.time_s[-1])
        y0, y1 = float(dataset.dist_m[0]),  float(dataset.dist_m[-1])

        # Render — disconnect sigImageChanged so pyqtgraph never auto-reads
        # the raw data range before we set our levels
        self.image_item.sigImageChanged.disconnect(self.histogram.item.imageChanged)
        self.image_item.setLevels([vmin, vmax])
        self.image_item.setImage(tr, autoLevels=False)
        self.image_item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self.image_item.sigImageChanged.connect(self.histogram.item.imageChanged)
        self._tr_display = tr   # keep reference for analysis windows

        # Update histogram manually
        self.histogram.item.imageChanged(autoLevel=False)
        self.histogram.setLevels(vmin, vmax)
        p999 = float(np.percentile(np.abs(tr), 99.9))
        self.histogram.setHistogramRange(-p999, p999, padding=0.05)

        # Update plot axes AFTER image is fully set
        self.plot_widget.setLimits(
            xMin=x0, xMax=x1,
            yMin=y0, yMax=y1,
        )
        self.plot_widget.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def apply_theme(self) -> None:
        """
        Re-apply the current theme's colours to elements pyqtgraph does NOT
        pick up from the Qt stylesheet: plot background, axis text colour,
        crosshair pen, and any bbox ROIs already drawn on this widget.
        Called by MainWindow whenever the user switches Dark/Light theme.
        """
        th = theme.current()
        self.plot_widget.setBackground(th["pg_background"])
        self.histogram.setBackground(th["pg_background"])
        # HistogramLUTWidget's internal ViewBox (the histogram plot area
        # itself, where the value-distribution curve is drawn) has its own
        # background independent of setBackground() on the outer widget —
        # without this it stays whatever colour it was created with.
        self.histogram.item.vb.setBackgroundColor(th["pg_background"])
        axis_pen = pg.mkPen(color=th["pg_axis_text"])
        for axis_name in ("bottom", "left"):
            axis = self.plot_widget.getPlotItem().getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)

        cross_pen = pg.mkPen(color=th["pg_crosshair"], width=1, style=QtCore.Qt.DashLine)
        self._vline.setPen(cross_pen)
        self._hline.setPen(cross_pen)

        for roi, text, idx in self._annotation_rois:
            roi.setPen(_bbox_pen())

    def display_array(self, tr: np.ndarray, vmin: float, vmax: float) -> None:
        """Re-render with a new array (e.g. after interactive Apply filter)."""
        self._set_rgb_mode(False)
        self.image_item.sigImageChanged.disconnect(self.histogram.item.imageChanged)
        self.image_item.setLevels([vmin, vmax])
        self.image_item.setImage(tr, autoLevels=False)
        self.image_item.sigImageChanged.connect(self.histogram.item.imageChanged)

        self.histogram.item.imageChanged(autoLevel=False)
        self.histogram.setLevels(vmin, vmax)
        p999 = float(np.percentile(np.abs(tr), 99.9))
        self.histogram.setHistogramRange(-p999, p999, padding=0.05)
        self._tr_display = tr   # keep reference for analysis windows

    def display_rgb_array(self, rgb: np.ndarray, dataset: DASDataset) -> None:
        """
        Display a composited RGB array (n_dist, n_time, 3) uint8 directly,
        bypassing the scalar colormap/histogram (not meaningful for an RGB
        composite). Time/distance axes are set from dataset, same convention
        as load_and_display.
        """
        self._set_rgb_mode(True)
        self.dataset = dataset

        x0, x1 = float(dataset.time_s[0]), float(dataset.time_s[-1])
        y0, y1 = float(dataset.dist_m[0]),  float(dataset.dist_m[-1])

        # IMPORTANT: pyqtgraph's autoDownsample (enabled by default on
        # image_item for scalar Raw/F-K waterfalls) can internally produce a
        # FLOAT intermediate array when binning down a large image for
        # display. With levels=None (required for our pre-scaled uint8 RGB,
        # see below), pyqtgraph's paint() then raises "levels argument is
        # required for float input types" — caught internally by Qt's paint
        # loop, so nothing crashes but NOTHING gets drawn either: a fully
        # black tab with no visible error. Disabling autoDownsample for the
        # RGB image keeps it strictly uint8 end-to-end.
        self.image_item.setAutoDownsample(False)

        # Clear any levels left over from a previous scalar display (e.g.
        # [0, 12] from Raw/F-K). For an RGB uint8 image, leftover levels are
        # applied as a per-channel contrast rescale and can saturate the
        # whole image to white. levels=None tells pyqtgraph "use the array
        # values as-is" (already normalised to 0-255 by compute_rgb_composite).
        #
        # sigImageChanged must be disconnected from the histogram first: the
        # histogram's imageChanged() handler calls getLevels() and crashes on
        # an RGB image with levels=None (it expects a 2-value scalar levels
        # tuple). The histogram is hidden in RGB mode anyway (_set_rgb_mode),
        # so it doesn't need to track this image. Guarded so repeated calls
        # to display_rgb_array (e.g. re-applying RGB params) don't try to
        # disconnect an already-disconnected signal.
        if not self._rgb_signal_disconnected:
            self.image_item.sigImageChanged.disconnect(self.histogram.item.imageChanged)
            self._rgb_signal_disconnected = True
        self.image_item.setImage(rgb, autoLevels=False)
        # setImage(levels=None) does NOT clear pre-existing levels (pyqtgraph
        # quirk) — an explicit setLevels(None) call afterwards is required.
        self.image_item.setLevels(None)
        self.image_item.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
        self._tr_display = rgb

        self.plot_widget.setLimits(xMin=x0, xMax=x1, yMin=y0, yMax=y1)
        self.plot_widget.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def _set_rgb_mode(self, rgb_mode: bool) -> None:
        """Show/hide the histogram+colormap column; irrelevant in RGB mode."""
        self.histogram.setVisible(not rgb_mode)
        self.combo_cmap.setVisible(not rgb_mode)

    def get_displayed_array(self) -> np.ndarray:
        """Return the array currently shown (filtered or raw). May be None."""
        return self._tr_display

    def compute_bandpass(self, fmin: float, fmax: float) -> np.ndarray:
        import scipy.signal as sp
        fs  = self.dataset.fs_hz
        nyq = fs / 2.0
        sos = sp.butter(5, [fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
        return sp.sosfiltfilt(sos, self.dataset.tr, axis=1).astype(np.float32)

    def compute_envelope(self, tr: np.ndarray) -> np.ndarray:
        """
        Compute the Hilbert envelope of tr with caching and next_fast_len
        optimisation.  Returns np.float32 array of same shape as tr.
        """
        # Cache hit: same source array (checked by id)
        if self._tr_envelope is not None and self._tr_envelope_src is tr:
            return self._tr_envelope

        import scipy.signal as sp
        from scipy.fft import next_fast_len
        n   = tr.shape[1]
        nfft = next_fast_len(n)
        env = np.abs(sp.hilbert(tr, N=nfft, axis=1)[:, :n]).astype(np.float32)

        self._tr_envelope     = env
        self._tr_envelope_src = tr
        return env

    def apply_bandpass(self, fmin: float, fmax: float,
                       envelope: bool = False) -> None:
        if self.dataset is None:
            return
        nyq = self.dataset.fs_hz / 2.0
        if fmin <= 0 or fmax <= fmin or fmax >= nyq:
            return
        vmin, vmax = self.histogram.getLevels()
        tr_filt = self.compute_bandpass(fmin, fmax)
        if envelope:
            tr_filt = self.compute_envelope(tr_filt)
        self.display_array(tr_filt, vmin=float(vmin), vmax=float(vmax))

    def apply_color_levels(self, vmin: float, vmax: float) -> None:
        if vmin >= vmax:
            return
        self.histogram.setLevels(vmin, vmax)

    def get_color_levels(self):
        return self.histogram.getLevels()

    def apply_time_range(self, t0: float, t1: float) -> None:
        self.plot_widget.setXRange(t0, t1, padding=0)

    def apply_distance_range(self, d0: float, d1: float) -> None:
        self.plot_widget.setYRange(d0, d1, padding=0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_context_menu(self) -> None:
        """
        Customise the right-click context menu of the waterfall plot:
        - Rename 'View All' → 'Full View'
        - Remove 'Mouse Mode' and 'Export'
        - Move 'X axis' and 'Y axis' into Plot Options (ctrlMenu)
        - Remove 'Average', 'Alpha' and 'Points' from Plot Options
        """
        vb     = self.plot_widget.getPlotItem().vb
        pi     = self.plot_widget.getPlotItem()
        vb_menu   = vb.menu
        ctrl_menu = pi.ctrlMenu

        # 1) Rename 'View All' → 'Full View'
        for a in vb_menu.actions():
            if a.text() == "View All":
                a.setText("Full View")
                break

        # 2) Remove 'Mouse Mode' from ViewBox menu
        for a in vb_menu.actions():
            if a.text() == "Mouse Mode":
                vb_menu.removeAction(a)
                break

        # 2b) Remove 'Export' from the GraphicsScene context menu
        scene = self.plot_widget.scene()
        scene.contextMenu = [a for a in scene.contextMenu
                             if "Export" not in a.text()]

        # 3) Move 'X axis' and 'Y axis' into Plot Options (ctrlMenu), at the top
        insert_before = ctrl_menu.actions()[0] if ctrl_menu.actions() else None
        for axis_text in ("X axis", "Y axis"):
            for a in list(vb_menu.actions()):
                if a.text() == axis_text:
                    vb_menu.removeAction(a)
                    if insert_before:
                        ctrl_menu.insertAction(insert_before, a)
                    else:
                        ctrl_menu.addAction(a)
                    break

        # 4) Remove 'Average', 'Alpha', 'Points' from Plot Options
        for a in list(ctrl_menu.actions()):
            if a.text() in ("Average", "Alpha", "Points"):
                ctrl_menu.removeAction(a)

    def _apply_colormap(self, name: str) -> None:
        try:
            cmap = pg.colormap.get(name, source="matplotlib")
        except Exception:
            cmap = pg.colormap.get(name)
        self.histogram.gradient.setColorMap(cmap)

    def _on_cmap_changed(self, index: int) -> None:
        _, mpl_name = COLORMAPS[index]
        self._apply_colormap(mpl_name)

    def _on_mouse_moved(self, scene_pos) -> None:
        if self.dataset is None:
            self.cursor_info.emit("")
            return

        view_pos = self.plot_widget.getPlotItem().vb.mapSceneToView(scene_pos)
        t, d = view_pos.x(), view_pos.y()

        # Update crosshair position whenever annotation mode is active
        if self._annotation_mode:
            self._vline.setPos(t)
            self._hline.setPos(d)

        time_s = self.dataset.time_s
        dist_m = self.dataset.dist_m

        if t < time_s[0] or t > time_s[-1] or d < dist_m[0] or d > dist_m[-1]:
            self.cursor_info.emit("")
            return

        idx_t = max(0, min(int(np.argmin(np.abs(time_s - t))), self.dataset.n_time - 1))
        idx_d = max(0, min(int(np.argmin(np.abs(dist_m - d))), self.dataset.n_dist - 1))

        value = float(self.dataset.tr[idx_d, idx_t])

        # Map internal units convention to the symbol shown to the user
        units_symbol = {
            "DC":         "DC",
            "strain":     "\u03b5",
            "nanostrain": "n\u03b5",
        }.get(self.dataset.units, self.dataset.units or "?")

        # Build UTC timestamp string if the dataset has a start time.
        utc_str = ""
        if self.dataset.start_datetime_utc is not None:
            import datetime as _dt
            utc_ts = self.dataset.start_datetime_utc + _dt.timedelta(seconds=float(t - time_s[0]))
            utc_str = f"    Timestamp [UTC]: {utc_ts.strftime('%H:%M:%S')}.{utc_ts.microsecond // 1000:03d}"

        self.cursor_info.emit(
            f"Time [s]: {t:.3f}{utc_str}    "
            f"Distance [m]: {d:.1f}    "
            f"Value ({units_symbol}): {value:.4g}"
        )
