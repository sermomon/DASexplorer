"""
Centralised theme system for DAS Explorer (Dark / Light).

Two things need to change together when the theme changes:
  1. The Qt stylesheet (panels, buttons, menus, etc.) — a plain CSS string.
  2. pyqtgraph plot colours (background, axes/text, and every hardcoded
     pen used for lines/markers across waterfall.py and analysis_dialogs.py),
     since pyqtgraph does NOT pick these up from a Qt stylesheet at all.

THEME is a module-level dict so other modules (waterfall.py,
analysis_dialogs.py) can read current colours via theme.current() without
import cycles, and so newly-created dialogs always pick up the active theme.
"""

from PyQt5 import QtCore

DARK = {
    "name": "dark",
    "qt_bg":        "#1e1e1e",
    "qt_bg_alt":    "#252525",
    "qt_panel":     "#2d2d2d",
    "qt_border":    "#3c3c3c",
    "qt_border_hi": "#5a5a5a",
    "qt_text":      "#e0e0e0",
    "qt_text_dim":  "#cfcfcf",
    "qt_accent":    "#4a9eff",
    "qt_select_bg": "#3a6ea5",
    "qt_select_hi": "#2d4a6a",
    "qt_tab_bg":    "#2a2a2a",
    "qt_tab_hover": "#333333",
    "qt_status_bg": "#1a1a1a",
    "qt_scroll":    "#4a4a4a",
    # pyqtgraph
    "pg_background": "k",        # black
    "pg_foreground":  "d",       # pyqtgraph default light grey/white
    "pg_axis_text":   "#cfcfcf",
    "pg_line_main":   (180, 180, 180),   # neutral line colour (was white/grey)
    "pg_line_avg":    (255, 255, 255),   # "average" emphasis line (Spectral Analysis etc.)
    "pg_crosshair":   (255, 220, 0),
    "pg_bbox":        (255, 220, 0),
    "pg_bbox_sel":    (255, 120, 0),
    "pg_label_bg":    "rgba(0,0,0,170)",
}

LIGHT = {
    "name": "light",
    "qt_bg":        "#f5f5f5",
    "qt_bg_alt":    "#ebebeb",
    "qt_panel":     "#ffffff",
    "qt_border":    "#c8c8c8",
    "qt_border_hi": "#a0a0a0",
    "qt_text":      "#1e1e1e",
    "qt_text_dim":  "#3a3a3a",
    "qt_accent":    "#1f6fd6",
    "qt_select_bg": "#a9cdf0",
    "qt_select_hi": "#cfe2f5",
    "qt_tab_bg":    "#e4e4e4",
    "qt_tab_hover": "#d6d6d6",
    "qt_status_bg": "#e8e8e8",
    "qt_scroll":    "#b5b5b5",
    # pyqtgraph
    "pg_background": "w",        # white
    "pg_foreground":  "k",       # black axes/text
    "pg_axis_text":   "#3a3a3a",
    "pg_line_main":   (60, 60, 60),      # dark neutral line (was light grey)
    "pg_line_avg":    (0, 0, 0),         # "average" emphasis line — BLACK in light theme
    "pg_crosshair":   (200, 130, 0),
    "pg_bbox":        (200, 130, 0),
    "pg_bbox_sel":    (200, 90, 0),
    "pg_label_bg":    "rgba(255,255,255,200)",
}

THEMES = {"dark": DARK, "light": LIGHT}

# Mutable "current theme" state, read by waterfall.py / analysis_dialogs.py.
_current = DARK


def current() -> dict:
    return _current


def set_current(name: str) -> dict:
    global _current
    _current = THEMES.get(name, DARK)
    return _current


def build_stylesheet(theme: dict) -> str:
    return f"""
QMainWindow, QWidget {{
    background-color: {theme['qt_bg']};
    color: {theme['qt_text']};
}}

QGroupBox {{
    border: 1px solid {theme['qt_border']};
    border-radius: 4px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: {theme['qt_text_dim']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}}

QPushButton {{
    background-color: {theme['qt_panel']};
    border: 1px solid {theme['qt_border']};
    border-radius: 4px;
    padding: 5px 10px;
    color: {theme['qt_text']};
}}

QPushButton:hover {{
    background-color: {theme['qt_tab_hover']};
    border-color: {theme['qt_border_hi']};
}}

QPushButton:pressed {{
    background-color: {theme['qt_bg_alt']};
}}

QComboBox {{
    background-color: {theme['qt_panel']};
    border: 1px solid {theme['qt_border']};
    border-radius: 4px;
    padding: 3px 6px;
    color: {theme['qt_text']};
}}

QComboBox QAbstractItemView {{
    background-color: {theme['qt_panel']};
    color: {theme['qt_text']};
    selection-background-color: {theme['qt_select_bg']};
}}

QLabel {{
    color: {theme['qt_text_dim']};
}}

QCheckBox {{
    color: {theme['qt_text_dim']};
}}

QStatusBar {{
    background-color: {theme['qt_status_bg']};
    color: {theme['qt_text_dim']};
}}

QMenuBar {{
    background-color: {theme['qt_bg']};
    color: {theme['qt_text']};
}}

QMenuBar::item:selected {{
    background-color: {theme['qt_tab_hover']};
}}

QMenu {{
    background-color: {theme['qt_panel']};
    color: {theme['qt_text']};
    border: 1px solid {theme['qt_border']};
}}

QMenu::item:selected {{
    background-color: {theme['qt_select_bg']};
}}

QTabWidget::pane {{
    border: 1px solid {theme['qt_border']};
    background-color: {theme['qt_bg']};
}}

QTabBar::tab {{
    background-color: {theme['qt_tab_bg']};
    color: {theme['qt_text']};
    padding: 5px 18px;
    border: 1px solid {theme['qt_border']};
    border-bottom: none;
    min-width: 48px;
}}

QTabBar::tab:selected {{
    background-color: {theme['qt_bg']};
    border-bottom: 2px solid {theme['qt_accent']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {theme['qt_tab_hover']};
}}

QListWidget {{
    background-color: {theme['qt_bg_alt']};
    border: 1px solid {theme['qt_border']};
    color: {theme['qt_text']};
}}

QListWidget::item:selected {{
    background-color: {theme['qt_select_bg']};
    color: {theme['qt_text']};
}}

QListWidget::item:hover {{
    background-color: {theme['qt_select_hi']};
}}

QScrollBar:vertical {{
    background-color: {theme['qt_bg']};
    width: 10px;
}}

QScrollBar::handle:vertical {{
    background-color: {theme['qt_scroll']};
    border-radius: 4px;
    min-height: 20px;
}}

QSplitter::handle {{
    background-color: {theme['qt_border']};
}}

QSplitter::handle:hover {{
    background-color: {theme['qt_border_hi']};
}}

QTableWidget {{
    background-color: {theme['qt_bg_alt']};
    color: {theme['qt_text']};
    gridline-color: {theme['qt_border']};
}}

QHeaderView::section {{
    background-color: {theme['qt_panel']};
    color: {theme['qt_text']};
    border: 1px solid {theme['qt_border']};
}}

QDoubleSpinBox, QSpinBox {{
    background-color: {theme['qt_panel']};
    border: 1px solid {theme['qt_border']};
    border-radius: 3px;
    color: {theme['qt_text']};
    padding: 2px 4px;
}}
"""
