"""
DAS Explorer - entry point.

Run with:
    python -m dasexplorer
or after pip install:
    dasexplorer
"""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

import sys

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui

from dasexplorer.gui.main_window import MainWindow
from dasexplorer.gui import theme as theme_mod
from dasexplorer.core.config import get_ui_defaults
from dasexplorer.version import __version__

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

DARK_STYLESHEET = theme_mod.build_stylesheet(theme_mod.DARK)


def main():
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "IGIC-UPV.DASExplorer." + __version__
            )
        except Exception:
            pass

    app = QtWidgets.QApplication(sys.argv)

    ui_cfg = get_ui_defaults()
    saved_theme = str(ui_cfg.get("theme", "dark")).lower()
    theme = theme_mod.set_current(saved_theme if saved_theme in theme_mod.THEMES else "dark")
    pg.setConfigOption("background", theme["pg_background"])
    pg.setConfigOption("foreground", theme["pg_foreground"])
    app.setStyleSheet(theme_mod.build_stylesheet(theme))

    # Icon path relative to the installed package location
    icon_dir = os.path.dirname(os.path.abspath(__file__))
    taskbar_icon_path = os.path.join(icon_dir, "icons", "icon_2.ico")
    if os.path.isfile(taskbar_icon_path):
        app.setWindowIcon(QtGui.QIcon(taskbar_icon_path))

    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
