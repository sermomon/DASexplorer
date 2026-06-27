"""
Custom QTabBar for DAS Explorer.

Renders the "RGB" tab with each letter in its own colour (R=red, G=green,
B=blue), perfectly centred regardless of the tab width.
"""

from PyQt5 import QtWidgets, QtGui, QtCore

RGB_LETTERS = [
    ("R", QtGui.QColor("#ff4444")),
    ("G", QtGui.QColor("#44cc44")),
    ("B", QtGui.QColor("#4488ff")),
]

# Index of the RGB tab (0-based: Raw=0, F-K=1, RGB=2, Live=3)
RGB_TAB_INDEX = 2


class DASTabBar(QtWidgets.QTabBar):
    """
    QTabBar subclass that draws the RGB tab with coloured letters,
    perfectly centred, by overriding paintEvent.

    The tab text is set to spaces so the base class draws nothing visible;
    we then paint the R/G/B letters ourselves with their correct colours.
    """

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)

        rect = self.tabRect(RGB_TAB_INDEX)
        fm   = painter.fontMetrics()

        total_w = sum(fm.horizontalAdvance(l) for l, _ in RGB_LETTERS)
        x = rect.x() + (rect.width() - total_w) // 2
        y = rect.y() + (rect.height() + fm.ascent() - fm.descent()) // 2

        for letter, color in RGB_LETTERS:
            painter.setPen(color)
            painter.drawText(x, y, letter)
            x += fm.horizontalAdvance(letter)

        painter.end()
