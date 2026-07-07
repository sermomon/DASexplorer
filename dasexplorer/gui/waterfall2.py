"""
Waterfall display widget for DASexplorer Live Mode
"""

import json
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui

from dasexplorer.core.data_model import DASDataset
from dasexplorer.core.annotations import AnnType
from dasexplorer.gui import theme

# 2D