
#%%

import os
import dasexplorer
import numpy as np

from dasexplorer.api import DASdataset, DASannotations
from dasexplorer.api import DASxarray

# %%

FILE_PATH = "C:/Lab/projects/DASexplorer/dasexplorer/datasets/hdas/upv-apl-alme/2026_01_01_04h47m30s_HDAS_2DRawData_Strain.bin"

ds = DASdataset()
