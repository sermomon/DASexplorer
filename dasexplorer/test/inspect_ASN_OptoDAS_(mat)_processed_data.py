import scipy.io as sio
import numpy as np

path = "C:/Lab/projects/DASexplorer/dasexplorer/datasets/optodas/svalbard-2020/20200627_052441_ch10001_to_ch15000_whale_raw_L160s.mat"
mat = sio.loadmat(path)
data = np.asarray(mat["data"])

print(f"dtype: {data.dtype}")
print(f"shape: {data.shape}")
print(f"min: {data.min()}")
print(f"max: {data.max()}")
print(f"mean abs: {np.abs(data).mean()}")
print(f"units: {np.asarray(mat['info_units']).ravel()}")