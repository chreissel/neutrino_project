import os
import numpy as np
import pandas as pd
import gzip, h5py
import pickle
import glob

file_list = glob.glob('/n/holystore01/LABS/iaifi_lab/Lab/hbinney/ssm_files/Project8Sims_noNoise_dataMode_100k_newmetadata/*.pkl.gz')
hdf5_path = '/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/combined_data_v2.hdf5'

if os.path.exists(hdf5_path):
    os.remove(hdf5_path)

# --- First pass: get columns, shapes, and total rows ---
with gzip.open(file_list[0], 'rb') as f:
    df = pickle.load(f)
columns = df.columns
shapes = {}
for col in columns:
    arr = df[col].to_numpy()
    if isinstance(arr[0], (list, tuple, np.ndarray)):
        arr0 = np.stack(arr)
        shapes[col] = arr0.shape[1:]
    else:
        shapes[col] = ()

total_rows = 0
for fname in file_list:
    with gzip.open(fname, 'rb') as f:
        df = pickle.load(f)
    total_rows += len(df)

# --- Second pass: create datasets and fill them ---
with h5py.File(hdf5_path, 'w') as h5f:
    dsets = {}
    for col in columns:
        shape = (total_rows,) + shapes[col]
        dsets[col] = h5f.create_dataset(col, shape=shape, dtype=np.float32)
    row_idx = 0
    for fname in file_list:
        with gzip.open(fname, 'rb') as f:
            df = pickle.load(f)
        rows = len(df)
        for col in columns:
            arr = df[col].to_numpy()
            if isinstance(arr[0], (list, tuple, np.ndarray)):
                arr = np.stack(arr).astype(np.float32)
            else:
                arr = arr.astype(np.float32)
            dsets[col][row_idx:row_idx+rows] = arr
        row_idx += rows

print(f'Saved all rows for each column as a single dataset in {hdf5_path}')
