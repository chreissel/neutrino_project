import os
import numpy as np
import gzip
import h5py
import pickle
import glob

print("did something")

# --- Generalizes code to work with compressed and uncompressed pickles ---
def open_pickle(fname):
    if fname.endswith(".gz"):
        with gzip.open(fname, "rb") as f:
            return pickle.load(f)
    else:
        with open(fname, "rb") as f:
            return pickle.load(f)

pickle_paths = ("/home/lfd34/project/project8/ssm/neutrino_project/data_preprocessing/pickles_20_bins/*.pkl")
output_dir = ("/home/lfd34/project/project8/ssm/neutrino_project/data_preprocessing/shards_20bins")

events_per_shard = 5000

os.makedirs(output_dir, exist_ok=True)

file_list = sorted(glob.glob(pickle_paths))

# --- Get columns, shapes, and total rows ---
df = open_pickle(file_list[0])
columns = df.columns
shapes = {}
for col in columns:
    arr = df[col].to_numpy()
    if isinstance(arr[0], (list, tuple, np.ndarray)):
        arr0 = np.stack(arr)
        shapes[col] = arr0.shape[1:]
    else:
        shapes[col] = ()

print("did something 2")

# --- Get number of events per pickle file ---
pickle_sizes = []
total_events = 0

for fname in file_list:
    df = open_pickle(fname)
    num_events = len(df)
    pickle_sizes.append(num_events)
    total_events += num_events

file_sizes = dict(zip(file_list, pickle_sizes))
print(f"Total events: {total_events}")

# --- Create shards as lists of pickle filenames ---
shards = []
current_shard = []
current_num_events = 0

for fname in file_list:
    num_events = file_sizes[fname]
    current_shard.append(fname)
    current_num_events += num_events

    if current_num_events >= events_per_shard:
        shards.append(current_shard)
        current_shard = []
        current_num_events = 0

if current_shard:  # make sure last shard is included
    shards.append(current_shard)

print(f"Built {len(shards)} shards")

# --- Create hdf5 file for each shard, create datasets and fill them ---
for shard_id, shard_files in enumerate(shards):
    shard_path = os.path.join(output_dir, f"shard_{shard_id:05d}.hdf5")
    if os.path.exists(shard_path):
        print(f"[skip] {shard_path}")
        continue

    total_rows = sum(file_sizes[fname] for fname in shard_files)
    print(f"[write] {shard_path} ({len(shard_files)} pickles, {total_rows} events)")

    with h5py.File(shard_path, "w") as h5f:
        dsets = {}
        for col in columns:
            shape = (total_rows,) + shapes[col]
            dsets[col] = h5f.create_dataset(
                col,
                shape=shape,
                dtype=np.float32)

        row_idx = 0
        for fname in shard_files:
            df = open_pickle(fname)
            rows = len(df)
            for col in columns:
                arr = df[col].to_numpy()
                if isinstance(arr[0], (list, tuple, np.ndarray)):
                    arr = np.stack(arr)
                arr = arr.astype(np.float32)
                dsets[col][row_idx : row_idx + rows] = arr
            row_idx += rows

print("All shards written successfully.")
