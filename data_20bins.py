import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
import numpy as np
import torch
from noise import noise_model, bandpass_filter
from utils import fft_func_IQ_complex_channels
import os
import glob

class Project8Sim(Dataset):
    def __init__(self, shard_paths, inputs, variables, observables,
                 cutoff=4000, norm=True, noise_const=1, apply_filter=False):

        self.shard_paths = shard_paths
        self.inputs = inputs
        self.variables = variables
        self.observables = observables
        self.cutoff = cutoff
        self.norm = norm
        self.apply_filter = apply_filter
        self.noise_const = noise_const

        # --- Create global index for sharded events ---
        self.index_map = []
        self.shard_lengths = []
        for shard_id, path in enumerate(shard_paths):
            with h5py.File(path, 'r') as f:
                num_rows = f[inputs[0]].shape[0]
                self.shard_lengths.append(num_rows)
                self.index_map.extend([(shard_id, i) for i in range(num_rows)])

        # --- Compute mean/std for variables across all events ---
        if norm:
            ys = []
            for path in shard_paths:
                with h5py.File(path, 'r') as f:
                    #y_shard = np.concatenate([f[v][:] for v in variables], axis=1)
                    y_shard = np.stack([f[v][:] if f[v][:].ndim > 0 else f[v][:][None] for v in variables], axis=1)
                    ys.append(y_shard)
            all_y = np.concatenate(ys, axis=0)
            self.mu = np.mean(all_y, axis=0)
            self.stds = np.std(all_y, axis=0)
        else:
            self.mu = 0
            self.stds = 1

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        shard_id, local_idx = self.index_map[idx]
        shard_path = self.shard_paths[shard_id]

        with h5py.File(shard_path, 'r') as f:
            arr = {}
            for i in self.inputs + self.variables + self.observables:
                x = f[i][local_idx]
                if x.ndim == 0:
                    x = x[None]
                arr[i] = x[:, None]

        #X_clean = np.concatenate([arr[i] for i in self.inputs], axis=1)
        #X_clean = np.swapaxes(X_clean, 0, 1)[:self.cutoff, :]
        X_clean = np.stack([arr[i].squeeze(-1) for i in self.inputs], axis=1)
        X_clean = X_clean[:self.cutoff, :]
        #y = np.concatenate([arr[v] for v in self.variables], axis=1)
        #obs = np.concatenate([arr[o] for o in self.observables], axis=1)
        y = np.stack([arr[v].squeeze(-1) for v in self.variables], axis=1)
        obs = np.stack([arr[o].squeeze(-1) for o in self.observables], axis=1)
        
        X_ts = X_clean.copy()

        for j in range(X_ts.shape[1]):
            noise_arr = noise_model(self.cutoff, self.noise_const)
            X_noise = X_clean[:, j] + noise_arr

            if self.apply_filter:
                X_noise = bandpass_filter(X_noise)

            std_X = np.std(X_noise) if self.norm else 1
            X_ts[:, j] = X_noise / std_X

        index_ts_I = self.inputs.index('output_ts_I')
        index_ts_Q = self.inputs.index('output_ts_Q')
        X_fft = np.zeros_like(X_ts)
       
        real_part, imag_part = fft_func_IQ_complex_channels(
                    X_ts[:, index_ts_I],
                    X_ts[:, index_ts_Q]

            )

        fft_case_data = np.stack([real_part, imag_part], axis=1)
        mu_fft_case = np.mean(fft_case_data, axis=0)
        stds_fft_case = np.std(fft_case_data, axis=0)
        real_part_norm = (real_part - mu_fft_case[0]) / stds_fft_case[0]
        imag_part_norm = (imag_part - mu_fft_case[1]) / stds_fft_case[1]
        X_fft[:, index_ts_I] = real_part_norm
        X_fft[:, index_ts_Q] = imag_part_norm

        ts = torch.tensor(X_ts, dtype=torch.float32)
        fft = torch.tensor(X_fft, dtype=torch.float32)
        var = torch.tensor((y - self.mu) / self.stds, dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32)
        return ts, fft, var, obs

    def outdim(self):
        return len(self.variables)

    def __indim_ts__(self):
        return len(self.inputs)

    def __indim_fft__(self):
        return len(self.inputs)

class GenericDataModule(L.LightningDataModule):
    def __init__(self, batch_size=512, num_workers=4, pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory
        }


class LitDataModule(GenericDataModule):
    def __init__(self, shard_dir, inputs, variables, observables,
                 cutoff=4000, norm=True, noise_const=1,
                 apply_filter=False, use_curriculum_learning=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.hdf5")))
        self.initial_noise_const = noise_const
        self.use_curriculum_learning = use_curriculum_learning
        
        self.dataset = Project8Sim(
            self.shard_paths, inputs, variables, observables,
            cutoff=cutoff, norm=norm,
            noise_const=noise_const,
            apply_filter=apply_filter
        )

        self.mu = self.dataset.mu
        self.stds = self.dataset.stds
        generator = torch.Generator().manual_seed(42)
        train_len = int(0.8 * len(self.dataset))
        val_len = int(0.1 * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_len, val_len, test_len], generator=generator
        )

        self.inputs = inputs
        self.variables = variables
        self.observables = observables
        self.input_channels_ts = self.dataset.__indim_ts__()
        self.input_channels_fft = self.dataset.__indim_fft__()
        self.save_hyperparameters(ignore=['use_curriculum_learning'])

    def set_noise_const(self, new_const):
        if self.use_curriculum_learning:
            self.dataset.noise_const = new_const

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
