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
            cutoff=4000, norm=True, noise_const=1, apply_filter=False, apply_fft=False):

        self.shard_paths = shard_paths
        self.inputs = inputs
        self.variables = variables
        self.observables = observables
        self.cutoff = cutoff
        self.norm = norm
        self.apply_filter = apply_filter
        self.noise_const = noise_const
        self.apply_fft = apply_fft

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
            data_channels = []
            for i in self.inputs:
                ch_data = f[i][local_idx][:self.cutoff].astype(np.float32)
                data_channels.append(ch_data)
            
            y_raw = []
            for v in self.variables:
                val = f[v][local_idx]
                y_raw.append(val.item() if np.ndim(val) >= 0 else val)
            y = np.array(y_raw, dtype=np.float32)

            obs_raw = []
            for o in self.observables:
                val = f[o][local_idx]
                obs_raw.append(val.item() if np.ndim(val) >= 0 else val)
                obs = np.array(obs_raw, dtype=np.float32)

        X_ts = np.stack(data_channels, axis=1)
        
        for j in range(X_ts.shape[1]):
            noise_arr = noise_model(self.cutoff, self.noise_const)
            X_noise = X_ts[:, j] + noise_arr
        
            if self.apply_filter:
                X_noise = bandpass_filter(X_noise)

            if self.norm:
                std_X = np.std(X_noise)
                X_ts[:, j] = X_noise / (std_X)
            else:
                X_ts[:, j] = X_noise
        
        if self.apply_fft:
            index_ts_I = self.inputs.index('output_ts_I')
            index_ts_Q = self.inputs.index('output_ts_Q')
            real_part, imag_part = fft_func_IQ_complex_channels(
                    X_ts[:, index_ts_I],
                    X_ts[:, index_ts_Q]
                )
        fft_case_data = np.stack([real_part, imag_part], axis=1) # (Length, 2)
        mu_fft = np.mean(fft_case_data, axis=0)
        std_fft = np.std(fft_case_data, axis=0)
        X_ts[:, index_ts_I] = (real_part - mu_fft[0]) / (std_fft[0])
        X_ts[:, index_ts_Q] = (imag_part - mu_fft[1]) / (std_fft[1])
        
        ts = torch.tensor(X_ts, dtype=torch.float32)
        var = torch.tensor((y - self.mu) / self.stds, dtype=torch.float32)
        obs = torch.tensor(obs, dtype=torch.float32)
        return ts, var, obs


    def outdim(self):
        return len(self.variables)
    
    def __indim__(self):
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
    def __init__(self, shard_dir, inputs, variables, observables, cutoff=4000, norm=True, noise_const=1, apply_filter=False, apply_fft=False, use_curriculum_learning=False, **kwargs):
        super().__init__(**kwargs)
        
        self.shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.hdf5")))
        self.initial_noise_const = noise_const
        self.use_curriculum_learning = use_curriculum_learning
        
        self.dataset = Project8Sim(self.shard_paths, inputs, variables, observables, cutoff=cutoff, norm=norm, noise_const=noise_const, apply_filter=apply_filter, apply_fft=apply_fft)
        
        self.mu = self.dataset.mu
        self.stds = self.dataset.stds
        generator = torch.Generator().manual_seed(42)
        train_len = int(0.8 * len(self.dataset))
        val_len = int(0.1 * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_len, val_len, test_len], generator=generator)
        
        self.inputs = inputs
        self.variables = variables
        self.observables = observables
        self.input_channels_ts = self.dataset.__indim__()
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
