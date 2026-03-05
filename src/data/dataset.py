import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
import numpy as np
import torch
from pathlib import Path
from src.utils.noise import *
from src.utils.transforms import fft_func_IQ_complex_channels, qtransform_func_IQ_complex_channels
import os

class Project8Sim(Dataset):
    def __init__(self,
            inputs, variables, observables,
            data_dir,           
            cutoff=4000,
            norm=True,
            noise_const=1,
            apply_filter=False,
            freq_transform='fft', # 'fft', 'qtransform', 'both', or None
            q_params=None):

        data_dir = str(data_dir)
        hdf5_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.hdf5') or f.endswith('.h5')
        ])
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in directory: {data_dir}")

        self.paths = hdf5_files

        self.inputs       = inputs
        self.variables    = variables
        self.observables  = observables
        self.cutoff       = cutoff
        self.norm         = norm
        self.noise_const  = noise_const
        self.apply_filter = apply_filter
        self.freq_transform = freq_transform

        self.q_params = q_params or {
            'fs': 200e6, 'fmin': 1e6, 'fmax': 100e6,
            'q_value': 5.0, 'num_freqs': 100
        }

        self._index = []
        self._file_lengths = {}
        for path in self.paths:
            with h5py.File(path, 'r') as f:
                n = f[(variables + observables + inputs)[0]].shape[0]
            self._file_lengths[path] = n
            self._index.extend((path, i) for i in range(n))

        if norm:
            self.mu, self.stds = self._compute_norm_stats()
        else:
            self.mu = self.stds = None

        self._file_handles: dict = {}

    def _compute_norm_stats(self):
        n_vars = len(self.variables)
        count  = 0
        mean   = np.zeros(n_vars, dtype=np.float64)
        M2     = np.zeros(n_vars, dtype=np.float64)

        for path in self.paths:
            with h5py.File(path, 'r') as f:
                # read only the variable columns (NOT inputs – much smaller)
                data = np.stack([f[v][:] for v in self.variables], axis=1)  # (N, n_vars)
                data = data.reshape(len(data), n_vars)
            for row in data:          # Welford online update
                count += 1
                delta  = row - mean
                mean  += delta / count
                M2    += delta * (row - mean)

        return mean.astype(np.float32), np.sqrt(M2 / count).astype(np.float32)

    def _get_handle(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r')
        return self._file_handles[path]

    def _read_row(self, path, local_idx, keys):
        f = self._get_handle(path)
        return {k: f[k][local_idx] for k in keys}

    def _compute_fft(self, X_ts, index_I, index_Q):
        real, imag = fft_func_IQ_complex_channels(X_ts[:, index_I], X_ts[:, index_Q])
        stack = np.stack([real, imag], axis=1)
        mu, std = np.mean(stack, axis=0), np.std(stack, axis=0)
        X_fft = np.zeros_like(X_ts)
        X_fft[:, index_I] = (real - mu[0]) / (std[0] + 1e-8)
        X_fft[:, index_Q] = (imag - mu[1]) / (std[1] + 1e-8)
        return X_fft

    def _compute_qtransform(self, X_ts, index_I, index_Q):
        real, imag = qtransform_func_IQ_complex_channels(
            X_ts[:, index_I], X_ts[:, index_Q], **self.q_params)
        nf = self.q_params['num_freqs']
        real = real.reshape(self.cutoff, nf)
        imag = imag.reshape(self.cutoff, nf)
        stack = np.stack([real, imag], axis=0)
        mu    = np.mean(stack, axis=(1, 2), keepdims=True)
        std   = np.std( stack, axis=(1, 2), keepdims=True)
        return np.stack([(real - mu[0]) / (std[0] + 1e-8),
                         (imag - mu[1]) / (std[1] + 1e-8)], axis=-1)  # (cutoff, nf, 2)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        path, local_idx = self._index[idx]
        row = self._read_row(path, local_idx, self.inputs + self.variables + self.observables)

        # inputs → (cutoff, n_inputs)
        X_clean = np.stack([row[k] for k in self.inputs], axis=-1)[:self.cutoff]
        X_ts    = X_clean.copy()

        for j in range(X_ts.shape[1]):
            noise  = noise_model(self.cutoff, self.noise_const)
            Xn     = X_clean[:, j] + noise
            if self.apply_filter:
                Xn = bandpass_filter(Xn)
            X_ts[:, j] = Xn / (np.std(Xn) + 1e-8) if self.norm else Xn

        y_raw    = np.array([row[v] for v in self.variables],   dtype=np.float32)
        var_norm = (y_raw - self.mu) / (self.stds + 1e-8) if self.norm else y_raw
        obs      = np.array([row[o] for o in self.observables], dtype=np.float32)

        ts  = torch.tensor(X_ts,     dtype=torch.float32)
        var = torch.tensor(var_norm, dtype=torch.float32)
        obs = torch.tensor(obs,      dtype=torch.float32)

        if self.freq_transform is None:
            return ts, var, obs

        iI = self.inputs.index('output_ts_I')
        iQ = self.inputs.index('output_ts_Q')

        if self.freq_transform == 'fft':
            return ts, torch.tensor(self._compute_fft(X_ts, iI, iQ), dtype=torch.float32), var, obs

        if self.freq_transform == 'qtransform':
            return torch.tensor(self._compute_qtransform(X_ts, iI, iQ), dtype=torch.float32), var, obs

        if self.freq_transform == 'both':
            fft = torch.tensor(self._compute_fft(X_ts, iI, iQ),        dtype=torch.float32)
            qt  = torch.tensor(self._compute_qtransform(X_ts, iI, iQ), dtype=torch.float32)
            return ts, fft, qt, var, obs

        raise ValueError(f"Invalid freq_transform '{self.freq_transform}'. "
                         "Choose 'fft', 'qtransform', 'both', or None.")

    def outdim(self):          return len(self.variables)
    def __indim_ts__(self):    return len(self.inputs)
    def __indim_fft__(self):
        if self.freq_transform in ('fft', 'both'):     return len(self.inputs)
        if self.freq_transform == 'qtransform':        return self.q_params['num_freqs'] * 2
    def __indim_qtransform__(self):
        if self.freq_transform in ('qtransform', 'both'): return self.q_params['num_freqs'] * 2

    def close(self):
        for fh in self._file_handles.values(): fh.close()
        self._file_handles.clear()
    def __del__(self): self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Required when using DataLoader with num_workers > 0
# HDF5 is not fork-safe; each worker must open its own handles.
#
#   loader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init_fn)
# ─────────────────────────────────────────────────────────────────────────────
def worker_init_fn(worker_id):
    import torch.utils.data
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._file_handles = {}


class Project8SimDenoising(Dataset):
    """Dataset for the S4D denoising task.

    Returns ``(X_noisy, X_clean)`` pairs where:
    - ``X_noisy`` – simulated I/Q signal with Gaussian noise added, optionally
      normalised per channel by ``std(X_noisy[:, j])``.
    - ``X_clean`` – the raw (noiseless) simulation signal, normalised by the
      *same* per-channel ``std(X_noisy[:, j])`` so that both tensors live in
      the same coordinate space (only when ``norm=True``).

    Both tensors have shape ``(cutoff, len(inputs))``.
    """

    def __init__(self,
                 inputs,
                 data_dir,
                 cutoff=4000,
                 noise_const=1,
                 apply_filter=False,
                 norm=True):

        data_dir = str(data_dir)
        hdf5_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.hdf5') or f.endswith('.h5')
        ])
        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in directory: {data_dir}")

        self.paths        = hdf5_files
        self.inputs       = inputs
        self.cutoff       = cutoff
        self.noise_const  = noise_const
        self.apply_filter = apply_filter
        self.norm         = norm

        self._index = []
        for path in self.paths:
            with h5py.File(path, 'r') as f:
                n = f[inputs[0]].shape[0]
            self._index.extend((path, i) for i in range(n))

        self._file_handles: dict = {}

    def _get_handle(self, path):
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r')
        return self._file_handles[path]

    def _read_row(self, path, local_idx, keys):
        f = self._get_handle(path)
        return {k: f[k][local_idx] for k in keys}

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        path, local_idx = self._index[idx]
        row = self._read_row(path, local_idx, self.inputs)

        X_clean = np.stack([row[k] for k in self.inputs], axis=-1)[:self.cutoff]
        X_noisy_norm = np.zeros_like(X_clean)
        X_clean_norm = np.zeros_like(X_clean)

        for j in range(X_clean.shape[1]):
            noise = noise_model(self.cutoff, self.noise_const)
            Xn = X_clean[:, j] + noise
            if self.apply_filter:
                Xn = bandpass_filter(Xn)
            if self.norm:
                s = np.std(Xn) + 1e-8
                X_noisy_norm[:, j] = Xn / s
                X_clean_norm[:, j] = X_clean[:, j] / s
            else:
                X_noisy_norm[:, j] = Xn
                X_clean_norm[:, j] = X_clean[:, j]

        return (
            torch.tensor(X_noisy_norm, dtype=torch.float32),
            torch.tensor(X_clean_norm, dtype=torch.float32),
        )

    def __indim__(self):
        return len(self.inputs)

    def close(self):
        for fh in self._file_handles.values():
            fh.close()
        self._file_handles.clear()

    def __del__(self):
        self.close()
