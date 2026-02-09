import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
import numpy as np
import torch
from src.utils.noise import *
from src.utils.transforms import fft_func_IQ_complex_channels, qtransform_func_IQ_complex_channels 

class Project8Sim(Dataset):
    def __init__(self, 
            inputs, variables, observables, 
            path='/gpfs/gibbs/pi/heeger/hb637/ssm_files_pi_heeger/combined_data_fullsim.hdf5', 
            cutoff=4000, 
            norm=True, 
            noise_const=1,
            apply_filter = False,
            freq_transform='fft',  # 'fft', 'qtransform', 'both', or None
            q_params=None):  # Dict with Q-transform parameters):

        arr = {}
        with h5py.File(path, 'r') as f:
            for i in inputs+variables+observables:
                arr[i] = f[i][:]
                arr[i] = arr[i][:, np.newaxis]

        X_original = np.concatenate([arr[i] for i in inputs], axis = 1)
        self.X_original = np.swapaxes(X_original,1,2)[:,:cutoff, :]
        self.y_original = np.concatenate([arr[v] for v in variables], axis = 1)
        self.obs = np.concatenate([arr[o] for o in observables], axis = 1)

        self.inputs = inputs
        self.cutoff = cutoff
        self.norm = norm
        self.apply_filter = apply_filter
        self.noise_const = noise_const

        self.freq_transform = freq_transform
        # Q-transform parameters with defaults
        if q_params is None:
            self.q_params = {
                'fs': 200e6,
                'fmin': 1e6,
                'fmax': 100e6,
                'q_value': 5.0,
                'num_freqs': 100
            }
        else:
            self.q_params = q_params

        if norm:
            self.mu = np.mean(self.y_original, axis=0)
            self.stds = np.std(self.y_original, axis=0)
            self.vars = (self.y_original - self.mu) / self.stds
        else:
            # TODO: see what changed here!
            self.mu = np.zeros_like(self.y_original[0])
            self.stds = np.ones_like(self.y_original[0])
            self.vars = self.y_original

    def _compute_fft(self, X_ts, index_ts_I, index_ts_Q):
        # function to call FFT transform
        X_fft = np.zeros_like(X_ts)
        real_part, imag_part = fft_func_IQ_complex_channels(
            X_ts[:, index_ts_I],
            X_ts[:, index_ts_Q]
        )
        fft_case_data = np.stack([real_part, imag_part], axis=1)
        mu_fft_case = np.mean(fft_case_data, axis=0)
        stds_fft_case = np.std(fft_case_data, axis=0)

        real_part_norm = (real_part - mu_fft_case[0]) / (stds_fft_case[0] + 1e-8)
        imag_part_norm = (imag_part - mu_fft_case[1]) / (stds_fft_case[1] + 1e-8)

        X_fft[:, index_ts_I] = real_part_norm
        X_fft[:, index_ts_Q] = imag_part_norm

        return X_fft

    def _compute_qtransform(self, X_ts, index_ts_I, index_ts_Q):
        # function to call Q tranform
        real_part, imag_part = qtransform_func_IQ_complex_channels(
            X_ts[:, index_ts_I],
            X_ts[:, index_ts_Q],
            **self.q_params
        )

        num_freqs = self.q_params['num_freqs']
        real_part = real_part.reshape(self.cutoff, num_freqs)
        imag_part = imag_part.reshape(self.cutoff, num_freqs)

        # Normalize
        q_case_data = np.stack([real_part, imag_part], axis=0)
        mu_q_case = np.mean(q_case_data, axis=(1, 2), keepdims=True)
        stds_q_case = np.std(q_case_data, axis=(1, 2), keepdims=True)

        real_part_norm = (real_part - mu_q_case[0]) / (stds_q_case[0] + 1e-8)
        imag_part_norm = (imag_part - mu_q_case[1]) / (stds_q_case[1] + 1e-8)

        # Stack: [cutoff, num_freqs, 2]
        X_q = np.stack([real_part_norm, imag_part_norm], axis=-1)

        return X_q

    def __getitem__(self, idx):
        X_clean = self.X_original[idx].copy()
        X_ts = X_clean.copy()
        
        # Apply noise to time series
        for j in range(X_ts.shape[1]):
            noise_arr = noise_model(self.cutoff, self.noise_const)
            X_noise = X_clean[:, j] + noise_arr
            if self.apply_filter:
                X_noise = bandpass_filter(X_noise)
            std_X = np.std(X_noise) if self.norm else 1
            X_ts[:, j] = X_noise / std_X
        
        ts = torch.tensor(X_ts, dtype=torch.float32)
        var = torch.tensor(self.vars[idx], dtype=torch.float32)
        obs = torch.tensor(self.obs[idx, :], dtype=torch.float32)
        
        # Compute frequency transforms based on configuration
        if self.freq_transform is None:
            return ts, var, obs
        
        index_ts_I = self.inputs.index('output_ts_I')
        index_ts_Q = self.inputs.index('output_ts_Q')
        
        if self.freq_transform == 'fft':
            X_fft = self._compute_fft(X_ts, index_ts_I, index_ts_Q)
            fft = torch.tensor(X_fft, dtype=torch.float32)
            return ts, fft, var, obs
        
        elif self.freq_transform == 'qtransform':
            X_q = self._compute_qtransform(X_ts, index_ts_I, index_ts_Q)
            qtransform = torch.tensor(X_q, dtype=torch.float32)
            return qtransform, var, obs
        
        elif self.freq_transform == 'both':
            X_fft = self._compute_fft(X_ts, index_ts_I, index_ts_Q)
            fft = torch.tensor(X_fft, dtype=torch.float32)
            return ts, fft, var, obs
        
        else:
            raise ValueError(f"Invalid freq_transform: {self.freq_transform}. "
                           f"Must be 'fft', 'qtransform', 'both', or None")

    def outdim(self):
        return self.vars.shape[1]

    def __len__(self):
        return self.vars.shape[0]

    def __indim_ts__(self):
        return self.X_original.shape[2]

    def __indim_fft__(self):
        if self.freq_transform in ['fft', 'both']:
            return self.X_original.shape[2]
        elif self.freq_transform == 'qtransform':
            return self.q_params['num_freqs'] * 2  # real + imag
        return None

    def __indim_qtransform__(self):
        if self.freq_transform in ['qtransform', 'both']:
            return self.q_params['num_freqs'] * 2  # real + imag
        return None
