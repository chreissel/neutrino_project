import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
import numpy as np
import torch
from noise import *
from utils import fft_func_IQ_complex_channels

class Project8Sim(Dataset):
    def __init__(self, inputs, variables, observables, path='/gpfs/gibbs/pi/heeger/hb637/ssm_files_pi_heeger/combined_data_fullsim.hdf5', cutoff=4000, norm=True, noise_const=1,
            apply_filter = False):

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

        if norm:
            self.mu = np.mean(self.y_original, axis=0)
            self.stds = np.std(self.y_original, axis=0)
            self.vars = (self.y_original - self.mu) / self.stds
        else:
            self.mu = np.zeros_like(self.y_original[0])
            self.stds = np.ones_like(self.y_original[0])
            self.vars = self.y_original

    def __getitem__(self,idx):
        X_clean = self.X_original[idx].copy()
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
        var = torch.tensor(self.vars[idx], dtype=torch.float32)
        obs = torch.tensor(self.obs[idx,:], dtype=torch.float32)
        return ts, fft, var, obs

    def outdim(self):
        return self.vars.shape[1]

    def __len__(self):
        return self.vars.shape[0]

    def __indim_ts__(self):
        return self.X_original.shape[2]

    def __indim_fft__(self):
        return self.X_original.shape[2]

class GenericDataModule(L.LightningDataModule):
    def __init__(self,batch_size=512,num_workers=4,pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {"batch_size":self.batch_size,
                              "num_workers":self.num_workers,
                              "pin_memory":self.pin_memory}

class LitDataModule(GenericDataModule):
    def __init__(self, inputs, variables, observables, cutoff=4000, path='/gpfs/gibbs/pi/heeger/hb637/ssm_files_pi_heeger/combined_data_fullsim.hdf5', norm=True, noise_const=1, apply_filter=False, 
            use_curriculum_learning=False, max_noise_const=1.0, **kwargs):
        super().__init__(**kwargs)
        
        initial_noise_const = noise_const
        self.use_curriculum_learning = use_curriculum_learning

        dataset = Project8Sim(inputs, variables, observables, path, cutoff, norm, initial_noise_const, apply_filter)
        self.dataset = dataset
        self.mu = self.dataset.mu
        self.stds = self.dataset.stds
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [0.8,0.1,0.1], generator=generator)
        self.observables = observables
        self.variables = variables
        self.input_channels_ts = self.dataset.__indim_ts__()
        self.input_channels_fft = self.dataset.__indim_fft__()
        self.save_hyperparameters(ignore=['use_curriculum_learning'])

    def set_noise_const(self, new_const):
        if self.use_curriculum_learning:
            self.dataset.noise_const = new_const

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True, **self.loader_kwargs)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
