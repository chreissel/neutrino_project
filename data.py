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
                #arr[i] = f[i][:100]
                arr[i] = f[i][:]
                arr[i] = arr[i][:, np.newaxis]
        X = np.concatenate([arr[i] for i in inputs], axis = 1)
        X = np.swapaxes(X,1,2)[:,:cutoff, :]
        y = np.concatenate([arr[v] for v in variables], axis = 1)
        obs = np.concatenate([arr[o] for o in observables], axis = 1)
        # add noise and normalize by std

        X_ts = np.copy(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[2]):
                noise_arr = noise_model(cutoff, noise_const)
                X_noise = X[i, :, j] + noise_arr
                if apply_filter:
                    X_noise = bandpass_filter(X_noise)
                if norm:
                    std_X = np.std(X_noise)
                else:
                    std_X = 1
                X_ts[i, :, j] = X_noise / std_X

        X_fft = None
        if ('output_ts_I' in inputs) and ('output_ts_Q' in inputs):
            index_ts_I = inputs.index('output_ts_I')
            index_ts_Q = inputs.index('output_ts_Q')
            X_fft = np.zeros_like(X_ts)
            for i in range(X.shape[0]):
                real_part, imag_part = fft_func_IQ_complex_channels(
                            X_ts[i, :, index_ts_I],
                            X_ts[i, :, index_ts_Q]
                    )
                X_fft[i, :, index_ts_I] = real_part
                X_fft[i, :, index_ts_Q] = imag_part

        if norm:
            mu_y = np.mean(y, axis=0)
            stds_y = np.std(y, axis=0)
            y = (y - mu_y) / stds_y

        self.mu = mu_y
        self.stds = stds_y
        self.timeseries = np.float32(X_ts)
        self.fft_data = np.float32(X_fft)
        self.vars = np.float32(y)
        self.obs = np.float32(obs)

    def __len__(self):
        return self.vars.shape[0]

    def __getitem__(self, idx):
        times = self.timeseries[idx, :, :]
        fft = self.fft_data[idx,:,:]
        var = self.vars[idx]
        obs = self.obs[idx, :]
        return times, fft, var, obs

    def __outdim__(self):
        return self.vars.shape[1]

    def __indim_ts__(self):
        return self.timeseries.shape[2]

    def __indim_fft__(self):
        return self.fft_data.shape[2]

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
    def __init__(self, inputs, variables, observables, cutoff=4000, path='/gpfs/gibbs/pi/heeger/hb637/ssm_files_pi_heeger/combined_data_fullsim.hdf5', norm=True, noise_const=1, apply_filter=False, **kwargs):
        super().__init__(**kwargs)
       
        dataset = Project8Sim(inputs, variables, observables, path, cutoff, norm, noise_const, apply_filter)
        self.mu = dataset.mu
        self.stds = dataset.stds
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [0.8,0.1,0.1], generator=generator)
        self.observables = observables
        self.variables = variables
        self.input_channels_ts = dataset.__indim_ts__()
        self.input_channels_fft = dataset.__indim_fft__()
        self.save_hyperparameters()

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=True, **self.loader_kwargs)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
