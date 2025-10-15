import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
import numpy as np
import torch
from noise import *
from utils import fft_func_IQ_abs, fft_func_IQ_complex_channels

class Project8Sim(Dataset):
    def __init__(self, inputs, variables, observables, path='/gpfs/gibbs/pi/heeger/hb637/ssm_files_pi_heeger/combined_data_fullsim.hdf5', cutoff=4000, norm=True, noise_const=1,
            apply_filter = False, apply_fft=False, complex_channels=False):
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

        if ('output_ts_I' in inputs) and ('output_ts_Q' in inputs) and apply_fft:
            index_ts_I = inputs.index('output_ts_I')
            index_ts_Q = inputs.index('output_ts_Q')
            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    noise_arr = noise_model(cutoff, noise_const)
                    X_noise = X[i, :, j] + noise_arr
                    if(apply_filter): # whether to apply the band pass filter
                        X_noise = bandpass_filter(X_noise)
                    if(norm): # whether to normalize by std
                        std_X = np.std(X_noise)
                    else:
                        std_X = 1
                    X[i, :, j] = X_noise/std_X

                if(complex_channels):
                    X[i,:,index_ts_I], X[i,:,index_ts_Q] = fft_func_IQ_complex_channels(X[i, :, index_ts_I], X[i, :, index_ts_Q])
                else:
                    X[i,:,0] = fft_func_IQ_abs(X[i, :, index_ts_I], X[i, :, index_ts_Q])


            if apply_fft and not(complex_channels):
                X = X[:,:,0]
                X = X[:, :, np.newaxis]

        else:
            for i in range(X.shape[0]):
                for j in range(X.shape[2]):
                    noise_arr = noise_model(cutoff, noise_const)
                    X_noise = X[i, :, j] + noise_arr
                    if(apply_filter): # whether to apply the band pass filter
                        X_noise = bandpass_filter(X_noise)
                    if(apply_fft):
                        X_noise = fft_func(X_noise)
                        if len(X_noise) < cutoff:
                            X_noise = np.pad(X_noise, (0, cutoff - len(X_noise)), mode='constant')
                    if(norm and not(apply_fft)): # whether to normalize by std
                        std_X = np.std(X_noise)
                    else:
                        std_X = 1
                    X[i, :, j] = X_noise/std_X

        if norm:
            mu_y = np.mean(y, axis=0)
            stds_y = np.std(y, axis=0)
            y = (y-mu_y)/stds_y

        self.mu = mu_y
        self.stds = stds_y
        self.timeseries = np.float32(X)
        self.vars = np.float32(y)
        self.obs = np.float32(obs)

    def __len__(self):
        return self.vars.shape[0]

    def __getitem__(self, idx):
        times = self.timeseries[idx, :, :]
        var = self.vars[idx]
        obs = self.obs[idx, :]
        return times, var, obs

    def __outdim__(self):
        return self.vars.shape[1]

    def __indim__(self):
        return self.timeseries.shape[1]

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
    def __init__(self, inputs, variables, observables, cutoff=4000, path='/gpfs/gibbs/pi/heeger/hb637/ssm_files_pi_heeger/combined_data_fullsim.hdf5', norm=True, noise_const=1, apply_filter=False, apply_fft=False, complex_channels=False, **kwargs):
        super().__init__(**kwargs)
       
        dataset = Project8Sim(inputs, variables, observables, path, cutoff, norm, noise_const, apply_filter, apply_fft, complex_channels)
        self.mu = dataset.mu
        self.stds = dataset.stds
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [0.8,0.1,0.1], generator=generator)
        self.observables = observables
        self.variables = variables
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
