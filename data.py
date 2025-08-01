import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
import numpy as np
import torch

class Project8Sim(Dataset):
    def __init__(self, inputs, variables, observables, path='/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/combined_data_v2.hdf5', cutoff=4000, norm=True):

        arr = {}
        with h5py.File(path, 'r') as f:
            for i in inputs+variables+observables:
                #arr[i] = f[i][:100]
                arr[i] = f[i][:]
                arr[i] = arr[i][:, np.newaxis]
        X = np.concatenate([arr[i] for i in inputs], axis = 1)
        X = np.swapaxes(X,1,2)[:,:cutoff, :]
        y = np.concatenate([arr[v] for v in variables], axis = 1)
        obs = np.concatenate([arr[i] for o in observables], axis = 1)
        
        if norm:
            mu_y = np.mean(y, axis=0)
            stds_y = np.std(y, axis=0)
            y = (y-mu_y)/stds_y

            stds_X = np.std(X, axis=1)
            for i in range(len(stds_X)):
                X[i, :, :] = X[i, :, :]/stds_X[i]
            
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
    def __init__(self, inputs, variables, observables, cutoff=4000, path='/n/holystore01/LABS/iaifi_lab/Lab/creissel/neutrino_mass/combined_data_v2.hdf5', norm=True, **kwargs):
        super().__init__(**kwargs)
       
        dataset = Project8Sim(inputs, variables, observables, path, cutoff, norm)
        self.mu = dataset.mu
        self.stds = dataset.stds
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [0.8,0.1,0.1], generator=generator)
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
