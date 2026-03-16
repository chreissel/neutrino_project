import lightning as L
from torch.utils.data import random_split, DataLoader
import h5py
import numpy as np
import torch
from src.data.dataset import Project8Sim, Project8SimDenoising, Project8SimCombined

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
    def __init__(self, 
            inputs, variables, observables,
            data_dir,
            cutoff=4000, 
            norm=True, 
            noise_const=1, 
            apply_filter=False, 
            use_curriculum_learning=False, 
            max_noise_const=1.0,
            freq_transform = 'fft',
            q_params = None,
            **kwargs):
        super().__init__(**kwargs)
        
        initial_noise_const = noise_const
        self.use_curriculum_learning = use_curriculum_learning

        self.dataset = Project8Sim(
            inputs, variables, observables, data_dir, cutoff, norm,
            initial_noise_const, apply_filter, freq_transform, q_params
        )
        self.mu = self.dataset.mu
        self.stds = self.dataset.stds
        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [0.8,0.1,0.1], generator=generator)
        self.observables = observables
        self.variables = variables
        self.input_channels_ts = self.dataset.__indim_ts__()
        self.input_channels_fft = self.dataset.__indim_fft__()
        self.input_channels_qtransform = self.dataset.__indim_qtransform__()
        
        self.save_hyperparameters()

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


class LitDenoisingDataModule(GenericDataModule):
    """DataModule for the S4D denoising task.

    Wraps :class:`~src.data.dataset.Project8SimDenoising` and exposes the
    same curriculum-learning interface as :class:`LitDataModule` so that
    :class:`~src.models.model.LitS4DenoisingModel` can optionally ramp up the
    noise level over training epochs.

    Batches are ``(X_noisy, X_clean)`` tensors of shape ``(B, cutoff, C)``
    where ``C = len(inputs)``.
    """

    def __init__(self,
                 inputs,
                 data_dir,
                 cutoff=4000,
                 noise_const=1.0,
                 apply_filter=False,
                 use_curriculum_learning=False,
                 norm=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_curriculum_learning = use_curriculum_learning

        self.dataset = Project8SimDenoising(
            inputs=inputs,
            data_dir=data_dir,
            cutoff=cutoff,
            noise_const=noise_const,
            apply_filter=apply_filter,
            norm=norm,
        )

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.8, 0.1, 0.1], generator=generator
        )

        self.input_channels = self.dataset.__indim__()
        self.save_hyperparameters()

    def set_noise_const(self, new_const):
        if self.use_curriculum_learning:
            self.dataset.noise_const = new_const

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)


class LitCombinedDataModule(GenericDataModule):
    """DataModule for the combined denoising + regression task.

    Wraps :class:`~src.data.dataset.Project8SimCombined` and produces batches
    of ``(X_noisy, X_clean, y, obs)`` for :class:`~src.models.model.LitS4CombinedModel`.
    """

    def __init__(self,
                 inputs,
                 variables,
                 observables,
                 data_dir,
                 cutoff=4000,
                 noise_const=1.0,
                 apply_filter=False,
                 use_curriculum_learning=False,
                 norm=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_curriculum_learning = use_curriculum_learning

        self.dataset = Project8SimCombined(
            inputs=inputs,
            variables=variables,
            observables=observables,
            data_dir=data_dir,
            cutoff=cutoff,
            noise_const=noise_const,
            apply_filter=apply_filter,
            norm=norm,
        )

        self.mu   = self.dataset.mu
        self.stds = self.dataset.stds

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [0.8, 0.1, 0.1], generator=generator
        )

        self.input_channels = self.dataset.__indim__()
        self.variables  = variables
        self.observables = observables
        self.save_hyperparameters()

    def set_noise_const(self, new_const):
        if self.use_curriculum_learning:
            self.dataset.noise_const = new_const

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
