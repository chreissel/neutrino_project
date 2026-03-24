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
            max_axial_freq=None,
            max_cyc_freq=None,
            multiplier=1,
            **kwargs):
        super().__init__(**kwargs)

        initial_noise_const = noise_const
        self.use_curriculum_learning = use_curriculum_learning

        # Build a dummy dataset to compute norm stats and collect frequency metadata
        dummy = Project8Sim(
            inputs=inputs, variables=variables, observables=observables,
            data_dir=data_dir, cutoff=cutoff, norm=norm,
            noise_const=initial_noise_const, apply_filter=apply_filter,
            freq_transform=freq_transform, multiplier=1
        )
        self.mu   = dummy.mu
        self.stds = dummy.stds

        generator = torch.Generator().manual_seed(42)
        train_split, val_split, test_split = random_split(range(len(dummy)), [0.8, 0.1, 0.1], generator=generator)

        self.train_indices = self._filter(dummy, train_split.indices, max_axial_freq, max_cyc_freq)
        self.val_indices   = self._filter(dummy, val_split.indices,   max_axial_freq, max_cyc_freq)
        self.test_indices  = test_split.indices

        dataset_conf = dict(
            inputs=inputs, variables=variables, observables=observables,
            data_dir=data_dir, cutoff=cutoff, norm=norm,
            noise_const=noise_const, apply_filter=apply_filter,
            freq_transform=freq_transform, q_params=q_params
        )

        self.train_dataset = Project8Sim(**dataset_conf, multiplier=multiplier, is_train=True)
        self.train_dataset.active_indices = np.array(self.train_indices)
        self.train_dataset.mu, self.train_dataset.stds = self.mu, self.stds

        self.val_dataset = Project8Sim(**dataset_conf, multiplier=1, is_train=False)
        self.val_dataset.active_indices = np.array(self.val_indices)
        self.val_dataset.mu, self.val_dataset.stds = self.mu, self.stds

        self.test_dataset = Project8Sim(**dataset_conf, multiplier=1, is_train=False)
        self.test_dataset.active_indices = np.array(self.test_indices)
        self.test_dataset.mu, self.test_dataset.stds = self.mu, self.stds

        self.observables = observables
        self.variables = variables
        self.input_channels_ts = self.train_dataset.__indim_ts__()
        self.input_channels_fft = self.train_dataset.__indim_fft__()
        self.input_channels_qtransform = self.train_dataset.__indim_qtransform__()

        self.save_hyperparameters()

    def _filter(self, dummy, indices, max_ax, max_cyc):
        idx_arr = np.array(indices)
        keep = np.ones(len(idx_arr), dtype=bool)
        if max_ax is not None and dummy.freq_metadata['axial'] is not None:
            keep &= (dummy.freq_metadata['axial'][idx_arr] < max_ax)
        if max_cyc is not None and dummy.freq_metadata['cyc'] is not None:
            keep &= (dummy.freq_metadata['cyc'][idx_arr] < max_cyc)
        return idx_arr[keep].tolist()

    def set_noise_const(self, new_const):
        if self.use_curriculum_learning:
            self.train_dataset.noise_const = new_const

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
                 max_axial_freq=None,
                 max_cyc_freq=None,
                 multiplier=1,
                 **kwargs):
        super().__init__(**kwargs)

        self.use_curriculum_learning = use_curriculum_learning

        # Build a dummy dataset to compute norm stats and collect frequency metadata
        dummy = Project8SimCombined(
            inputs=inputs, variables=variables, observables=observables,
            data_dir=data_dir, cutoff=cutoff, norm=norm,
            noise_const=noise_const, apply_filter=apply_filter, multiplier=1
        )
        self.mu   = dummy.mu
        self.stds = dummy.stds

        generator = torch.Generator().manual_seed(42)
        train_split, val_split, test_split = random_split(range(len(dummy)), [0.8, 0.1, 0.1], generator=generator)

        train_indices = self._filter(dummy, train_split.indices, max_axial_freq, max_cyc_freq)
        val_indices   = self._filter(dummy, val_split.indices,   max_axial_freq, max_cyc_freq)
        test_indices  = test_split.indices

        dataset_conf = dict(
            inputs=inputs, variables=variables, observables=observables,
            data_dir=data_dir, cutoff=cutoff, norm=norm,
            noise_const=noise_const, apply_filter=apply_filter
        )

        self.train_dataset = Project8SimCombined(**dataset_conf, multiplier=multiplier, is_train=True)
        self.train_dataset.active_indices = np.array(train_indices)
        self.train_dataset.mu, self.train_dataset.stds = self.mu, self.stds

        self.val_dataset = Project8SimCombined(**dataset_conf, multiplier=1, is_train=False)
        self.val_dataset.active_indices = np.array(val_indices)
        self.val_dataset.mu, self.val_dataset.stds = self.mu, self.stds

        self.test_dataset = Project8SimCombined(**dataset_conf, multiplier=1, is_train=False)
        self.test_dataset.active_indices = np.array(test_indices)
        self.test_dataset.mu, self.test_dataset.stds = self.mu, self.stds

        self.input_channels = self.train_dataset.__indim__()
        self.variables  = variables
        self.observables = observables
        self.save_hyperparameters()

    def _filter(self, dummy, indices, max_ax, max_cyc):
        idx_arr = np.array(indices)
        keep = np.ones(len(idx_arr), dtype=bool)
        if max_ax is not None and dummy.freq_metadata['axial'] is not None:
            keep &= (dummy.freq_metadata['axial'][idx_arr] < max_ax)
        if max_cyc is not None and dummy.freq_metadata['cyc'] is not None:
            keep &= (dummy.freq_metadata['cyc'][idx_arr] < max_cyc)
        return idx_arr[keep].tolist()

    def set_noise_const(self, new_const):
        if self.use_curriculum_learning:
            self.train_dataset.noise_const = new_const

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
