"""Pure-JAX dataloader for the combined denoising + regression task.

Replaces the PyTorch ``Project8SimCombined`` / ``LitCombinedDataModule`` for
the fully JAX-native training pipeline in ``scripts/train_linoss_jax.py``.

Design
------
* ``CombinedDatasetJAX`` indexes all HDF5 files at construction time (same as
  ``Project8SimCombined``).  Batches are yielded by ``loop`` / ``loop_epoch``
  which group reads by file to minimise open/close overhead.
* Normalisation follows the original exactly:
    - Signals  : per-sample per-channel  ``x / std(x_noisy)``
    - Targets  : Welford online  ``(y - mu) / std``
* Returns JAX arrays so that the training loop never touches PyTorch.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from src.utils.noise import bandpass_filter, noise_model


# ---------------------------------------------------------------------------
# Norm-stats computation (Welford online algorithm — identical to dataset.py)
# ---------------------------------------------------------------------------

def _compute_norm_stats(
    paths: List[str],
    variables: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) for ``variables`` using Welford's algorithm."""
    n_vars = len(variables)
    count  = 0
    mean   = np.zeros(n_vars, dtype=np.float64)
    M2     = np.zeros(n_vars, dtype=np.float64)

    for path in paths:
        with h5py.File(path, 'r') as f:
            data = np.stack([f[v][:] for v in variables], axis=1)
        for row in data:
            count += 1
            delta  = row - mean
            mean  += delta / count
            M2    += delta * (row - mean)

    return mean.astype(np.float32), np.sqrt(M2 / count).astype(np.float32)


# ---------------------------------------------------------------------------
# Core dataset
# ---------------------------------------------------------------------------

class CombinedDatasetJAX:
    """Index wrapper around a collection of HDF5 files.

    Parameters
    ----------
    paths         : list of HDF5 file paths (pre-filtered)
    index         : list of ``(path, local_idx)`` tuples (pre-built)
    active_indices: 1-D int array — subset of ``index`` actually used
    inputs        : signal dataset keys (e.g. ``['output_ts_I', 'output_ts_Q']``)
    variables     : regression target keys
    observables   : auxiliary observable keys (returned but not normalised)
    cutoff        : number of time-steps to read per sample
    noise_const   : multiplicative constant for the noise model
    apply_filter  : whether to apply the Butterworth bandpass filter
    mu, stds      : pre-computed normalisation stats for ``variables``
    """

    def __init__(
        self,
        paths         : List[str],
        index         : List[Tuple[str, int]],
        active_indices: np.ndarray,
        inputs        : List[str],
        variables     : List[str],
        observables   : List[str],
        cutoff        : int,
        noise_const   : float,
        apply_filter  : bool,
        mu            : np.ndarray,
        stds          : np.ndarray,
    ):
        self.paths          = paths
        self.index          = index
        self.active_indices = active_indices
        self.inputs         = inputs
        self.variables      = variables
        self.observables    = observables
        self.cutoff         = cutoff
        self.noise_const    = noise_const
        self.apply_filter   = apply_filter
        self.mu             = mu
        self.stds           = stds

    # ------------------------------------------------------------------
    # Internal: fetch a single sample (numpy, no JAX yet)
    # ------------------------------------------------------------------

    def _process_sample(
        self,
        X_clean_raw : np.ndarray,   # (cutoff, C) float32
        y_raw       : np.ndarray,   # (n_vars,)  float32
        obs_raw     : np.ndarray,   # (n_obs,)   float32
        rng         : Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        C = X_clean_raw.shape[1]
        X_noisy_norm = np.zeros_like(X_clean_raw)
        X_clean_norm = np.zeros_like(X_clean_raw)

        for j in range(C):
            noise = noise_model(self.cutoff, self.noise_const, rng=rng)
            Xn    = X_clean_raw[:, j] + noise
            if self.apply_filter:
                Xn = bandpass_filter(Xn)
            s = float(np.std(Xn)) + 1e-8
            X_noisy_norm[:, j] = Xn / s
            X_clean_norm[:, j] = X_clean_raw[:, j] / s

        y_norm = (y_raw - self.mu) / (self.stds + 1e-8)
        return X_noisy_norm, X_clean_norm, y_norm, obs_raw

    # ------------------------------------------------------------------
    # Batch fetching — reads grouped by file
    # ------------------------------------------------------------------

    def _fetch_batch(
        self,
        indices   : np.ndarray,           # (B,) indices into active_indices
        numpy_rng : np.random.Generator,  # for reproducible noise inside a batch
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_noisy, X_clean, y, obs) as numpy arrays shaped (B, L, C)."""
        B   = len(indices)
        L   = self.cutoff
        C   = len(self.inputs)
        n_v = len(self.variables)
        n_o = len(self.observables)

        X_noisy_batch = np.zeros((B, L, C),  dtype=np.float32)
        X_clean_batch = np.zeros((B, L, C),  dtype=np.float32)
        y_batch       = np.zeros((B, n_v),   dtype=np.float32)
        obs_batch     = np.zeros((B, n_o),   dtype=np.float32)

        # Group by file path to avoid repeated opens
        by_file: dict = defaultdict(list)
        for batch_pos, global_idx in enumerate(indices):
            path, local_idx = self.index[int(self.active_indices[int(global_idx)])]
            by_file[path].append((batch_pos, local_idx))

        all_keys = self.inputs + self.variables + self.observables

        for path, items in by_file.items():
            with h5py.File(path, 'r') as f:
                for batch_pos, local_idx in items:
                    X_c = np.stack(
                        [f[k][local_idx] for k in self.inputs], axis=-1
                    )[:L].astype(np.float32)
                    y_r = np.array(
                        [f[v][local_idx] for v in self.variables], dtype=np.float32
                    )
                    ob  = np.array(
                        [f[o][local_idx] for o in self.observables], dtype=np.float32
                    )
                    xn, xc, yn, obs = self._process_sample(X_c, y_r, ob, rng=numpy_rng)
                    X_noisy_batch[batch_pos] = xn
                    X_clean_batch[batch_pos] = xc
                    y_batch[batch_pos]       = yn
                    obs_batch[batch_pos]     = obs

        return X_noisy_batch, X_clean_batch, y_batch, obs_batch

    # ------------------------------------------------------------------
    # Public iteration interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.active_indices)

    def loop(
        self,
        batch_size : int,
        key        : jax.Array,
        drop_last  : bool = True,
    ):
        """Infinite generator that shuffles each epoch and yields JAX batches.

        Yields
        ------
        (X_noisy, X_clean, y, obs) — JAX float32 arrays.
        """
        n = len(self.active_indices)
        while True:
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, n)
            perm_np = np.array(perm)
            yield from self._epoch_batches(perm_np, batch_size, drop_last, key)

    def loop_epoch(
        self,
        batch_size : int,
        key        : jax.Array,
        drop_last  : bool = False,
    ):
        """Single-epoch generator — shuffles once, yields all batches, stops.

        Yields
        ------
        (X_noisy, X_clean, y, obs) — JAX float32 arrays.
        """
        n = len(self.active_indices)
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, n)
        perm_np = np.array(perm)
        yield from self._epoch_batches(perm_np, batch_size, drop_last, key)

    def _epoch_batches(self, perm_np, batch_size, drop_last, key):
        n = len(perm_np)
        numpy_rng = np.random.default_rng(seed=int(jr.randint(key, (), 0, 2**31 - 1)))
        for start in range(0, n, batch_size):
            batch_idx = perm_np[start : start + batch_size]
            if drop_last and len(batch_idx) < batch_size:
                break
            xn, xc, y, obs = self._fetch_batch(batch_idx, numpy_rng)
            yield (
                jnp.array(xn),
                jnp.array(xc),
                jnp.array(y),
                jnp.array(obs),
            )


# ---------------------------------------------------------------------------
# Data module — holds train / val / test splits
# ---------------------------------------------------------------------------

class CombinedDataModuleJAX:
    """Mirrors ``LitCombinedDataModule`` for the pure-JAX training pipeline.

    Parameters
    ----------
    data_dir       : directory containing ``.hdf5`` / ``.h5`` files
    inputs         : list of signal field names
    variables      : list of regression target field names
    observables    : list of auxiliary observable field names
    cutoff         : time-steps per sample
    noise_const    : noise multiplier
    apply_filter   : apply Butterworth bandpass filter to noisy signal
    max_axial_freq : upper cut on ``avg_axial_frequency_Hz`` (train/val only)
    max_cyc_freq   : upper cut on ``avg_carrier_frequency_Hz`` (train/val only)
    seed           : random seed for the 80/10/10 split (default 42, matches Lightning)
    """

    def __init__(
        self,
        data_dir       : str,
        inputs         : List[str],
        variables      : List[str],
        observables    : List[str],
        cutoff         : int   = 4000,
        noise_const    : float = 1.0,
        apply_filter   : bool  = False,
        max_axial_freq : Optional[float] = None,
        max_cyc_freq   : Optional[float] = None,
        seed           : int   = 42,
    ):
        self.inputs      = inputs
        self.variables   = variables
        self.observables = observables
        self.cutoff      = cutoff
        self.noise_const = noise_const
        self.apply_filter = apply_filter

        # Discover HDF5 files
        paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.hdf5') or f.endswith('.h5')
        ])
        if not paths:
            raise FileNotFoundError(f"No HDF5 files found in: {data_dir}")

        # Build global index (same order as Project8SimCombined)
        index: List[Tuple[str, int]] = []
        axial_parts, cyc_parts = [], []
        for path in paths:
            with h5py.File(path, 'r') as f:
                n = f[inputs[0]].shape[0]
                if 'avg_axial_frequency_Hz' in f:
                    axial_parts.append(f['avg_axial_frequency_Hz'][:])
                if 'avg_carrier_frequency_Hz' in f:
                    cyc_parts.append(f['avg_carrier_frequency_Hz'][:])
            index.extend((path, i) for i in range(n))

        axial_freq = np.concatenate(axial_parts) if axial_parts else None
        cyc_freq   = np.concatenate(cyc_parts)   if cyc_parts   else None

        # Compute normalisation stats (once, over all data)
        mu, stds = _compute_norm_stats(paths, variables)
        self.mu   = mu
        self.stds = stds

        # Reproducible 80/10/10 split (same seed=42 as Lightning pipeline)
        n_total = len(index)
        rng_split = np.random.default_rng(seed)
        perm = rng_split.permutation(n_total)
        n_train = int(0.8 * n_total)
        n_val   = int(0.1 * n_total)
        train_idx = perm[:n_train]
        val_idx   = perm[n_train : n_train + n_val]
        test_idx  = perm[n_train + n_val :]

        # Frequency filtering (train / val only)
        train_idx = self._filter(train_idx, axial_freq, cyc_freq, max_axial_freq, max_cyc_freq)
        val_idx   = self._filter(val_idx,   axial_freq, cyc_freq, max_axial_freq, max_cyc_freq)

        ds_kwargs = dict(
            paths          = paths,
            index          = index,
            inputs         = inputs,
            variables      = variables,
            observables    = observables,
            cutoff         = cutoff,
            noise_const    = noise_const,
            apply_filter   = apply_filter,
            mu             = mu,
            stds           = stds,
        )
        self.train = CombinedDatasetJAX(active_indices=train_idx, **ds_kwargs)
        self.val   = CombinedDatasetJAX(active_indices=val_idx,   **ds_kwargs)
        self.test  = CombinedDatasetJAX(active_indices=test_idx,  **ds_kwargs)

        self.d_input  = len(inputs)
        self.d_output = len(variables)

    @staticmethod
    def _filter(
        indices    : np.ndarray,
        axial_freq : Optional[np.ndarray],
        cyc_freq   : Optional[np.ndarray],
        max_ax     : Optional[float],
        max_cyc    : Optional[float],
    ) -> np.ndarray:
        keep = np.ones(len(indices), dtype=bool)
        if max_ax is not None and axial_freq is not None:
            keep &= axial_freq[indices] < max_ax
        if max_cyc is not None and cyc_freq is not None:
            keep &= cyc_freq[indices] < max_cyc
        return indices[keep]
