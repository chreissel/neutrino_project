"""Pure-JAX training script for the combined LinOSS denoising + regression task.

Usage
-----
    python scripts/train_linoss_jax.py --config configs/config_combined_linoss_jax.yaml

    # override individual options
    python scripts/train_linoss_jax.py \\
        --config configs/config_combined_linoss_jax.yaml \\
        --batch_size 64 --max_epochs 50

All paths default to the values in the YAML config; CLI flags override them.

Pipeline
--------
1.  ``CombinedDataModuleJAX``  builds train / val / test splits (80/10/10).
2.  ``LinOSSCombinedJAX``      is constructed via ``eqx.nn.make_with_state``
    so BatchNorm running stats are handled correctly.
3.  ``optax.adamw``            is used as the optimizer.
4.  Each training step JIT-compiles via ``eqx.filter_jit``; gradients flow
    through the JAX kernel via ``eqx.filter_value_and_grad``.
5.  Optional WandB logging (disabled if ``wandb_project`` is ``null``).
6.  Checkpoints saved via ``eqx.tree_serialise_leaves`` every epoch (best
    val-loss model kept separately).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import yaml

# Make sure the project root is on the path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.data.dataset_jax import CombinedDataModuleJAX
from src.models.linoss_jax import LinOSSCombinedJAX


# ---------------------------------------------------------------------------
# Loss / step functions
# ---------------------------------------------------------------------------

@eqx.filter_jit
def train_step(
    diff_model  : Any,
    static_model: Any,
    opt_state   : Any,
    optimizer   : optax.GradientTransformation,
    X_noisy     : jax.Array,   # (B, L, d_input)
    X_clean     : jax.Array,   # (B, L, d_input)
    y           : jax.Array,   # (B, d_output)
    state       : eqx.nn.State,
    key         : jax.Array,
    lambda_d    : float,
    lambda_r    : float,
):
    """One gradient step.  Returns (diff_model, opt_state, state, loss_d, loss_r)."""

    @eqx.filter_value_and_grad(has_aux=True)
    def _loss_fn(diff_model):
        model = eqx.combine(diff_model, static_model)
        x_denoised, y_pred, new_state = jax.vmap(
            model,
            axis_name="batch",
            in_axes=(0, None, None),
            out_axes=(0, 0, None),
        )(X_noisy, state, key)

        loss_d = jnp.mean((x_denoised - X_clean) ** 2)
        loss_r = jnp.mean((y_pred - y) ** 2)
        total  = lambda_d * loss_d + lambda_r * loss_r
        return total, (new_state, loss_d, loss_r)

    (total_loss, (new_state, loss_d, loss_r)), grads = _loss_fn(diff_model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(diff_model, eqx.is_array)
    )
    new_diff = eqx.apply_updates(diff_model, updates)
    return new_diff, new_opt_state, new_state, loss_d, loss_r


@eqx.filter_jit
def eval_step(
    diff_model  : Any,
    static_model: Any,
    X_noisy     : jax.Array,
    X_clean     : jax.Array,
    y           : jax.Array,
    state       : eqx.nn.State,
    key         : jax.Array,
):
    """Inference-only forward pass — BatchNorm in eval mode."""
    model = eqx.tree_inference(eqx.combine(diff_model, static_model), value=True)
    x_denoised, y_pred, _ = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, None),
        out_axes=(0, 0, None),
    )(X_noisy, state, key)
    loss_d = jnp.mean((x_denoised - X_clean) ** 2)
    loss_r = jnp.mean((y_pred - y) ** 2)
    return loss_d, loss_r


# ---------------------------------------------------------------------------
# LR / lambda schedules
# ---------------------------------------------------------------------------

def _make_schedule(cfg: Dict) -> optax.Schedule:
    stype = cfg.get('schedule_type', 'constant')
    start = float(cfg.get('start_value', 1.0))
    if stype == 'constant':
        return optax.constant_schedule(start)
    elif stype == 'linear':
        end        = float(cfg.get('end_value', start))
        max_epochs = int(cfg.get('max_epochs', 80))
        # steps is determined at call time; approximate via epoch count
        return lambda step: start + (end - start) * min(step / max_epochs, 1.0)
    elif stype == 'cosine':
        total = int(cfg.get('total_steps', 1000))
        return optax.cosine_decay_schedule(start, decay_steps=total, alpha=0.0)
    else:
        raise ValueError(f"Unknown schedule_type: {stype!r}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(path: str, diff_model, opt_state, state, epoch: int, val_loss: float):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    eqx.tree_serialise_leaves(path + '.eqx', diff_model)
    meta = {'epoch': epoch, 'val_loss': float(val_loss)}
    with open(path + '.json', 'w') as f:
        json.dump(meta, f)


def _load_checkpoint(path: str, model_template):
    return eqx.tree_deserialise_leaves(path + '.eqx', model_template)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Dict):
    # ── Logging ──────────────────────────────────────────────────────────
    use_wandb = bool(cfg.get('wandb_project'))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg['wandb_project'],
                name=cfg.get('run_name', None),
                config=cfg,
            )
        except ImportError:
            print("WandB not installed — logging disabled.", flush=True)
            use_wandb = False

    # ── Data ─────────────────────────────────────────────────────────────
    dm = CombinedDataModuleJAX(
        data_dir        = cfg['data_dir'],
        inputs          = cfg['inputs'],
        variables       = cfg['variables'],
        observables     = cfg.get('observables', []),
        cutoff          = int(cfg.get('cutoff', 4000)),
        noise_const     = float(cfg.get('noise_const', 1.0)),
        apply_filter    = bool(cfg.get('apply_filter', False)),
        max_axial_freq  = cfg.get('max_axial_freq', None),
        max_cyc_freq    = cfg.get('max_cyc_freq', None),
        seed            = int(cfg.get('split_seed', 42)),
    )
    print(f"Train: {len(dm.train)}  Val: {len(dm.val)}  Test: {len(dm.test)}", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────
    key = jr.PRNGKey(int(cfg.get('seed', 0)))
    key, model_key = jr.split(key)

    model_cfg = cfg.get('model', {})
    model, state = eqx.nn.make_with_state(LinOSSCombinedJAX)(
        d_input                  = dm.d_input,
        d_output                 = dm.d_output,
        denoiser_d_model         = int(model_cfg.get('denoiser_d_model', 128)),
        denoiser_n_layers        = int(model_cfg.get('denoiser_n_layers', 6)),
        denoiser_ssm_size        = int(model_cfg.get('denoiser_ssm_size', 64)),
        denoiser_discretization  = str(model_cfg.get('denoiser_discretization', 'IM')),
        denoiser_dropout         = float(model_cfg.get('denoiser_dropout', 0.0)),
        regressor_d_model        = int(model_cfg.get('regressor_d_model', 128)),
        regressor_n_layers       = int(model_cfg.get('regressor_n_layers', 6)),
        regressor_ssm_size       = int(model_cfg.get('regressor_ssm_size', 64)),
        regressor_discretization = str(model_cfg.get('regressor_discretization', 'IM')),
        regressor_dropout        = float(model_cfg.get('regressor_dropout', 0.0)),
        regressor_fc_hidden      = list(model_cfg.get('regressor_fc_hidden', [64, 32])),
        key                      = model_key,
    )

    # ── Optimizer ────────────────────────────────────────────────────────
    lr         = float(cfg.get('learning_rate', 1e-3))
    wd         = float(cfg.get('weight_decay',  1e-2))
    max_epochs = int(cfg.get('max_epochs', 80))
    batch_size = int(cfg.get('batch_size', 128))

    optimizer = optax.adamw(learning_rate=lr, weight_decay=wd)
    diff_model, static_model = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(eqx.filter(diff_model, eqx.is_array))

    # ── Lambda schedules ─────────────────────────────────────────────────
    d_sched_cfg = cfg.get('lambda_denoise_schedule', {'schedule_type': 'constant', 'start_value': 1.0})
    r_sched_cfg = cfg.get('lambda_regress_schedule', {'schedule_type': 'constant', 'start_value': 1.0})
    if 'max_epochs' not in d_sched_cfg:
        d_sched_cfg['max_epochs'] = max_epochs
    if 'max_epochs' not in r_sched_cfg:
        r_sched_cfg['max_epochs'] = max_epochs

    d_schedule = _make_schedule(d_sched_cfg)
    r_schedule = _make_schedule(r_sched_cfg)

    # ── Output directory ─────────────────────────────────────────────────
    out_dir = str(cfg.get('output_dir', 'runs/linoss_jax'))
    os.makedirs(out_dir, exist_ok=True)
    ckpt_last = os.path.join(out_dir, 'last')
    ckpt_best = os.path.join(out_dir, 'best')
    best_val  = float('inf')

    # ── Training loop ────────────────────────────────────────────────────
    steps_per_epoch = max(1, len(dm.train) // batch_size)
    key, data_key = jr.split(key)

    for epoch in range(max_epochs):
        lambda_d = float(d_schedule(epoch))
        lambda_r = float(r_schedule(epoch))

        # Training
        train_losses_d, train_losses_r = [], []
        key, epoch_key = jr.split(key)
        key, data_key  = jr.split(data_key)

        for X_noisy, X_clean, y, _ in dm.train.loop_epoch(batch_size, data_key, drop_last=True):
            key, step_key = jr.split(key)
            diff_model, opt_state, state, ld, lr_ = train_step(
                diff_model, static_model, opt_state, optimizer,
                X_noisy, X_clean, y, state, step_key, lambda_d, lambda_r,
            )
            train_losses_d.append(float(ld))
            train_losses_r.append(float(lr_))

        train_loss_d = float(np.mean(train_losses_d)) if train_losses_d else float('nan')
        train_loss_r = float(np.mean(train_losses_r)) if train_losses_r else float('nan')
        train_total  = lambda_d * train_loss_d + lambda_r * train_loss_r

        # Validation
        val_losses_d, val_losses_r = [], []
        key, val_data_key = jr.split(key)
        for X_noisy, X_clean, y, _ in dm.val.loop_epoch(batch_size, val_data_key, drop_last=False):
            key, val_key = jr.split(key)
            ld, lr_ = eval_step(diff_model, static_model, X_noisy, X_clean, y, state, val_key)
            val_losses_d.append(float(ld))
            val_losses_r.append(float(lr_))

        val_loss_d = float(np.mean(val_losses_d)) if val_losses_d else float('nan')
        val_loss_r = float(np.mean(val_losses_r)) if val_losses_r else float('nan')
        val_total  = lambda_d * val_loss_d + lambda_r * val_loss_r

        # Logging
        print(
            f"Epoch {epoch + 1:>4}/{max_epochs}  "
            f"train loss={train_total:.4f} (d={train_loss_d:.4f} r={train_loss_r:.4f})  "
            f"val loss={val_total:.4f} (d={val_loss_d:.4f} r={val_loss_r:.4f})  "
            f"λ_d={lambda_d:.3f} λ_r={lambda_r:.3f}",
            flush=True,
        )

        if use_wandb:
            wandb.log({
                'epoch'         : epoch + 1,
                'train/loss'    : train_total,
                'train/loss_d'  : train_loss_d,
                'train/loss_r'  : train_loss_r,
                'val/loss'      : val_total,
                'val/loss_d'    : val_loss_d,
                'val/loss_r'    : val_loss_r,
                'lambda_denoise': lambda_d,
                'lambda_regress': lambda_r,
            })

        # Checkpointing
        _save_checkpoint(ckpt_last, diff_model, opt_state, state, epoch + 1, val_total)
        if val_total < best_val:
            best_val = val_total
            _save_checkpoint(ckpt_best, diff_model, opt_state, state, epoch + 1, val_total)
            print(f"  ↑ new best val loss: {best_val:.4f}", flush=True)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}", flush=True)
    print(f"Checkpoints saved to: {out_dir}", flush=True)

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Train LinOSS combined model (pure JAX)")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    # Allow per-run CLI overrides of common fields
    parser.add_argument('--data_dir',    default=None)
    parser.add_argument('--batch_size',  type=int,   default=None)
    parser.add_argument('--max_epochs',  type=int,   default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--output_dir',  default=None)
    parser.add_argument('--seed',        type=int,   default=None)
    parser.add_argument('--wandb_project', default=None)
    return parser.parse_args()


def main():
    args = _parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for field in ('data_dir', 'batch_size', 'max_epochs', 'learning_rate',
                  'output_dir', 'seed', 'wandb_project'):
        val = getattr(args, field)
        if val is not None:
            cfg[field] = val

    train(cfg)


if __name__ == '__main__':
    main()
