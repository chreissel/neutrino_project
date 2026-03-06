#!/usr/bin/env python
"""
evaluate_full_track.py – sliding-window parameter estimation over a full track.

A model trained on segments of length `segment_length` (e.g. 8 192 samples) is
applied to every non-overlapping window of that length across a longer track,
yielding one (energy_eV, pitch_angle_deg) estimate per segment.  Results are
written to a CSV file and visualised in a summary figure.

Typical usage
-------------
python scripts/evaluate_full_track.py \\
    --checkpoint runs/my_run/lightning_logs/abc/checkpoints/last.ckpt \\
    --config     runs/my_run/config.yaml \\
    --data       /path/to/data.hdf5 \\
    --track-idx  0 \\
    --n-segments 8 \\
    --output-dir results/full_track_eval

If the HDF5 row for `output_ts_I` / `output_ts_Q` is exactly `segment_length`
samples long you will get one segment (the standard evaluation case).  If the
rows are longer you get one estimate per window of `segment_length` samples.

Normalisation statistics (mean / std of the target variables) are computed on
the fly from the same data file and must therefore match the training data.
Pass `--norm-stats <file.npz>` to supply pre-computed stats instead (see
`--save-norm-stats`).
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

# ── make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt

from src.utils.transforms import fft_func_IQ_complex_channels
from src.utils.noise import noise_model, bandpass_filter


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def _build_encoder(enc_class: str, enc_init: dict):
    from src.models.networks import (
        S4DModel, S4D_CNN_Model, S4D_S4D_Model, S4D_S4D_GatedModel,
    )
    # normalise the class path (notebooks use 'models.networks.*',
    # the package uses 'src.models.networks.*')
    short = enc_class.split(".")[-1]
    constructors = {
        "S4DModel":           S4DModel,
        "S4D_CNN_Model":      S4D_CNN_Model,
        "S4D_S4D_Model":      S4D_S4D_Model,
        "S4D_S4D_GatedModel": S4D_S4D_GatedModel,
    }
    if short not in constructors:
        raise ValueError(
            f"Unknown encoder class '{enc_class}'. "
            f"Supported: {list(constructors)}"
        )
    return constructors[short](**enc_init)


def load_model_from_config(checkpoint: str, config: dict, device: str):
    """Return (lit_model, is_dual, apply_fft, loss_type)."""
    from src.models.model import LitS4Model, LitS4DualModel

    model_class = config["model"]["class_path"]
    model_init  = config["model"]["init_args"]
    enc_cfg     = model_init["encoder"]

    encoder = _build_encoder(enc_cfg["class_path"], enc_cfg["init_args"])

    short_class = model_class.split(".")[-1]
    if short_class == "LitS4Model":
        lit = LitS4Model.load_from_checkpoint(
            checkpoint, encoder=encoder, map_location=device
        )
        is_dual   = False
        apply_fft = bool(model_init.get("apply_fft", False))
    elif short_class == "LitS4DualModel":
        lit = LitS4DualModel.load_from_checkpoint(
            checkpoint, encoder=encoder, map_location=device
        )
        is_dual   = True
        apply_fft = True   # dual model always uses both ts and fft
    else:
        raise ValueError(f"Unsupported model class '{model_class}'.")

    lit = lit.to(device).eval()
    loss_type = model_init.get("loss", "MSELoss")
    return lit, is_dual, apply_fft, loss_type


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_norm_stats(hdf5_paths: list, variables: list) -> tuple:
    """Welford online mean / std over *all* rows in the supplied HDF5 files."""
    n_vars = len(variables)
    count, mean, M2 = 0, np.zeros(n_vars, np.float64), np.zeros(n_vars, np.float64)
    for path in hdf5_paths:
        with h5py.File(path, "r") as f:
            data = np.stack([f[v][:] for v in variables], axis=1)
        for row in data:
            count += 1
            delta  = row - mean
            mean  += delta / count
            M2    += delta * (row - mean)
    return mean.astype(np.float32), np.sqrt(M2 / count).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Per-segment preprocessing  (mirrors Project8Sim.__getitem__)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_segment(
    seg_I: np.ndarray,
    seg_Q: np.ndarray,
    noise_const: float = 1.0,
    apply_filter: bool = False,
    norm: bool = True,
    freq_transform: str = "fft",
) -> tuple:
    """
    Preprocess a single I/Q segment identically to the training dataset.

    Returns
    -------
    ts  : (L, 2) float32 tensor  – normalised time-series
    fft : (L, 2) float32 tensor  – normalised FFT channels  (or None)
    """
    L       = len(seg_I)
    X_clean = np.stack([seg_I, seg_Q], axis=-1).astype(np.float32)  # (L, 2)
    X_ts    = X_clean.copy()

    for j in range(2):
        noise = noise_model(L, noise_const)
        Xn    = X_clean[:, j] + noise
        if apply_filter:
            Xn = bandpass_filter(Xn)
        X_ts[:, j] = Xn / (np.std(Xn) + 1e-8) if norm else Xn

    ts = torch.tensor(X_ts, dtype=torch.float32)

    if freq_transform is None:
        return ts, None

    if freq_transform in ("fft", "both"):
        real, imag = fft_func_IQ_complex_channels(X_ts[:, 0], X_ts[:, 1])
        stack      = np.stack([real, imag], axis=1)          # (L, 2)
        mu         = np.mean(stack, axis=0)                  # (2,)
        std        = np.std(stack,  axis=0)                  # (2,)
        X_fft      = np.zeros_like(X_ts)
        X_fft[:, 0] = (real - mu[0]) / (std[0] + 1e-8)
        X_fft[:, 1] = (imag - mu[1]) / (std[1] + 1e-8)
        return ts, torch.tensor(X_fft, dtype=torch.float32)

    raise ValueError(
        f"freq_transform='{freq_transform}' is not supported in this script. "
        "Use 'fft', 'both', or None."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Core evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_track(
    hdf5_path: str,
    track_idx: int,
    model,
    is_dual: bool,
    apply_fft: bool,
    loss_type: str,
    inputs: list,
    variables: list,
    mu: np.ndarray,
    stds: np.ndarray,
    segment_length: int,
    n_segments: Optional[int],
    noise_const: float,
    apply_filter: bool,
    freq_transform: str,
    device: str,
) -> dict:
    """
    Slide over a single track and return per-segment predictions.

    Returns a dict with keys:
      preds_norm   – (S, n_out) model outputs (normalised space)
      preds        – (S, n_out) de-normalised predictions in physical units
      preds_var    – (S, n_out) predicted variances (only for GaussianNLLLoss)
      true         – (n_out,)  true parameter values in physical units
      segment_ids  – list of (start, end) sample indices
    """
    with h5py.File(hdf5_path, "r") as f:
        track_I  = f[inputs[0]][track_idx][:]   # full I channel
        track_Q  = f[inputs[1]][track_idx][:]   # full Q channel
        true_raw = np.array([f[v][track_idx] for v in variables], dtype=np.float32)

    total_len = len(track_I)
    max_segs  = total_len // segment_length

    if max_segs == 0:
        raise ValueError(
            f"Track length ({total_len} samples) is shorter than "
            f"segment_length ({segment_length}).  Choose a smaller "
            f"--segment-length."
        )

    if n_segments is not None:
        max_segs = min(max_segs, n_segments)

    preds_norm_list = []
    segment_ids     = []

    for s in range(max_segs):
        start = s * segment_length
        end   = start + segment_length
        seg_I = track_I[start:end].astype(np.float32)
        seg_Q = track_Q[start:end].astype(np.float32)

        ts, fft_t = preprocess_segment(
            seg_I, seg_Q,
            noise_const    = noise_const,
            apply_filter   = apply_filter,
            norm           = True,
            freq_transform = freq_transform,
        )

        ts_b = ts.unsqueeze(0).to(device)   # (1, L, 2)

        with torch.no_grad():
            if is_dual:
                fft_b     = fft_t.unsqueeze(0).to(device)
                raw_out   = model(ts_b, fft_b).cpu().numpy()[0]
            elif apply_fft and fft_t is not None:
                fft_b     = fft_t.unsqueeze(0).to(device)
                raw_out   = model(fft_b).cpu().numpy()[0]
            else:
                raw_out   = model(ts_b).cpu().numpy()[0]

        preds_norm_list.append(raw_out)
        segment_ids.append((start, end))

    preds_norm = np.array(preds_norm_list)     # (S, n_out) or (S, 2*n_out) for GaussianNLL

    # ── handle GaussianNLLLoss: output is [mean | log_var] ────────────────────
    n_vars     = len(variables)
    preds_var  = None
    if loss_type == "GaussianNLLLoss" and preds_norm.shape[1] == 2 * n_vars:
        import torch.nn.functional as F
        means_norm = preds_norm[:, :n_vars]
        log_vars   = preds_norm[:, n_vars:]
        # softplus to get positive variances, then scale to physical space
        preds_var  = np.log1p(np.exp(log_vars)) * stds ** 2   # (S, n_vars)
        preds_norm = means_norm

    preds = preds_norm * stds + mu             # de-normalise to physical units

    return dict(
        preds_norm  = preds_norm,
        preds       = preds,
        preds_var   = preds_var,
        true        = true_raw,
        segment_ids = segment_ids,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    results: dict,
    variables: list,
    track_idx: int,
    segment_length: int,
    output_dir: str,
) -> str:
    """
    One row of panels per target variable showing:
      •  per-segment prediction (dots + line)
      •  running mean (dashed)
      •  true value (horizontal red dashed line)
      •  ±1 σ band from uncertainty if available (GaussianNLLLoss)
    """
    preds       = results["preds"]          # (S, n_vars)
    preds_var   = results["preds_var"]      # (S, n_vars) or None
    true        = results["true"]           # (n_vars,)
    segment_ids = results["segment_ids"]

    S            = len(preds)
    seg_centers  = np.array([(s + e) / 2 for s, e in segment_ids])
    n_vars       = len(variables)

    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 4 * n_vars), squeeze=False)

    for vi, var in enumerate(variables):
        ax   = axes[vi, 0]
        unit = var.split("_")[-1]

        # per-segment predictions
        ax.plot(
            seg_centers, preds[:, vi], "o-",
            color="steelblue", markersize=6, lw=1.5,
            label="Segment prediction",
        )

        # per-segment ±1σ error bars (Gaussian NLL only)
        if preds_var is not None:
            ax.fill_between(
                seg_centers,
                preds[:, vi] - np.sqrt(preds_var[:, vi]),
                preds[:, vi] + np.sqrt(preds_var[:, vi]),
                alpha=0.25, color="steelblue", label="±1σ (model)",
            )

        # running mean
        running_mean = np.cumsum(preds[:, vi]) / np.arange(1, S + 1)
        ax.plot(
            seg_centers, running_mean, "s--",
            color="darkorange", markersize=5, lw=1.5,
            label=f"Running mean (final: {running_mean[-1]:.3f})",
        )

        # ensemble spread band (±1 std across segments)
        band_mu  = np.mean(preds[:, vi])
        band_std = np.std(preds[:, vi])
        ax.axhspan(
            band_mu - band_std, band_mu + band_std,
            alpha=0.10, color="steelblue",
        )

        # true value
        ax.axhline(
            true[vi], color="crimson", lw=2, ls="--",
            label=f"True: {true[vi]:.4f} [{unit}]",
        )

        # formatting
        ax.set_xlabel(
            f"Sample index (segment centre)\n"
            f"segment length = {segment_length} samples"
        )
        ax.set_ylabel(f"{var.replace('_', ' ')} [{unit}]")
        ax.set_title(
            f"Track {track_idx} – {var.replace('_', ' ')} "
            f"({S} segments × {segment_length} samples)"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"track{track_idx}_segment_predictions.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multi_track_summary(
    all_results: dict,     # {track_idx: results_dict}
    variables: list,
    segment_length: int,
    output_dir: str,
) -> str:
    """
    Residual (true − mean_pred) for every evaluated track, one panel per variable.
    Gives a quick overview of systematic bias across tracks.
    """
    n_vars = len(variables)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 4), squeeze=False)

    track_indices = sorted(all_results)
    for vi, var in enumerate(variables):
        ax   = axes[0, vi]
        unit = var.split("_")[-1]

        residuals = [
            all_results[ti]["true"][vi] - np.mean(all_results[ti]["preds"][:, vi])
            for ti in track_indices
        ]
        ax.bar(range(len(track_indices)), residuals, tick_label=track_indices, color="steelblue")
        ax.axhline(0, color="crimson", lw=1.5, ls="--")
        ax.set_xlabel("Track index")
        ax.set_ylabel(f"true − mean_pred [{unit}]")
        ax.set_title(f"{var.replace('_', ' ')} residuals (L={segment_length})")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "multi_track_residuals.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _hdf5_paths_from_arg(data_arg: str) -> list:
    """Accept either a single HDF5 file or a directory of HDF5 files."""
    p = Path(data_arg)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted(
            str(f) for f in p.iterdir()
            if f.suffix in (".hdf5", ".h5")
        )
        if not files:
            sys.exit(f"No HDF5 files found in directory: {p}")
        return files
    sys.exit(f"--data path does not exist: {p}")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a trained model on non-overlapping track segments. "
            "Produces a CSV of per-segment predictions and a summary figure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # required
    p.add_argument("--checkpoint",     required=True,
                   help="Path to Lightning .ckpt file")
    p.add_argument("--config",         required=True,
                   help="Path to the training config.yaml used to train the model")
    p.add_argument("--data",           required=True,
                   help="HDF5 file *or* directory of HDF5 files to evaluate on")

    # track selection
    p.add_argument("--track-idx",      type=int, nargs="+", default=[0],
                   help="One or more track (row) indices in the HDF5 file(s) to evaluate")

    # segmentation
    p.add_argument("--segment-length", type=int, default=None,
                   help="Samples per segment (defaults to 'cutoff' from config)")
    p.add_argument("--n-segments",     type=int, default=None,
                   help="Maximum number of segments per track (default: all that fit)")

    # preprocessing overrides
    p.add_argument("--noise-const",    type=float, default=None,
                   help="Noise amplitude multiplier (default: value from config)")
    p.add_argument("--no-noise",       action="store_true",
                   help="Set noise_const=0 (evaluate on clean signal only)")
    p.add_argument("--no-filter",      action="store_true",
                   help="Disable the bandpass filter regardless of config setting")

    # normalisation statistics
    p.add_argument("--norm-stats",     default=None,
                   help="Path to a .npz file with 'mu' and 'stds' arrays. "
                        "If not provided they are computed from --data.")
    p.add_argument("--save-norm-stats", default=None,
                   help="Save the computed norm stats to this .npz file for reuse")

    # output
    p.add_argument("--output-dir",     default="results/full_track_eval",
                   help="Directory for CSV files and figures")
    p.add_argument("--device",         default=None,
                   help="'cpu' or 'cuda' (auto-detected when omitted)")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg      = cfg["data"]["init_args"]
    variables     = data_cfg["variables"]
    inputs        = data_cfg["inputs"]
    freq_transform = data_cfg.get("freq_transform", "fft")

    segment_length = args.segment_length or data_cfg.get("cutoff", 8192)

    noise_const = 0.0 if args.no_noise else (
        args.noise_const if args.noise_const is not None
        else data_cfg.get("noise_const", 1.0)
    )
    apply_filter = (
        data_cfg.get("apply_filter", False) and not args.no_filter
    )

    print(f"Segment length  : {segment_length} samples")
    print(f"Noise constant  : {noise_const}")
    print(f"Freq transform  : {freq_transform}")
    print(f"Variables       : {variables}")

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.checkpoint}")
    model, is_dual, apply_fft, loss_type = load_model_from_config(
        args.checkpoint, cfg, device
    )
    print(f"Model type      : {'dual-branch' if is_dual else 'single-branch'}")
    print(f"Input mode      : {'FFT' if (apply_fft and not is_dual) else ('TS+FFT' if is_dual else 'time-series')}")
    print(f"Loss type       : {loss_type}")

    # ── find HDF5 files ───────────────────────────────────────────────────────
    hdf5_files = _hdf5_paths_from_arg(args.data)
    print(f"\nHDF5 files ({len(hdf5_files)}):")
    for fp in hdf5_files:
        print(f"  {fp}")

    # ── normalisation stats ───────────────────────────────────────────────────
    if args.norm_stats:
        ns   = np.load(args.norm_stats)
        mu   = ns["mu"].astype(np.float32)
        stds = ns["stds"].astype(np.float32)
        print(f"\nNorm stats loaded from: {args.norm_stats}")
    else:
        print("\nComputing normalisation statistics from data …")
        mu, stds = compute_norm_stats(hdf5_files, variables)
        print(f"  mu   = {mu}")
        print(f"  stds = {stds}")
        if args.save_norm_stats:
            np.savez(args.save_norm_stats, mu=mu, stds=stds)
            print(f"  Saved to: {args.save_norm_stats}")

    # ── output directory ──────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── evaluate each requested track ─────────────────────────────────────────
    all_results = {}

    for track_idx in args.track_idx:
        # determine which file contains this row index
        hdf5_path = hdf5_files[0]   # extend to multi-file if needed
        with h5py.File(hdf5_path, "r") as f:
            total_len = f[inputs[0]][track_idx].shape[0]

        max_possible = total_len // segment_length
        n_segs_used  = min(max_possible, args.n_segments) if args.n_segments else max_possible
        print(
            f"\n── Track {track_idx} ──────────────────────────────────────────"
        )
        print(f"  Track length     : {total_len} samples")
        print(f"  Segments possible: {max_possible}  (using {n_segs_used})")

        results = evaluate_track(
            hdf5_path      = hdf5_path,
            track_idx      = track_idx,
            model          = model,
            is_dual        = is_dual,
            apply_fft      = apply_fft,
            loss_type      = loss_type,
            inputs         = inputs,
            variables      = variables,
            mu             = mu,
            stds           = stds,
            segment_length = segment_length,
            n_segments     = args.n_segments,
            noise_const    = noise_const,
            apply_filter   = apply_filter,
            freq_transform = freq_transform,
            device         = device,
        )
        all_results[track_idx] = results

        preds = results["preds"]
        true  = results["true"]
        S     = len(preds)

        # print summary table
        col = 25
        print(f"\n  {'Variable':<{col}}  {'True':>12}  {'Mean pred':>12}  "
              f"{'Std pred':>10}  {'Residual':>10}")
        print("  " + "-" * (col + 50))
        for vi, var in enumerate(variables):
            mean_pred = np.mean(preds[:, vi])
            std_pred  = np.std(preds[:, vi])
            residual  = true[vi] - mean_pred
            print(
                f"  {var:<{col}}  {true[vi]:>12.4f}  {mean_pred:>12.4f}  "
                f"{std_pred:>10.4f}  {residual:>10.4f}"
            )

        # ── save CSV ──────────────────────────────────────────────────────────
        csv_path = os.path.join(
            args.output_dir, f"track{track_idx}_predictions.csv"
        )
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            header = (
                ["segment", "start_sample", "end_sample"]
                + [f"pred_{v}" for v in variables]
                + ([f"pred_var_{v}" for v in variables] if results["preds_var"] is not None else [])
                + [f"true_{v}" for v in variables]
            )
            writer.writerow(header)
            for s, (start, end) in enumerate(results["segment_ids"]):
                row = [s, start, end] + list(preds[s])
                if results["preds_var"] is not None:
                    row += list(results["preds_var"][s])
                row += list(true)
                writer.writerow(row)
        print(f"\n  CSV  → {csv_path}")

        # ── per-track plot ────────────────────────────────────────────────────
        fig_path = plot_results(
            results, variables, track_idx, segment_length, args.output_dir
        )
        print(f"  Plot → {fig_path}")

    # ── multi-track residual summary (only when >1 track) ─────────────────────
    if len(args.track_idx) > 1:
        summary_path = plot_multi_track_summary(
            all_results, variables, segment_length, args.output_dir
        )
        print(f"\nMulti-track residuals → {summary_path}")

    print(f"\nDone.  Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
