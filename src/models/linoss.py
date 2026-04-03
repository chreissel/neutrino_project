"""LinOSS PyTorch integration layer.

The core SSM computation (``apply_linoss_im`` / ``apply_linoss_imex`` +
``jax.lax.associative_scan``) is executed by the **original JAX
implementation** in ``linoss_jax.py``.  A thin ``torch.autograd.Function``
bridge handles tensor conversion and gradient propagation so that PyTorch's
AdamW optimizer can update the parameters as normal.

Architecture of the bridge
--------------------------
::

    PyTorch nn.Module (LinOSSLayer)
       │  stores parameters as nn.Parameter
       │
       ▼
    _LinOSSFunction.apply(u_seq, A_diag, B, C, D, steps, discretization)
       │  forward : PyTorch → numpy → JAX → jax.vjp → store vjp_fn → numpy → PyTorch
       │  backward: call stored vjp_fn(grad_output) → numpy → PyTorch
       │
       ▼
    linoss_layer_apply(...)          ← linoss_jax.py  (unchanged original)
       │  apply_linoss_im / apply_linoss_imex
       │  jax.lax.associative_scan  (parallel prefix scan, JIT-compiled)
       ▼
    (B, L, H) output JAX array

The SiGLU output-mixing step (analogous to S4D's output_linear Conv+GLU) and
the GELU activation are applied in **PyTorch** after the JAX call, consistent
with the original ``LinOSSBlock`` block structure.

``LinOSSLayer`` exposes the same ``(B, H, L) → (y, None)`` interface as S4D
so that it is a drop-in replacement in ``LinOSSModel`` / ``LinOSSSeq2SeqModel``
/ ``LinOSSCombinedModel``.

Dependencies
------------
Requires ``jax`` and ``equinox``::

    pip install "jax[cuda12]" equinox   # GPU
    pip install jax equinox             # CPU-only

A clear ``RuntimeError`` is raised at first use if these are missing.
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Optional JAX import — fail loudly at *use time*, not import time
# ---------------------------------------------------------------------------
try:
    import functools
    import jax
    import jax.numpy as jnp
    from src.models.linoss_jax import linoss_layer_apply
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _require_jax():
    if not _JAX_AVAILABLE:
        raise RuntimeError(
            "JAX and equinox are required to use LinOSS models.\n"
            "Install with:  pip install 'jax[cuda12]' equinox   (GPU)\n"
            "            or  pip install jax equinox             (CPU)"
        )


# ---------------------------------------------------------------------------
# JIT-compiled batched JAX forward (one compiled version per discretization)
# ---------------------------------------------------------------------------

# We define these lazily so that importing this module does not require JAX.
_jax_fwd_cache: dict = {}

def _get_jax_forward(discretization: str):
    """Return (and cache) a JIT-compiled batched LinOSS forward function."""
    if discretization not in _jax_fwd_cache:
        # Capture discretization as a Python constant (not a traced value) by
        # defining a new function for each variant.
        @jax.jit
        def _fwd(params, u_batch):
            """params = (A_diag, B, C, D, steps);  u_batch = (B, L, H)."""
            A_diag, B, C, D, steps = params
            return jax.vmap(
                lambda x: linoss_layer_apply(
                    A_diag, B, C, D, steps, discretization, x
                )
            )(u_batch)

        _jax_fwd_cache[discretization] = _fwd
    return _jax_fwd_cache[discretization]


# ---------------------------------------------------------------------------
# torch.autograd.Function bridge
# ---------------------------------------------------------------------------

class _LinOSSFunction(torch.autograd.Function):
    """Bridges one batched LinOSS forward/backward call between JAX and PyTorch.

    Inputs (positional, passed via ``apply``):
        u          (B, L, H)  — input sequence, float32
        A_diag     (P,)
        B          (P, H, 2)
        C          (H, P, 2)
        D          (H,)
        steps      (P,)
        discretization  str  — ``'IM'`` or ``'IMEX'``

    Output:
        (B, L, H) float32 — SSM output + D skip (before GELU / SiGLU)
    """

    @staticmethod
    def forward(ctx, u, A_diag, B, C, D, steps, discretization):
        _fwd = _get_jax_forward(discretization)

        def _to_jax(t: torch.Tensor):
            # Always convert through CPU float32 numpy — works for both
            # CPU and CUDA PyTorch tensors; JAX runs on its own device.
            return jnp.array(t.detach().cpu().float().numpy())

        params_jax = (_to_jax(A_diag), _to_jax(B),
                      _to_jax(C),      _to_jax(D), _to_jax(steps))
        u_jax = _to_jax(u)

        needs_grad = any(ctx.needs_input_grad[:6])

        if needs_grad:
            # jax.vjp gives us the output AND a function to compute
            # (grad_params, grad_u) from grad_output in one call.
            output_jax, vjp_fn = jax.vjp(_fwd, params_jax, u_jax)
            ctx.vjp_fn = vjp_fn          # stored as a Python attr (not tensor)
        else:
            output_jax = _fwd(params_jax, u_jax)

        ctx.device = u.device
        ctx.dtype  = u.dtype

        out_np = np.array(output_jax)    # JAX → numpy (CPU)
        return torch.from_numpy(out_np).to(u.device).to(u.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        vjp_fn = ctx.vjp_fn
        dev, dt = ctx.device, ctx.dtype

        grad_jax = jnp.array(grad_output.detach().cpu().float().numpy())
        grad_params, grad_u = vjp_fn(grad_jax)

        def _to_torch(g):
            return torch.from_numpy(np.array(g)).to(dev).to(dt)

        return (
            _to_torch(grad_u),           # ∂L/∂u
            _to_torch(grad_params[0]),   # ∂L/∂A_diag
            _to_torch(grad_params[1]),   # ∂L/∂B
            _to_torch(grad_params[2]),   # ∂L/∂C
            _to_torch(grad_params[3]),   # ∂L/∂D
            _to_torch(grad_params[4]),   # ∂L/∂steps
            None,                        # discretization — not differentiable
        )


# ---------------------------------------------------------------------------
# PyTorch nn.Module wrapper (drop-in for S4D)
# ---------------------------------------------------------------------------

class LinOSSLayer(nn.Module):
    """PyTorch nn.Module wrapping the JAX ``LinOSSLayer``.

    Parameters are stored as ``nn.Parameter`` and updated by the PyTorch
    optimizer.  The forward pass calls the original JAX implementation via
    ``_LinOSSFunction``; gradients flow back through ``jax.vjp``.

    A SiGLU output-mixing step (two linear projections, mirrors S4D's
    ``output_linear = Conv1d + GLU``) is applied in PyTorch after the JAX
    kernel, keeping the PyTorch/JAX boundary at the SSM boundary.

    Interface: **(B, H, L) → (y, None)**  — identical to S4D(transposed=True).

    Parameters
    ----------
    d_model       : int    Feature / channel dimension H.
    ssm_size      : int    State-space dimension P (analogue of d_state in S4D).
    discretization: str    ``'IM'`` (default, dissipative) or ``'IMEX'``
                           (symplectic, preserves time-reversibility).
    dropout       : float  Dropout after GELU (inside the layer).
    lr            : float  Optional per-parameter LR override (same convention
                           as S4D — picked up by ``configure_optimizers``).
    """

    def __init__(self,
                 d_model: int,
                 ssm_size: int = 64,
                 discretization: str = 'IM',
                 dropout: float = 0.0,
                 lr=None):
        super().__init__()
        _require_jax()
        H, P = d_model, ssm_size
        self.H = H
        self.P = P
        self.discretization = discretization

        # --- SSM parameters (matching LinOSSLayer.__init__ initialisations) ---
        self.A_diag    = nn.Parameter(torch.rand(P))
        std_B = 1.0 / math.sqrt(H)
        self.B         = nn.Parameter(torch.rand(P, H, 2) * 2 * std_B - std_B)
        std_C = 1.0 / math.sqrt(P)
        self.C         = nn.Parameter(torch.rand(H, P, 2) * 2 * std_C - std_C)
        self.D         = nn.Parameter(torch.empty(H).normal_(std=1.0))
        self.steps     = nn.Parameter(torch.rand(P))

        # --- Post-SSM mixing in PyTorch (mirrors LinOSSBlock's GLU) ----------
        self.out_proj1 = nn.Linear(H, H, bias=True)
        self.out_proj2 = nn.Linear(H, H, bias=True)
        self.drop      = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Optional per-parameter LR (picked up by configure_optimizers)
        if lr is not None:
            for p in [self.A_diag, self.B, self.C, self.D, self.steps]:
                p._optim = {"weight_decay": 0.0, "lr": lr}

    def forward(self, u, **kwargs):
        """
        u : (B, H, L)
        Returns (y, None) with y : (B, H, L).
        """
        u_seq = u.permute(0, 2, 1).contiguous()   # (B, H, L) → (B, L, H)

        # --- JAX SSM kernel (forward + backward registered via autograd) ---
        ys = _LinOSSFunction.apply(
            u_seq,
            self.A_diag, self.B, self.C, self.D, self.steps,
            self.discretization,
        )   # (B, L, H)

        # --- Post-SSM: GELU + SiGLU mixing (pure PyTorch) ---
        ys = self.drop(F.gelu(ys))
        ys = self.out_proj1(ys) * torch.sigmoid(self.out_proj2(ys))

        return ys.permute(0, 2, 1), None   # (B, H, L), None

# LinOSSModel / LinOSSSeq2SeqModel / LinOSSCombinedModel are defined in
# networks.py (which imports LinOSSLayer from this module).  All config
# class_path references should use src.models.networks.*

