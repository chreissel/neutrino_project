"""PyTorch port of LinOSS (Linear Oscillatory State-Space Models).

Reference:
    Rusch & Mishra (2025), "Oscillatory State-Space Models", ICLR 2025 (Oral).
    https://github.com/tk-rusch/linoss

The original implementation uses JAX/Equinox with jax.lax.associative_scan.
This port implements the identical algorithm via a log-depth parallel prefix
scan in pure PyTorch, preserving the same mathematical formulation for both
the IM (implicit) and IMEX (implicit-explicit / symplectic) discretizations.

Interface convention
--------------------
``LinOSSLayer`` accepts **(B, H, L)** input (channels-first, same as S4D with
``transposed=True``) and returns ``(y, None)``.  The ``None`` is a dummy state
slot kept for API compatibility with the existing S4D-based model code so that
``LinOSSLayer`` is a **drop-in replacement for S4D** in
``LinOSSModel`` / ``LinOSSSeq2SeqModel``.

Discretization variants
-----------------------
* **IM**   – implicit; eigenvalue magnitude ≤ 1 (dissipative, stable).
* **IMEX** – implicit-explicit (symplectic); det(M) = 1, |eigenvalues| = 1
              (conservative / oscillatory, preserves time-reversibility).
  IM is recommended for denoising + regression tasks; IMEX for forecasting.

Parallel scan
-------------
Since the LinOSS transition matrix M is **constant across time** (it depends
only on the learned parameters A_diag and steps, not on the input), M is
stored as shape (L, 4P) without the batch dimension.  The forcing b_el is
(B, L, 2P).  The log-depth doubling algorithm runs in ceil(log2 L) rounds,
each round being a fully data-parallel tensor operation — no Python loops
over the sequence length at inference time.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Parallel prefix scan helpers
# ---------------------------------------------------------------------------

def _combine(M_i: torch.Tensor, b_i: torch.Tensor,
             M_j: torch.Tensor, b_j: torch.Tensor,
             P: int):
    """LinOSS associative binary operator  (M_j, b_j) ∘ (M_i, b_i).

    Computes the composition of two (matrix, state) pairs where each matrix
    is represented as four diagonal blocks [M11, M12, M21, M22] of size P:

        M_new = M_j @ M_i      (2×2 block-diagonal matrix product)
        b_new = M_j @ b_i + b_j

    Shapes
    ------
    M_i, M_j : (..., 4P)  — may differ in batch dimensions (broadcasting OK)
    b_i, b_j : (..., 2P)
    Returns  : (M_new: ..., 4P), (b_new: ..., 2P)
    """
    iA = M_i[..., 0*P:1*P];  iB = M_i[..., 1*P:2*P]
    iC = M_i[..., 2*P:3*P];  iD = M_i[..., 3*P:4*P]
    jA = M_j[..., 0*P:1*P];  jB = M_j[..., 1*P:2*P]
    jC = M_j[..., 2*P:3*P];  jD = M_j[..., 3*P:4*P]

    M_new = torch.cat([
        jA * iA + jB * iC,   # new M11 block
        jA * iB + jB * iD,   # new M12 block
        jC * iA + jD * iC,   # new M21 block
        jC * iB + jD * iD,   # new M22 block
    ], dim=-1)

    bi1, bi2 = b_i[..., :P], b_i[..., P:]
    b_new = torch.cat([
        jA * bi1 + jB * bi2,
        jC * bi1 + jD * bi2,
    ], dim=-1) + b_j

    return M_new, b_new


def _parallel_scan(M_el: torch.Tensor,
                   b_el: torch.Tensor,
                   P: int) -> torch.Tensor:
    """Inclusive prefix scan for the LinOSS linear recurrence.

    After the scan, ``b_el[..., t, :]`` contains the cumulative hidden state

        h_t = (M_t ∘ … ∘ M_0)(0)

    where ∘ is the LinOSS binary operator and the initial hidden state is 0.

    Uses the log-depth *doubling* algorithm:

        for k = 0, 1, …, ⌈log₂ L⌉ − 1:
            element[i] ← element[i − 2^k] ∘ element[i]   for i ≥ 2^k

    Because the transition matrix M is constant across the batch dimension,
    M_el has no batch axis (shape (L, 4P)) while b_el is (B, L, 2P).
    Broadcasting in ``_combine`` handles the mixed shapes correctly.

    Parameters
    ----------
    M_el : (L, 4P)   transition matrices (constant across batch)
    b_el : (B, L, 2P) input forcing terms
    P    : int        SSM state size

    Returns
    -------
    (B, L, 2P)  hidden state sequence  [x1_t, x2_t]
    """
    L = M_el.shape[0]

    if L == 1:
        return b_el  # single step — no scan needed

    levels = math.ceil(math.log2(L))

    for k in range(levels):
        stride = 1 << k  # 2^k

        # Build identity-padded left-shifted arrays:
        #   M_prev[t] = M_el[t - stride]  for t >= stride
        #   M_prev[t] = I (identity)       for t <  stride
        id_M = M_el.new_zeros(stride, 4 * P)
        id_M[:, 0*P:1*P] = 1.0   # M11 = 1
        id_M[:, 3*P:4*P] = 1.0   # M22 = 1
        M_prev = torch.cat([id_M, M_el[:L - stride]], dim=0)  # (L, 4P)

        id_b = b_el.new_zeros(b_el.shape[0], stride, 2 * P)
        b_prev = torch.cat([id_b, b_el[:, :L - stride]], dim=1)  # (B, L, 2P)

        M_new, b_new = _combine(M_prev, b_prev, M_el, b_el, P)

        # Only overwrite positions t >= stride; keep earlier positions unchanged
        mask = (torch.arange(L, device=M_el.device) >= stride)
        M_el = torch.where(mask.view(L, 1),   M_new, M_el)
        b_el = torch.where(mask.view(1, L, 1), b_new, b_el)

    return b_el  # (B, L, 2P) — contains [x1_t, x2_t] for each t


# ---------------------------------------------------------------------------
# Core LinOSS layer
# ---------------------------------------------------------------------------

class LinOSSLayer(nn.Module):
    """Single LinOSS layer — drop-in replacement for S4D.

    Implements the forced harmonic-oscillator linear recurrence:

        [x1_{t+1}]   =   M  @  [x1_t]   +   F_t(u_t)
        [x2_{t+1}]            [x2_t]

    where M is the 2P×2P block-diagonal transition matrix from either the
    IM or IMEX discretization of the second-order ODE  x'' + A x = Bu.

    The output at each time step is  y_t = C @ x2_t + D * u_t  (skip).

    A SiGLU mixing layer (analogous to S4D's ``output_linear``) is applied
    before returning.

    Interface
    ---------
    Input  : (B, H, L)   channels-first (``transposed=True`` convention)
    Output : ``(y, None)``  with  y : (B, H, L)

    Parameters
    ----------
    d_model       : int    Feature / channel dimension H.
    ssm_size      : int    State-space dimension P (analogous to d_state in S4D).
    discretization: str    ``'IM'`` (default) or ``'IMEX'``.
    dropout       : float  Dropout after GELU activation (inside the layer).
    lr            : float  Optional per-parameter LR override (mirrors S4D).
    """

    def __init__(self,
                 d_model: int,
                 ssm_size: int = 64,
                 discretization: str = 'IM',
                 dropout: float = 0.0,
                 lr=None):
        super().__init__()
        H, P = d_model, ssm_size
        self.H = H
        self.P = P
        self.discretization = discretization

        # ── SSM parameters ────────────────────────────────────────────────
        # A_diag: oscillator frequencies, constrained ≥ 0 via F.relu at runtime
        self.A_diag = nn.Parameter(torch.rand(P))

        # B: complex input matrix (P, H), stored as real (P, H, 2)
        std_B = 1.0 / math.sqrt(H)
        self.B = nn.Parameter(torch.rand(P, H, 2) * 2 * std_B - std_B)

        # C: complex output matrix (H, P), stored as real (H, P, 2)
        std_C = 1.0 / math.sqrt(P)
        self.C = nn.Parameter(torch.rand(H, P, 2) * 2 * std_C - std_C)

        # D: skip-connection scale (H,)
        self.D = nn.Parameter(torch.randn(H))

        # steps: per-state discretization step sizes, learned via sigmoid → (0,1)
        self.steps_raw = nn.Parameter(torch.rand(P))

        # ── Post-SSM mixing (SiGLU) ───────────────────────────────────────
        # Mirrors S4D's output_linear (Conv1d + GLU); keeps H→H dimension.
        self.out_proj1 = nn.Linear(H, H, bias=True)
        self.out_proj2 = nn.Linear(H, H, bias=True)

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Attach optional per-parameter LR override (used by configure_optimizers)
        if lr is not None:
            for p in [self.A_diag, self.B, self.C, self.D, self.steps_raw]:
                p._optim = {"weight_decay": 0.0, "lr": lr}

    # ------------------------------------------------------------------
    def forward(self, u, **kwargs):
        """Forward pass.

        Parameters
        ----------
        u : (B, H, L)

        Returns
        -------
        y    : (B, H, L)
        None : dummy state (API compatibility with S4D)
        """
        B_b, H, L = u.shape
        P = self.P

        A_diag = F.relu(self.A_diag)                          # (P,)  ≥ 0
        steps  = torch.sigmoid(self.steps_raw)                # (P,)  ∈ (0,1)
        B_c    = torch.view_as_complex(self.B.contiguous())   # (P, H) complex
        C_c    = torch.view_as_complex(self.C.contiguous())   # (H, P) complex

        u_seq = u.permute(0, 2, 1)   # (B, H, L) → (B, L, H)

        if self.discretization == 'IM':
            ys = self._apply_im  (A_diag, B_c, C_c, u_seq, steps, B_b, L)
        else:
            ys = self._apply_imex(A_diag, B_c, C_c, u_seq, steps, B_b, L)

        # D skip connection (broadcast over batch and time)
        ys = ys + u_seq * self.D                              # (B, L, H)

        # GELU activation + dropout
        ys = self.drop(F.gelu(ys))                            # (B, L, H)

        # SiGLU output mixing (per-timestep; Linear applies to last dim)
        ys = self.out_proj1(ys) * torch.sigmoid(self.out_proj2(ys))  # (B, L, H)

        return ys.permute(0, 2, 1), None   # (B, H, L), None

    # ------------------------------------------------------------------
    def _build_M_el(self,
                    M11: torch.Tensor, M12: torch.Tensor,
                    M21: torch.Tensor, M22: torch.Tensor,
                    L: int) -> torch.Tensor:
        """Stack four (P,) diagonal blocks into (L, 4P) for the prefix scan."""
        P = self.P
        # cat → (4P,), repeat → (L, 4P) contiguous
        M_row = torch.cat([M11, M12, M21, M22])          # (4P,)
        return M_row.unsqueeze(0).repeat(L, 1)            # (L, 4P)

    def _apply_im(self, A_diag, B_c, C_c, u_seq, steps, B_b, L):
        """IM (implicit) discretization  →  (B, L, H)."""
        P = self.P

        schur = 1.0 / (1.0 + steps ** 2 * A_diag)         # (P,)
        M11 =  1.0 - steps ** 2 * A_diag * schur           # (P,)
        M12 = -steps * A_diag * schur                       # (P,)
        M21 =  steps * schur                                # (P,)
        M22 =  schur                                        # (P,)

        # Project input onto state space: (B, L, P) real
        Bu = torch.einsum('ph, blh -> blp',
                          B_c, u_seq.to(B_c.dtype)).real    # (B, L, P)

        b_el = torch.cat([M11 * steps * Bu,
                          M21 * steps * Bu], dim=-1)        # (B, L, 2P)
        M_el = self._build_M_el(M11, M12, M21, M22, L)     # (L, 4P)

        xs = _parallel_scan(M_el, b_el, P)                  # (B, L, 2P)
        x2 = xs[..., P:]                                    # (B, L, P)  velocity state

        return torch.einsum('hp, blp -> blh',
                            C_c, x2.to(C_c.dtype)).real     # (B, L, H)

    def _apply_imex(self, A_diag, B_c, C_c, u_seq, steps, B_b, L):
        """IMEX (symplectic) discretization  →  (B, L, H)."""
        P = self.P

        M11 =  torch.ones_like(A_diag)                     # (P,)
        M12 = -steps * A_diag                               # (P,)
        M21 =  steps * torch.ones_like(A_diag)             # (P,)
        M22 =  1.0 - steps ** 2 * A_diag                   # (P,)

        Bu = torch.einsum('ph, blh -> blp',
                          B_c, u_seq.to(B_c.dtype)).real    # (B, L, P)

        b_el = torch.cat([Bu * steps,
                          Bu * steps ** 2], dim=-1)         # (B, L, 2P)
        M_el = self._build_M_el(M11, M12, M21, M22, L)     # (L, 4P)

        xs = _parallel_scan(M_el, b_el, P)                  # (B, L, 2P)
        x2 = xs[..., P:]                                    # (B, L, P)

        return torch.einsum('hp, blp -> blh',
                            C_c, x2.to(C_c.dtype)).real     # (B, L, H)
