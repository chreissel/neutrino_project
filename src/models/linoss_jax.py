# Original LinOSS implementation — kept verbatim from:
#   https://github.com/tk-rusch/linoss/blob/main/models/LinOSS.py
#
# The only addition to this file is the ``linoss_layer_apply`` function at
# the bottom, which is a functional (stateless) version of
# ``LinOSSLayer.__call__`` required to interface with ``jax.vjp`` from the
# PyTorch training loop.  Every other symbol in this file is unchanged.

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
from jax.nn.initializers import normal
import math
from jax import random

def simple_uniform_init(rng, shape, std=1.):
    weights = random.uniform(rng, shape)*2.*std - std
    return weights

class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))

# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    N = A_i.size // 4
    iA_ = A_i[0 * N: 1 * N]
    iB_ = A_i[1 * N: 2 * N]
    iC_ = A_i[2 * N: 3 * N]
    iD_ = A_i[3 * N: 4 * N]
    jA_ = A_j[0 * N: 1 * N]
    jB_ = A_j[1 * N: 2 * N]
    jC_ = A_j[2 * N: 3 * N]
    jD_ = A_j[3 * N: 4 * N]
    A_new = jA_ * iA_ + jB_ * iC_
    B_new = jA_ * iB_ + jB_ * iD_
    C_new = jC_ * iA_ + jD_ * iC_
    D_new = jC_ * iB_ + jD_ * iD_
    Anew = jnp.concatenate([A_new, B_new, C_new, D_new])

    b_i1 = b_i[0:N]
    b_i2 = b_i[N:]

    new_b1 = jA_ * b_i1 + jB_ * b_i2
    new_b2 = jC_ * b_i1 + jD_ * b_i2
    new_b = jnp.concatenate([new_b1, new_b2])

    return Anew, new_b + b_j

def apply_linoss_im(A_diag, B, C_tilde, input_sequence, step):
    """Compute the LxH output of LinOSS-IM given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    schur_comp = 1. / (1. + step ** 2. * A_diag)
    M_IM_11 = 1. - step ** 2. * A_diag * schur_comp
    M_IM_12 = -1. * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp

    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])

    M_IM_elements = M_IM * jnp.ones((input_sequence.shape[0],
                                         4 * A_diag.shape[0]))

    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IM_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


def apply_linoss_imex(A_diag, B, C, input_sequence, step):
    """Compute the LxH output of of LinOSS-IMEX given an LxH input.
    Args:
        A_diag  (float32):   diagonal state matrix   (P,)
        B       (complex64): input matrix            (P, H)
        C       (complex64): output matrix           (H, P)
        input_sequence (float32): input sequence of features    (L, H)
        step    (float):     discretization time-step $\Delta_t$  (P,)
    Returns:
        outputs (float32): the SSM outputs (LinOSS_IMEX layer preactivations)      (L, H)
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)

    A_ = jnp.ones_like(A_diag)
    B_ = -1. * step * A_diag
    C_ = step
    D_ = 1. - (step ** 2.) * A_diag

    M_IMEX = jnp.concatenate([A_, B_, C_, D_])

    M_IMEX_elements = M_IMEX * jnp.ones((input_sequence.shape[0],
                                          4 * A_diag.shape[0]))

    F1 = Bu_elements * step
    F2 = Bu_elements * (step ** 2.)
    F = jnp.hstack((F1, F2))

    _, xs = jax.lax.associative_scan(binary_operator, (M_IMEX_elements, F))
    ys = xs[:, A_diag.shape[0]:]

    return jax.vmap(lambda x: (C @ x).real)(ys)

class LinOSSLayer(eqx.Module):
    A_diag: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    steps: jax.Array
    discretization: str

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        *,
        key
    ):

        B_key, C_key, D_key, A_key, step_key, key = jr.split(key, 6)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        self.B = simple_uniform_init(B_key,shape=(ssm_size, H, 2),std=1./math.sqrt(H))
        self.C = simple_uniform_init(C_key,shape=(H, ssm_size, 2),std=1./math.sqrt(ssm_size))
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key,shape=(ssm_size,))
        self.discretization = discretization

    def __call__(self, input_sequence):
        A_diag = nn.relu(self.A_diag)

        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]

        steps = nn.sigmoid(self.steps)
        if self.discretization == 'IMEX':
            ys = apply_linoss_imex(A_diag, B_complex, C_complex, input_sequence, steps)
        elif self.discretization == 'IM':
            ys = apply_linoss_im(A_diag, B_complex, C_complex, input_sequence, steps)
        else:
            print('Discretization type not implemented')

        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


class LinOSSBlock(eqx.Module):

    norm: eqx.nn.BatchNorm
    ssm: LinOSSLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
        self,
        ssm_size,
        H,
        discretization,
        drop_rate=0.05,
        *,
        key
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.ssm = LinOSSLayer(
            ssm_size,
            H,
            discretization,
            key=ssmkey,
        )
        self.glu = GLU(H, H, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute LinOSS block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)
        x = skip + x
        return x, state


class LinOSS(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: List[LinOSSBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        num_blocks,
        N,
        ssm_size,
        H,
        output_dim,
        classification,
        output_step,
        discretization,
        *,
        key
    ):

        linear_encoder_key, *block_keys, linear_layer_key, weightkey = jr.split(
            key, num_blocks + 3
        )
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)
        self.blocks = [
            LinOSSBlock(
                ssm_size,
                H,
                discretization,
                key=key,
            )
            for key in block_keys
        ]
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)
        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute LinOSS."""
        dropkeys = jr.split(key, len(self.blocks))
        x = jax.vmap(self.linear_encoder)(x)
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1 :: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))
        return x, state


# ---------------------------------------------------------------------------
# Functional interface — the only addition to the original file.
#
# ``linoss_layer_apply`` is a stateless version of ``LinOSSLayer.__call__``
# that accepts explicit parameter arrays instead of ``self``.  This allows
# calling the JAX kernel from ``jax.vjp`` in the PyTorch bridge without
# constructing an equinox module, while keeping every line of the original
# LinOSS computation unchanged.
# ---------------------------------------------------------------------------

def linoss_layer_apply(A_diag_raw, B, C, D, steps_raw, discretization,
                       input_sequence):
    """Functional (stateless) version of ``LinOSSLayer.__call__``.

    Identical computation to the method body; parameters are passed
    explicitly so this function can be differentiated with ``jax.vjp``.

    Args:
        A_diag_raw    (float32): raw (pre-relu) diagonal frequencies  (P,)
        B             (float32): input matrix, real-valued storage     (P, H, 2)
        C             (float32): output matrix, real-valued storage    (H, P, 2)
        D             (float32): skip-connection scale                 (H,)
        steps_raw     (float32): raw (pre-sigmoid) step sizes         (P,)
        discretization (str):   ``'IM'`` or ``'IMEX'``
        input_sequence (float32): input features                      (L, H)

    Returns:
        (float32): SSM output + D skip                                (L, H)
    """
    A_diag = jax.nn.relu(A_diag_raw)
    B_complex = B[..., 0] + 1j * B[..., 1]
    C_complex = C[..., 0] + 1j * C[..., 1]
    steps = jax.nn.sigmoid(steps_raw)

    if discretization == 'IMEX':
        ys = apply_linoss_imex(A_diag, B_complex, C_complex, input_sequence, steps)
    elif discretization == 'IM':
        ys = apply_linoss_im(A_diag, B_complex, C_complex, input_sequence, steps)
    else:
        raise ValueError(f'Unknown discretization: {discretization!r}')

    Du = jax.vmap(lambda u: D * u)(input_sequence)
    return ys + Du


# ---------------------------------------------------------------------------
# Equinox model classes for the pure-JAX training pipeline
# (used by scripts/train_linoss_jax.py and src/data/dataset_jax.py)
# ---------------------------------------------------------------------------

class LinOSSSeq2SeqJAX(eqx.Module):
    """Sequence-to-sequence denoiser built from ``LinOSSBlock`` layers.

    Maps ``(L, d_input) → (L, d_input)`` — used as the denoising stage of
    ``LinOSSCombinedJAX``.

    All ``LinOSSBlock`` layers contain ``eqx.nn.BatchNorm``; construct via
    ``eqx.nn.make_with_state`` to obtain the initial state object::

        model, state = eqx.nn.make_with_state(LinOSSSeq2SeqJAX)(...)
    """

    encoder : eqx.nn.Linear
    blocks  : list          # List[LinOSSBlock]
    decoder : eqx.nn.Linear

    def __init__(
        self,
        d_input        : int,
        d_model        : int,
        n_layers       : int,
        ssm_size       : int   = 64,
        discretization : str   = 'IM',
        dropout        : float = 0.0,
        *,
        key            : jax.Array,
    ):
        enc_key, *blk_keys, dec_key = jr.split(key, n_layers + 2)
        self.encoder = eqx.nn.Linear(d_input, d_model, key=enc_key)
        self.blocks  = [
            LinOSSBlock(ssm_size, d_model, discretization, drop_rate=dropout, key=k)
            for k in blk_keys
        ]
        self.decoder = eqx.nn.Linear(d_model, d_input, key=dec_key)

    def __call__(
        self,
        x     : jax.Array,   # (L, d_input)
        state : eqx.nn.State,
        *,
        key   : jax.Array,
    ):
        """Forward pass for a single (un-batched) sequence."""
        x = jax.vmap(self.encoder)(x)                           # (L, d_model)
        keys = jr.split(key, len(self.blocks))
        for block, k in zip(self.blocks, keys):
            x, state = block(x, state, key=k)
        x = jax.vmap(self.decoder)(x)                           # (L, d_input)
        return x, state


class LinOSSRegressionJAX(eqx.Module):
    """Sequence-to-scalar regression model built from ``LinOSSBlock`` layers.

    Maps ``(L, d_input) → (d_output,)`` — used as the regression stage of
    ``LinOSSCombinedJAX``.

    Construct via ``eqx.nn.make_with_state``::

        model, state = eqx.nn.make_with_state(LinOSSRegressionJAX)(...)
    """

    encoder  : eqx.nn.Linear
    blocks   : list            # List[LinOSSBlock]
    fc_layers: list            # List of (Linear, activation) pairs + final Linear

    def __init__(
        self,
        d_input        : int,
        d_model        : int,
        d_output       : int,
        n_layers       : int,
        ssm_size       : int         = 64,
        discretization : str         = 'IM',
        dropout        : float       = 0.0,
        fc_hidden      : list        = None,
        *,
        key            : jax.Array,
    ):
        if fc_hidden is None:
            fc_hidden = [64, 32]
        # Split into: 1 enc key + n_layers block keys + 1 fc seed key
        all_keys = jr.split(key, n_layers + 2)
        enc_key  = all_keys[0]
        blk_keys = all_keys[1 : n_layers + 1]
        fc_seed  = all_keys[n_layers + 1]

        self.encoder = eqx.nn.Linear(d_input, d_model, key=enc_key)
        self.blocks  = [
            LinOSSBlock(ssm_size, d_model, discretization, drop_rate=dropout, key=k)
            for k in blk_keys
        ]

        # MLP: d_model → fc_hidden[0] → ... → d_output
        dims    = [d_model] + list(fc_hidden) + [d_output]
        fc_keys = jr.split(fc_seed, len(dims) - 1)
        self.fc_layers = [
            eqx.nn.Linear(dims[i], dims[i + 1], key=fc_keys[i])
            for i in range(len(dims) - 1)
        ]

    def __call__(
        self,
        x     : jax.Array,   # (L, d_input)
        state : eqx.nn.State,
        *,
        key   : jax.Array,
    ):
        """Forward pass for a single (un-batched) sequence."""
        x = jax.vmap(self.encoder)(x)                           # (L, d_model)
        keys = jr.split(key, len(self.blocks))
        for block, k in zip(self.blocks, keys):
            x, state = block(x, state, key=k)
        x = jnp.mean(x, axis=0)                                 # (d_model,) — mean pool
        for fc in self.fc_layers[:-1]:
            x = jax.nn.gelu(fc(x))
        x = self.fc_layers[-1](x)                               # (d_output,)
        return x, state


class LinOSSCombinedJAX(eqx.Module):
    """Combined denoising + regression model — fully JAX / equinox.

    Mirrors ``src.models.networks.LinOSSCombinedModel`` for the pure-JAX
    training pipeline.

    Architecture
    ------------
    1. **Denoiser** (``LinOSSSeq2SeqJAX``) — maps noisy I/Q ``(L, d_input)``
       to a denoised sequence of the same shape.
    2. **Regressor** (``LinOSSRegressionJAX``) — reads the denoised sequence
       and outputs a scalar prediction vector ``(d_output,)``.

    Usage (single sample, un-batched)::

        model, state = eqx.nn.make_with_state(LinOSSCombinedJAX)(
            d_input=2, d_output=2, ...)
        x_denoised, y_pred, state = model(x_noisy, state, key=key)

    For batched training use ``jax.vmap`` with ``axis_name="batch"`` so that
    the ``eqx.nn.BatchNorm`` layers inside each block see the full batch::

        vmapped = jax.vmap(
            model, axis_name="batch",
            in_axes=(0, None, None), out_axes=(0, 0, None)
        )
        x_denoised, y_pred, state = vmapped(X_noisy, state, key)
    """

    denoiser  : LinOSSSeq2SeqJAX
    regressor : LinOSSRegressionJAX

    def __init__(
        self,
        d_input                  : int,
        d_output                 : int,
        # Denoiser
        denoiser_d_model         : int   = 128,
        denoiser_n_layers        : int   = 6,
        denoiser_ssm_size        : int   = 64,
        denoiser_discretization  : str   = 'IM',
        denoiser_dropout         : float = 0.0,
        # Regressor
        regressor_d_model        : int   = 128,
        regressor_n_layers       : int   = 6,
        regressor_ssm_size       : int   = 64,
        regressor_discretization : str   = 'IM',
        regressor_dropout        : float = 0.0,
        regressor_fc_hidden      : list  = None,
        *,
        key                      : jax.Array,
    ):
        key_d, key_r = jr.split(key)
        self.denoiser = LinOSSSeq2SeqJAX(
            d_input        = d_input,
            d_model        = denoiser_d_model,
            n_layers       = denoiser_n_layers,
            ssm_size       = denoiser_ssm_size,
            discretization = denoiser_discretization,
            dropout        = denoiser_dropout,
            key            = key_d,
        )
        self.regressor = LinOSSRegressionJAX(
            d_input        = d_input,
            d_model        = regressor_d_model,
            d_output       = d_output,
            n_layers       = regressor_n_layers,
            ssm_size       = regressor_ssm_size,
            discretization = regressor_discretization,
            dropout        = regressor_dropout,
            fc_hidden      = regressor_fc_hidden,
            key            = key_r,
        )

    def __call__(
        self,
        x_noisy : jax.Array,    # (L, d_input)
        state   : eqx.nn.State,
        *,
        key     : jax.Array,
    ):
        """Forward pass for a single (un-batched) sequence.

        Returns
        -------
        x_denoised : (L, d_input)
        y_pred     : (d_output,)
        state      : updated BatchNorm running stats
        """
        key_d, key_r = jr.split(key)
        x_denoised, state = self.denoiser(x_noisy, state, key=key_d)
        y_pred,     state = self.regressor(x_denoised, state, key=key_r)
        return x_denoised, y_pred, state
