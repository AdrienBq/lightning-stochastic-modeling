"""Transformer-based conditional velocity field and the flow-matching path/sampler for the diffusion branch.

The diffusion branch models lightning maps (or the residual of an upstream model's prediction) with conditional
flow matching: a velocity field ``v_theta(x_t, t, cond)`` is trained so that integrating the probability-flow ODE
from a standard-normal latent transports it to the data distribution conditioned on the ERA5 predictors.

⚠️ GENERATION HAPPENS IN THE RAW TARGET SPACE — lightning-hours in ``0-24``, or the signed residual against the
upstream prediction. There is no warp and no standardization of the generation target, so nothing here is inverted
downstream: training space == evaluation space, as everywhere else in this project. (The source branch generated in
a log1p-standardized space and inverted it; that was the removed target transform under another name. A plain
affine standardization was considered and rejected as buying nothing the stem's first convolution cannot absorb —
it does not touch the real difficulty, which is that ~99.93 % of cells are exactly zero.)

Backbone (the "hybrid conv + transformer" choice): a small convolutional stem fuses the many conditioning channels
(up to ~120 in daily hourly-stack mode) together with the single noisy-target channel and patch-embeds them onto a
coarse token grid; a stack of DiT-style transformer blocks with adaptive-LayerNorm-Zero (adaLN-Zero) conditioning
on the diffusion time (and a pooled conditioning summary) processes the tokens; a convolutional decoder upsamples
back to the input resolution and a 1x1 head emits the velocity map. Inputs are padded internally to a multiple of
the patch size and the output is cropped back, so any grid size is accepted (mirroring the U-net backbone).

The flow-matching path is the straight (rectified-flow / linear OT) interpolant ``x_t = (1 - t) z + t x_1`` whose
target velocity is the constant ``x_1 - z``; :func:`flow_matching_targets` builds a training tuple and
:func:`sample` integrates the learned field with an explicit Euler scheme.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.modeling.unet import make_activation, make_normalization


# ----------------------------------------------------------------------------------------------------------------
# positional / time embeddings
# ----------------------------------------------------------------------------------------------------------------
def sinusoidal_embedding(values: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Classic sinusoidal embedding of a 1-D tensor of scalars into ``dim`` features (``dim`` even)."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=values.device) / max(half, 1)
    )
    angles = values.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2:                                                     # odd dim: pad the last column
        embedding = F.pad(embedding, (0, 1))
    return embedding


def build_2d_sincos_pos_embed(height: int, width: int, dim: int, device, dtype) -> torch.Tensor:
    """Fixed 2-D sin-cos positional embedding ``[height * width, dim]`` (``dim`` divisible by 4), row-major.

    Computed on the fly so the network is agnostic to the (padded) token-grid size, like the U-net backbone."""
    if dim % 4:
        raise ValueError(f'2-D sin-cos positional embedding needs dim divisible by 4, got {dim}.')
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    row_embed = sinusoidal_embedding(rows, dim // 2)                # [H, dim/2]
    col_embed = sinusoidal_embedding(cols, dim // 2)                # [W, dim/2]
    grid = torch.cat([
        row_embed[:, None, :].expand(height, width, dim // 2),
        col_embed[None, :, :].expand(height, width, dim // 2)
    ], dim=-1)                                                      # [H, W, dim]
    return grid.reshape(height * width, dim).to(dtype)


class TimestepEmbedder(nn.Module):
    """Embed the scalar diffusion time ``t in [0, 1]`` into a ``hidden``-wide conditioning vector."""

    def __init__(self, hidden: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(t, self.frequency_dim))


# ----------------------------------------------------------------------------------------------------------------
# DiT-style transformer block with adaLN-Zero conditioning
# ----------------------------------------------------------------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Affine FiLM modulation of a LayerNorm output (broadcast over the token axis)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Transformer block with multi-head self-attention and an MLP, each wrapped in adaptive LayerNorm with a
    zero-initialised residual gate (adaLN-Zero): the six modulation parameters (shift/scale/gate for the
    attention and the MLP branches) are produced from the conditioning vector by a linear layer whose weights and
    bias start at zero, so every block is an identity map at initialisation and learns its contribution gradually
    (the DiT initialisation that stabilises training)."""

    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, hidden)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = \
            self.modulation(conditioning).chunk(6, dim=-1)
        normed = modulate(self.norm1(x), shift_attention, scale_attention)
        attended, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + gate_attention.unsqueeze(1) * attended
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ----------------------------------------------------------------------------------------------------------------
# convolutional stem / decoder
# ----------------------------------------------------------------------------------------------------------------
class ConvStem(nn.Module):
    """Fuse the (noisy target + conditioning) channels at full resolution, then patch-embed onto the token grid
    with a strided convolution of stride/kernel ``patch_size``."""

    def __init__(self, in_channels: int, stem_channels: int, hidden: int, patch_size: int,
                 normalization: str, activation: str):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1),
            make_normalization(normalization, stem_channels),
            make_activation(activation)
        )
        self.patch_embed = nn.Conv2d(stem_channels, hidden, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(self.fuse(x))                       # [B, hidden, H', W']


class ConvDecoder(nn.Module):
    """Upsample the token-grid feature map back to the input resolution (``log2(patch_size)`` transposed-conv x2
    stages) and emit the single-channel velocity map."""

    def __init__(self, hidden: int, patch_size: int, normalization: str, activation: str):
        super().__init__()
        num_upsamples = int(round(math.log2(patch_size)))
        if 2 ** num_upsamples != patch_size:
            raise ValueError(f'patch_size must be a power of two, got {patch_size}.')
        layers = []
        channels = hidden
        for _ in range(num_upsamples):
            out_channels = max(channels // 2, 16)
            layers += [
                nn.ConvTranspose2d(channels, out_channels, kernel_size=2, stride=2),
                make_normalization(normalization, out_channels),
                make_activation(activation)
            ]
            channels = out_channels
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.head.weight)                            # velocity starts at 0 (with adaLN-Zero blocks)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


# ----------------------------------------------------------------------------------------------------------------
# velocity field
# ----------------------------------------------------------------------------------------------------------------
class FlowVelocityNet(nn.Module):
    """Conditional velocity field ``v_theta(x_t, t, cond)`` for flow matching (the hybrid conv + transformer
    backbone described in the module docstring).

    Args:
        cond_channels: Number of conditioning feature channels (standardized ERA5 predictors plus, in residual
            mode, the upstream-prediction channel). The single noisy-target channel is concatenated internally.
        config: Architecture keys, translated from the trial's ``flow`` section by the LightningModule (the
            config calls the DiT-block count ``n_blocks``, because ``depth`` already means the down/upsampling
            level count in the ``unet`` block). Keys: ``hidden_dim``, ``depth``, ``num_heads``,
            ``mlp_ratio``, ``patch_size``, ``stem_channels``, ``dropout``, ``normalization``, ``activation``,
            ``time_frequency_dim``.
    """

    def __init__(self, cond_channels: int, config: dict):
        super().__init__()
        hidden = int(config['hidden_dim'])
        depth = int(config['depth'])
        num_heads = int(config['num_heads'])
        mlp_ratio = float(config.get('mlp_ratio', 4.0))
        patch_size = int(config['patch_size'])
        stem_channels = int(config.get('stem_channels', 64))
        dropout = float(config.get('dropout', 0.0))
        normalization = config.get('normalization', 'group')
        activation = config.get('activation', 'silu')
        if hidden % num_heads:
            raise ValueError(f'hidden_dim ({hidden}) must be divisible by num_heads ({num_heads}).')
        if hidden % 4:
            raise ValueError(f'hidden_dim ({hidden}) must be divisible by 4 for the 2-D positional embedding.')

        self.cond_channels = cond_channels
        self.hidden = hidden
        self.patch_size = patch_size

        self.stem = ConvStem(cond_channels + 1, stem_channels, hidden, patch_size, normalization, activation)
        self.time_embedder = TimestepEmbedder(hidden, int(config.get('time_frequency_dim', 256)))
        # pooled conditioning summary -> conditioning vector (gives adaLN a global view of the predictors)
        self.cond_pool = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList(
            DiTBlock(hidden, num_heads, mlp_ratio, dropout) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        nn.init.zeros_(self.final_modulation[1].weight)
        nn.init.zeros_(self.final_modulation[1].bias)
        self.decoder = ConvDecoder(hidden, patch_size, normalization, activation)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Predict the velocity at ``(x_t, t)`` given the conditioning maps.

        Args:
            x_t: Noisy target map ``[B, H, W]`` or ``[B, 1, H, W]`` on the flow path (raw target space).
            t: Diffusion time ``[B]`` in ``[0, 1]``.
            cond: Conditioning maps ``[B, cond_channels, H, W]`` (already standardized).
        """
        if x_t.dim() == 3:
            x_t = x_t.unsqueeze(1)
        height, width = x_t.shape[-2:]
        pad_h = (-height) % self.patch_size
        pad_w = (-width) % self.patch_size
        stacked = torch.cat([x_t, cond], dim=1)
        if pad_h or pad_w:
            stacked = F.pad(stacked, (0, pad_w, 0, pad_h), mode='replicate')

        tokens_map = self.stem(stacked)                             # [B, hidden, H', W']
        _, _, grid_h, grid_w = tokens_map.shape
        tokens = tokens_map.flatten(2).transpose(1, 2)              # [B, N, hidden]
        tokens = tokens + build_2d_sincos_pos_embed(
            grid_h, grid_w, self.hidden, tokens.device, tokens.dtype
        ).unsqueeze(0)

        conditioning = self.time_embedder(t) + self.cond_pool(tokens.mean(dim=1))
        for block in self.blocks:
            tokens = block(tokens, conditioning)

        shift, scale = self.final_modulation(conditioning).chunk(2, dim=-1)
        tokens = modulate(self.final_norm(tokens), shift, scale)
        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], self.hidden, grid_h, grid_w)
        velocity = self.decoder(feature_map)                        # [B, 1, Hpad, Wpad]
        return velocity.squeeze(1)[..., :height, :width]            # [B, H, W]


# ----------------------------------------------------------------------------------------------------------------
# rectified-flow path and sampler
# ----------------------------------------------------------------------------------------------------------------
def flow_matching_targets(
        x1: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the straight (rectified-flow / linear OT) training tuple for a data sample ``x1``.

    With ``z ~ N(0, I)`` and the interpolant ``x_t = (1 - t) z + t x1``, the path's velocity is the constant
    ``x1 - z``. Returns ``(x_t, target_velocity)``; ``t`` is broadcast over the spatial dimensions.

    Args:
        x1: Data sample in the RAW target space (hours, or the signed residual), ``[B, H, W]``.
        t: Per-sample time ``[B]`` in ``[0, 1]``.
        noise: Optional pre-sampled ``z`` (same shape as ``x1``); drawn from ``N(0, I)`` when None.
    """
    if noise is None:
        noise = torch.randn_like(x1)
    t_map = t.view(-1, *([1] * (x1.dim() - 1)))
    x_t = (1.0 - t_map) * noise + t_map * x1
    return x_t, x1 - noise


@torch.no_grad()
def sample(
        net: FlowVelocityNet,
        cond: torch.Tensor,
        spatial_shape: Tuple[int, int],
        num_steps: int,
        generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Integrate the probability-flow ODE ``dx/dt = v_theta(x, t, cond)`` from ``x(0) ~ N(0, I)`` to ``x(1)`` with
    an explicit Euler scheme of ``num_steps`` uniform steps, returning the generated sample in the RAW target
    space (``[B, H, W]``) -- unclamped, since the LightningModule owns the reconstruction and the 0-max_hours
    clamp."""
    batch = cond.shape[0]
    height, width = spatial_shape
    x = torch.randn(batch, height, width, device=cond.device, dtype=cond.dtype, generator=generator)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch,), step * dt, device=cond.device, dtype=cond.dtype)
        x = x + dt * net(x, t, cond)
    return x
