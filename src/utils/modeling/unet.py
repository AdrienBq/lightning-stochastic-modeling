"""Configurable U-net backbone and the single-head prediction network, shared by all three families.

The architecture hyperparameters mirror the ``unet`` section of each family's config/<family>/search_space_<task>.yaml:
depth, base channels, kernel size, blocks per level, normalization, activation, dropout, upsampling mode,
skip connections and an optional multi-head self-attention block at the bottleneck (cheap at this grid size,
~7x10 tokens at depth 4 on the 0.25-degree European domain).

``DeterministicUnetNet`` adds ONE 1x1 output head, whose meaning the LightningModule fixes from the data's ``mode``
(hours in daily, an occurrence logit in hourly), plus the two optional calibration layers. Inputs are padded
internally to a multiple of ``2 ** depth`` and outputs cropped back, so any grid size is accepted.

ONE backbone, three families. The deterministic U-net trains it directly; MC-dropout re-runs it with dropout left
active at inference (:func:`enable_mc_dropout`); diffusion conditions it on the flow state. Nothing here is
family-specific, which is what keeps the three comparable at equal architecture:

- ``ConvBlock`` ALWAYS emits an ``nn.Dropout2d`` (with ``p = unet.dropout``, which may be 0), so the MC-dropout family
  needs no architectural change at all -- only the eval-time mode flip. Emitting it unconditionally is what makes the
  two families' ``state_dict`` KEYS identical, and therefore what makes the warm start possible; see the comment on
  :class:`ConvBlock` for the bug that taught us this.
- ``Fp32BilinearUpsample`` is PARAMETER-FREE, so a checkpoint is interchangeable with one trained against a plain
  ``nn.Upsample``. That is what lets MC-dropout warm-start from a deterministic U-net checkpoint.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_mc_dropout(module: nn.Module) -> None:
    """Put ONLY the ``Dropout2d`` submodules into train mode, leaving everything else in eval mode.

    MC-dropout inference needs the network in ``eval()`` so normalization statistics stay frozen, while the dropout
    units keep sampling fresh masks — that difference between members IS the ensemble. Call this AFTER
    ``model.eval()``; calling ``model.train()`` instead would also unfreeze the normalization layers and make the
    members differ for the wrong reason.

    This is why the MC-dropout family requires ``normalization: group``: GroupNorm normalizes within a sample and
    carries no running statistics, so a member's output does not depend on the rest of its batch. Under BatchNorm
    the members would additionally covary through the batch.

    Silent no-op when ``unet.dropout == 0``: the ``Dropout2d`` submodules are still THERE (:class:`ConvBlock` emits
    them unconditionally so the state-dict keys do not depend on the dropout value), but ``p = 0`` makes each an
    identity, so every "member" is identical — which ``spread_skill_sums`` reports as zero spread rather than an error.
    """
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout2d):
            submodule.train()


def make_normalization(kind: str, channels: int) -> nn.Module:
    if kind == 'batch':
        return nn.BatchNorm2d(channels)
    if kind == 'group':
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f'Unknown normalization "{kind}" (expected "batch" or "group").')


def make_activation(name: str) -> nn.Module:
    activations = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}
    if name not in activations:
        raise ValueError(f'Unknown activation "{name}" (expected one of {list(activations)}).')
    return activations[name]()


class ConvBlock(nn.Module):
    """``blocks`` convolution -> normalization -> activation (-> dropout) stages at constant output width."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            normalization: str,
            activation: str,
            dropout: float,
            blocks: int
    ):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(blocks):
            layers.append(nn.Conv2d(channels, out_channels, kernel_size, padding=kernel_size // 2))
            layers.append(make_normalization(normalization, out_channels))
            layers.append(make_activation(activation))
            # 🐛 ALWAYS appended, never conditionally on `dropout > 0`. `Dropout2d(0.0)` is an identity, so this costs
            # nothing — but the layer's PRESENCE is what keeps the `nn.Sequential` numbering identical whether dropout
            # is on or off, and those indices ARE the state_dict keys.
            #
            # Inserting it conditionally broke the MC-dropout warm start outright: at `blocks_per_level: 2` (the value
            # every shipped search space fixes) a deterministic checkpoint carries `body.3`/`body.4` while a
            # dropout-bearing net expects `body.4`/`body.5`, so `from_upstream`'s strict `load_state_dict` failed on
            # every key with "size mismatch for body.4.weight: [64] vs [64, 64, 3, 3]" — a norm where a conv was
            # expected. The deterministic family FIXES `dropout: 0.0` and MC-dropout REQUIRES dropout > 0, so the warm
            # start could never have worked. Found by the Step 4 block 4e real-data gate; the unit tests missed it
            # because `tests/conftest.py`'s UNET fixture uses `blocks_per_level: 1`, where the dropout lands after the
            # last layer and nothing shifts. `diffusion.py`'s MLP block always inserted its dropout, so this now
            # matches it.
            layers.append(nn.Dropout2d(dropout))
            channels = out_channels
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class BottleneckAttention(nn.Module):
    """Multi-head self-attention over the (small) bottleneck spatial grid, with a residual connection."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)                   # [B, H*W, C]
        tokens_norm = self.norm(tokens)
        attended, _ = self.attention(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        tokens = tokens + attended
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class Fp32BilinearUpsample(nn.Module):
    """2x bilinear upsampling forced through float32: reduced-precision inputs (bf16 under autocast) lack
    bilinear-interpolation CUDA kernels on older torch builds, and the op is memory-bound anyway. Parameter-free,
    so checkpoints are interchangeable with a plain ``nn.Upsample``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.bfloat16, torch.float16):
            return F.interpolate(x.float(), scale_factor=2, mode='bilinear', align_corners=False).to(x.dtype)
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)


class UpBlock(nn.Module):
    """Upsampling (transposed convolution or bilinear + 1x1 convolution) followed by a ConvBlock, with an
    optional skip-connection concatenation."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            normalization: str,
            activation: str,
            dropout: float,
            blocks: int,
            upsampling: str,
            skip_connections: bool
    ):
        super().__init__()
        if upsampling == 'transposed_conv':
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsampling == 'bilinear_conv':
            self.upsample = nn.Sequential(
                Fp32BilinearUpsample(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            raise ValueError(f'Unknown upsampling "{upsampling}" (expected "transposed_conv" or "bilinear_conv").')

        merged_channels = out_channels * 2 if skip_connections else out_channels
        self.skip_connections = skip_connections
        self.block = ConvBlock(merged_channels, out_channels, kernel_size, normalization, activation, dropout, blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.skip_connections:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class UNetBackbone(nn.Module):
    """Encoder-decoder trunk returning a ``base_channels``-wide feature map at the input resolution.

    Args:
        in_channels: Number of input channels.
        unet_config: The ``unet`` section of a sampled trial (see module docstring for the expected keys).
    """

    def __init__(self, in_channels: int, unet_config: dict):
        super().__init__()
        depth = int(unet_config['depth'])
        base = int(unet_config['base_channels'])
        multiplier = int(unet_config.get('channel_multiplier', 2))
        common = dict(
            kernel_size=int(unet_config['kernel_size']),
            normalization=unet_config['normalization'],
            activation=unet_config['activation'],
            dropout=float(unet_config['dropout']),
            blocks=int(unet_config['blocks_per_level'])
        )
        skip_connections = bool(unet_config.get('skip_connections', True))

        widths = [base * multiplier ** level for level in range(depth + 1)]
        self.depth = depth

        self.stem = ConvBlock(in_channels, widths[0], **common)
        self.encoders = nn.ModuleList(
            ConvBlock(widths[level], widths[level + 1], **common) for level in range(depth)
        )
        self.pool = nn.MaxPool2d(2)
        self.attention = BottleneckAttention(widths[-1]) if unet_config.get('bottleneck_attention', False) else None
        self.decoders = nn.ModuleList(
            UpBlock(
                widths[level + 1], widths[level],
                upsampling=unet_config['upsampling'], skip_connections=skip_connections, **common
            )
            for level in reversed(range(depth))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(self.pool(x))
        if self.attention is not None:
            x = self.attention(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return x


class PlattScaling(nn.Module):
    """Classical Platt scaling of an occurrence logit: ``z -> weight * z + bias``, two learnable scalars applied
    BEFORE the sigmoid, so the calibrated probability is ``sigmoid(weight * z + bias)``. Initialised to the
    identity so that enabling it leaves the pre-calibration logit unchanged until its dedicated training phase
    fits the scalars (plain BCE, with the rest of the backbone frozen).

    HOURLY MODE ONLY: it calibrates the single head's logit, which is where the classification task lives. In daily
    mode the head emits hours and there is no logit to scale — see :class:`MonotoneCalibration` for that task's
    calibrator.

    The map is monotonic in the logit, so it leaves the RANKING untouched (hence ``roc_auc`` and
    ``average_precision`` are invariant to it); what it moves is the probabilities themselves — Brier, reliability
    and explained deviance.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


# softplus(SOFTPLUS_ONE) == 1.0: lets a softplus-constrained positive parameter initialise to exactly 1
SOFTPLUS_ONE = 0.5413248538970947
REGRESSION_CALIBRATION_STRUCTURES = ('power_law', 'monotone_smooth')


class MonotoneCalibration(nn.Module):
    """Monotone, zero-preserving recalibration of a NON-NEGATIVE predicted target, fitted as a separate phase
    with the backbone frozen. The warp lives in ``log1p`` space, so it is scale-aware across the target's wide
    dynamic range and preserves ``g(0) = 0``; it initialises to the identity.

    Structures (selectable as a hyperparameter):
      * ``power_law``  : ``g(p) = s * ((1 + p) ** a - 1)`` with ``s, a > 0`` (2 parameters);
      * ``monotone_smooth`` : ``g(p) = expm1(m(log1p(p)))`` with ``m`` a monotone-increasing map anchored at
        ``m(0) = 0`` — a positive base slope plus ``num_sigmoids`` non-negative sigmoid bumps — for non-affine
        corrections across the range.

    Monotonicity (hence rank preservation of the prediction) is guaranteed by construction: all coefficients are
    constrained positive through softplus and the smooth map's bumps are non-decreasing.
    """

    def __init__(self, structure: str, num_sigmoids: int = 4):
        super().__init__()
        if structure not in REGRESSION_CALIBRATION_STRUCTURES:
            raise ValueError(
                f'Unknown regression calibration structure "{structure}" '
                f'(expected one of {REGRESSION_CALIBRATION_STRUCTURES}).'
            )
        self.structure = structure
        if structure == 'power_law':
            self.raw_scale = nn.Parameter(torch.tensor(SOFTPLUS_ONE))       # s = softplus(.) -> 1 at init
            self.raw_exponent = nn.Parameter(torch.tensor(SOFTPLUS_ONE))    # a = softplus(.) -> 1 at init
        else:
            self.raw_slope = nn.Parameter(torch.tensor(SOFTPLUS_ONE))       # base slope -> 1 at init (identity)
            self.raw_weights = nn.Parameter(torch.full((num_sigmoids,), -7.0))   # softplus(.) ~ 0 -> no bumps
            self.raw_sharpness = nn.Parameter(torch.full((num_sigmoids,), SOFTPLUS_ONE))
            # bump centres spread across log1p space (covers counts up to ~e^4.5); learnable
            self.centers = nn.Parameter(torch.linspace(0.5, 4.5, num_sigmoids))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        p = p.clamp(min=0.0)
        q = torch.log1p(p)
        if self.structure == 'power_law':
            scale = F.softplus(self.raw_scale)
            exponent = F.softplus(self.raw_exponent)
            return scale * torch.expm1(exponent * q)
        slope = F.softplus(self.raw_slope)
        weights = F.softplus(self.raw_weights)                              # [K] >= 0
        sharpness = F.softplus(self.raw_sharpness)                          # [K] >= 0
        # bump_k(q) = sigmoid(s_k (q - c_k)) - sigmoid(s_k (0 - c_k))  -> 0 at q = 0, so m(0) = 0
        z = sharpness * (q.unsqueeze(-1) - self.centers)
        z0 = sharpness * (-self.centers)
        bumps = (torch.sigmoid(z) - torch.sigmoid(z0)) * weights
        m = slope * q + bumps.sum(dim=-1)
        return torch.expm1(m)


class DeterministicUnetNet(nn.Module):
    """U-net with a SINGLE output head, plus the two optional post-hoc calibration layers. Each calibrator is
    fitted in its own final phase with the rest of the backbone frozen, and exactly one of them can exist in a
    given run because which is meaningful follows from the task (see the two classes above).

    ONE HEAD, BY DESIGN. There was formerly a second, occurrence-classifier head that gated the regression output.
    It is gone: it answers an unbounded count target, whereas this one is bounded 0-24 where the regression covers
    the zeros directly, and in hourly mode THIS head already emits the occurrence logit — a second identical
    ``Conv2d(base, 1, 1)`` would only produce a redundant probability. What the head's output MEANS is decided by
    the LightningModule from the data's ``mode``, not here.

    Args:
        in_channels: Number of input channels.
        unet_config: The ``unet`` section of a sampled trial.
        output_calibration: Whether to append a :class:`PlattScaling` layer to the head's logit. HOURLY mode only —
            in daily mode the head emits hours, and there is no logit to scale.
        regression_calibration: ``None`` / falsy to disable, else a dict ``{'structure': ..., 'num_sigmoids':
            ...}`` selecting a :class:`MonotoneCalibration` for the predicted hours (``structure == 'none'`` also
            disables it). DAILY mode only. The layer is owned here (so it is checkpointed and discoverable) but
            APPLIED by the LightningModule after the output activation, where the prediction is non-negative.

    Both calibrators live HERE rather than on the module so they travel inside ``net.state_dict()`` — which is what
    a warm start loads (``MCDropoutModule.from_upstream``). A module-level layer would be silently dropped by it.
    """

    def __init__(
            self,
            in_channels: int,
            unet_config: dict,
            output_calibration: bool = False,
            regression_calibration: Optional[dict] = None
    ):
        super().__init__()
        base = int(unet_config['base_channels'])
        self.pad_multiple = 2 ** int(unet_config['depth'])

        self.backbone = UNetBackbone(in_channels, unet_config)
        self.head = nn.Conv2d(base, 1, kernel_size=1)

        self.output_calibration = PlattScaling() if output_calibration else None

        regression_calibration = regression_calibration or {}
        structure = regression_calibration.get('structure', 'none')
        self.regression_calibration = MonotoneCalibration(
            structure, int(regression_calibration.get('num_sigmoids', 4))
        ) if structure and structure != 'none' else None

    def output_calibration_parameters(self):
        """Parameters of the Platt layer (calibration-phase freezing); empty when disabled."""
        if self.output_calibration is not None:
            yield from self.output_calibration.parameters()

    def regression_calibration_parameters(self):
        """Parameters of the monotone regression calibrator (calibration-phase freezing); empty when disabled."""
        if self.regression_calibration is not None:
            yield from self.regression_calibration.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw head output, ``[B, H, W]`` — hours-to-be-activated in daily mode, an occurrence logit in hourly.

        Platt scaling is applied here because it is affine in the logit and belongs before the sigmoid; it is
        ``None`` outside hourly mode, so no branch on the task is needed.
        """
        height, width = x.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        output = self.head(self.backbone(x)).squeeze(1)[..., :height, :width]
        if self.output_calibration is not None:
            output = self.output_calibration(output)
        # NOTE: the monotone regression calibration is applied by the LightningModule AFTER the output activation,
        # where the prediction is non-negative — not here on the raw head output.
        return output
