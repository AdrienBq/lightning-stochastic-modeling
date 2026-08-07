"""Configurable U-net backbone and the (optionally hierarchical) prediction network, shared by all three families.

The architecture hyperparameters mirror the ``unet`` section of each family's config/<family>/search_space.yaml:
depth, base channels, kernel size, blocks per level, normalization, activation, dropout, upsampling mode,
skip connections and an optional multi-head self-attention block at the bottleneck (cheap at this grid size,
~7x10 tokens at depth 4 on the 0.25-degree European domain).

``DeterministicUnetNet`` adds a 1x1 regression head and, when the hierarchy is enabled, an occurrence-classifier
head, either sharing the trunk (``shared_encoder``) or with its own backbone (``standalone_unet``). Inputs are
padded internally to a multiple of ``2 ** depth`` and outputs cropped back, so any grid size is accepted.

ONE backbone, three families. The deterministic U-net trains it directly; MC-dropout re-runs it with dropout left
active at inference (:func:`enable_mc_dropout`); diffusion conditions it on the flow state. Nothing here is
family-specific, which is what keeps the three comparable at equal architecture:

- ``ConvBlock`` already emits ``nn.Dropout2d`` whenever ``unet.dropout > 0``, so the MC-dropout family needs no
  architectural change at all -- only the eval-time mode flip.
- ``Fp32BilinearUpsample`` is PARAMETER-FREE, so a checkpoint is interchangeable with one trained against a plain
  ``nn.Upsample``. That is what lets MC-dropout warm-start from a deterministic U-net checkpoint.
"""
from typing import Optional, Tuple

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

    Silent no-op when ``unet.dropout == 0`` — :class:`ConvBlock` then contains no ``Dropout2d`` at all and every
    "member" is identical, which ``spread_skill_sums`` reports as zero spread rather than an error.
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
            if dropout > 0:
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
    """Classical Platt scaling of a classifier logit: ``z -> weight * z + bias``, two learnable scalars applied
    BEFORE the sigmoid, so the calibrated probability is ``sigmoid(weight * z + bias)``. Initialised to the
    identity so that enabling it leaves the pre-calibration logit unchanged until its dedicated training phase
    fits the scalars (plain BCE, with the rest of the backbone frozen).

    The map is monotonic in the logit, so it leaves the classifier's RANKING (and therefore a recall-calibrated
    hard occurrence mask) untouched; it recalibrates the probabilities themselves (Brier/reliability, and the
    soft-masked prediction).
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
    """U-net regressor with an optional occurrence-classifier head (the top level of the hierarchy) and optional
    post-hoc calibration layers: Platt scaling on the classifier logits and a monotone calibration on the
    regression output. Each is fitted in its own phase with the rest of the backbone frozen.

    Args:
        in_channels: Number of input channels.
        unet_config: The ``unet`` section of a sampled trial.
        classifier_architecture: ``None`` (hierarchy disabled), ``shared_encoder`` (classification head on the
            shared trunk) or ``standalone_unet`` (dedicated backbone for the classifier).
        classifier_calibration: Whether to append a :class:`PlattScaling` layer to the classifier logits.
            Ignored when there is no classifier head.
        regression_calibration: ``None`` / falsy to disable, else a dict ``{'structure': ..., 'num_sigmoids':
            ...}`` selecting a :class:`MonotoneCalibration` for the regression output (``structure == 'none'``
            also disables it). The layer is owned here (so it is checkpointed and discoverable) but APPLIED by
            the LightningModule after the regression activation, where the prediction is non-negative.
    """

    def __init__(
            self,
            in_channels: int,
            unet_config: dict,
            classifier_architecture: Optional[str] = None,
            classifier_calibration: bool = False,
            regression_calibration: Optional[dict] = None
    ):
        super().__init__()
        base = int(unet_config['base_channels'])
        self.pad_multiple = 2 ** int(unet_config['depth'])

        self.backbone = UNetBackbone(in_channels, unet_config)
        self.regression_head = nn.Conv2d(base, 1, kernel_size=1)

        self.classifier_architecture = classifier_architecture
        self.classifier_backbone = None
        self.classifier_head = None
        if classifier_architecture == 'shared_encoder':
            self.classifier_head = nn.Conv2d(base, 1, kernel_size=1)
        elif classifier_architecture == 'standalone_unet':
            self.classifier_backbone = UNetBackbone(in_channels, unet_config)
            self.classifier_head = nn.Conv2d(base, 1, kernel_size=1)
        elif classifier_architecture is not None:
            raise ValueError(
                f'Unknown classifier architecture "{classifier_architecture}" '
                f'(expected "shared_encoder" or "standalone_unet").'
            )

        # Platt scaling only applies to the classifier; it is a no-op without a classifier head
        self.classifier_calibration = PlattScaling() \
            if (classifier_calibration and self.classifier_head is not None) else None

        regression_calibration = regression_calibration or {}
        structure = regression_calibration.get('structure', 'none')
        self.regression_calibration = MonotoneCalibration(
            structure, int(regression_calibration.get('num_sigmoids', 4))
        ) if structure and structure != 'none' else None

    def classifier_parameters(self):
        """Parameters belonging exclusively to the classifier level (for sequential-phase freezing)."""
        modules = [m for m in (self.classifier_backbone, self.classifier_head) if m is not None]
        for module in modules:
            yield from module.parameters()

    def classifier_calibration_parameters(self):
        """Parameters of the classifier Platt layer (calibration-phase freezing); empty when disabled."""
        if self.classifier_calibration is not None:
            yield from self.classifier_calibration.parameters()

    def regression_calibration_parameters(self):
        """Parameters of the monotone regression calibrator (calibration-phase freezing); empty when disabled."""
        if self.regression_calibration is not None:
            yield from self.regression_calibration.parameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.pad_multiple
        pad_w = (-width) % self.pad_multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        features = self.backbone(x)
        regression = self.regression_head(features).squeeze(1)[..., :height, :width]

        classifier_logits = None
        if self.classifier_head is not None:
            cls_features = features if self.classifier_backbone is None else self.classifier_backbone(x)
            classifier_logits = self.classifier_head(cls_features).squeeze(1)[..., :height, :width]
            if self.classifier_calibration is not None:     # Platt scaling of the classifier logits
                classifier_logits = self.classifier_calibration(classifier_logits)

        # NOTE: the monotone regression calibration is applied by the LightningModule AFTER the regression
        # activation (where the prediction is non-negative), not here on the raw head output.
        return regression, classifier_logits
