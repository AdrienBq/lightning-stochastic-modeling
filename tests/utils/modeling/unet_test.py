"""Tests for src/utils/modeling/unet.py — the shared backbone and the two calibration layers.

Neither source branch tested this file. Its union was the delicate part of the merge: ``ConvBlock``,
``BottleneckAttention`` and ``UNetBackbone`` were byte-identical on both branches, D added only ``enable_mc_dropout``,
and A added ``Fp32BilinearUpsample``, ``PlattScaling``, ``MonotoneCalibration`` and the calibration wiring.

Block 3d-0 then dropped the second head and REWIRED the Platt layer onto the main one, which closed a latent
warm-start hole: a module-level Platt is not in ``net.state_dict()``, so ``from_upstream``'s ``load_state_dict`` would
have silently discarded fitted Platt weights. The state-dict test below is what makes that hole impossible rather than
merely absent.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils.modeling.unet import (
    REGRESSION_CALIBRATION_STRUCTURES, SOFTPLUS_ONE, BottleneckAttention, ConvBlock, DeterministicUnetNet,
    Fp32BilinearUpsample, MonotoneCalibration, PlattScaling, UNetBackbone, enable_mc_dropout, make_activation,
    make_normalization,
)

IN_CHANNELS = 5
HEIGHT, WIDTH = 24, 32
UNET = {'base_channels': 8, 'depth': 2, 'kernel_size': 3, 'blocks_per_level': 1, 'upsampling': 'bilinear_conv',
        'dropout': 0.0, 'normalization': 'group', 'activation': 'relu', 'bottleneck_attention': False}

PUBLIC_SURFACE = ('enable_mc_dropout', 'Fp32BilinearUpsample', 'PlattScaling', 'MonotoneCalibration',
                  'REGRESSION_CALIBRATION_STRUCTURES', 'UNetBackbone', 'ConvBlock', 'BottleneckAttention',
                  'DeterministicUnetNet')


@pytest.mark.parametrize('name', PUBLIC_SURFACE)
def test_the_merged_surface_is_all_present(name):
    """The union of both branches, which is what made this file the delicate part of the merge: D contributed only
    ``enable_mc_dropout`` and A contributed the two calibration layers and the fp32 upsample."""
    from src.utils.modeling import unet as unet_module

    assert hasattr(unet_module, name)


@pytest.mark.source_invariant
def test_the_typing_imports_match_what_the_signatures_still_NEED():
    """Block 3d-0 made ``forward`` return a single tensor instead of ``Tuple[Tensor, Optional[Tensor]]``, which orphaned
    ``Tuple``. ``Optional`` stays, because ``regression_calibration`` is still optional. A leftover import is harmless in
    itself; it is the marker that the second head's machinery was removed by deletion rather than by rewiring."""
    import ast

    from src.utils.modeling import unet as unet_module

    tree = ast.parse(open(unet_module.__file__).read())
    typing_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
                    and node.module == 'typing' for alias in node.names}
    assert 'Tuple' not in typing_names, 'forward returns a single tensor now'
    assert 'Optional' in typing_names, 'regression_calibration is still optional'


def test_the_old_net_name_is_GONE():
    """``DistrRegressionNet`` was named for a distributional head this project does not have — one output head, chosen by
    mode. The rename is what makes the checkpoint family marker meaningful."""
    from src.utils.modeling import unet as unet_module

    assert not hasattr(unet_module, 'DistrRegressionNet')


@pytest.fixture
def net():
    def build(output_calibration=False, regression_calibration=None, **overrides):
        return DeterministicUnetNet(IN_CHANNELS, {**UNET, **overrides},
                                    output_calibration=output_calibration,
                                    regression_calibration=regression_calibration).eval()
    return build


# =====================================================================================================================
# One head, and forward returns a single tensor
# =====================================================================================================================
def test_forward_returns_a_single_tensor(net):
    """Block 3d-0 dropped the ``Tuple[Tensor, Optional[Tensor]]`` return with the second head. In hourly mode the MAIN
    head IS the classifier — both were ``Conv2d(base, 1, 1)``, architecturally identical — so a second one would emit a
    redundant probability with no role."""
    output = net()(torch.randn(2, IN_CHANNELS, HEIGHT, WIDTH))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, HEIGHT, WIDTH) or output.shape == (2, 1, HEIGHT, WIDTH)


def test_the_second_head_and_its_machinery_are_gone(net):
    model = net()
    for attribute in ('classifier_head', 'classifier_backbone', 'classifier_calibration'):
        assert not hasattr(model, attribute), attribute
    assert not hasattr(model, 'classifier_parameters')


def test_the_head_is_named_head_not_regression_head(net):
    """Renamed in 3d-0 because in hourly mode ``regression_head`` lies. Safe only because no checkpoint existed yet —
    it is a ``state_dict`` KEY change, and ``from_upstream`` reads those keys."""
    model = net()
    assert hasattr(model, 'head')
    assert not hasattr(model, 'regression_head')
    assert any(key.startswith('head.') for key in model.state_dict())


# =====================================================================================================================
# The warm-start hole 3d-0 closed
# =====================================================================================================================
def test_the_platt_layer_lives_IN_the_net_state_dict(net):
    """The hole: ``from_upstream`` calls ``net.load_state_dict(...)``, and a module-level Platt is not in
    ``net.state_dict()``. Warm-starting an hourly MC-dropout run from an hourly deterministic upstream would have
    silently discarded the fitted Platt weights — harmless today (daily has no Platt) but a trap set for later."""
    model = net(output_calibration=True)
    platt_keys = [key for key in model.state_dict() if 'calibration' in key]
    assert platt_keys, f'no calibration parameters in the net state_dict: {sorted(model.state_dict())}'


def test_the_monotone_calibrator_is_also_in_the_net_state_dict(net):
    """The symmetry that 3d-0 restored: both calibrators live in the net, neither on the module."""
    model = net(regression_calibration={'structure': 'monotone_smooth', 'num_sigmoids': 4})
    assert [key for key in model.state_dict() if 'calibration' in key]


def test_output_calibration_is_absent_when_not_requested(net):
    model = net(output_calibration=False)
    assert getattr(model, 'output_calibration', None) is None


def test_the_calibration_parameters_accessor_is_renamed(net):
    model = net(output_calibration=True)
    assert hasattr(model, 'output_calibration_parameters')
    assert not hasattr(model, 'classifier_calibration_parameters')
    assert list(model.output_calibration_parameters())


# =====================================================================================================================
# enable_mc_dropout flips ONLY the dropout layers
# =====================================================================================================================
def test_enable_mc_dropout_flips_dropout_and_nothing_else(net):
    """The whole MC-dropout inference trick. Flipping the module wholesale to ``train()`` would also unfreeze the group
    norm, so the members would differ for the wrong reason and the spread would be meaningless."""
    model = net(dropout=0.2)
    model.eval()
    enable_mc_dropout(model)

    for submodule in model.modules():
        if isinstance(submodule, (nn.Dropout, nn.Dropout2d)):
            assert submodule.training, 'dropout must be active'
        elif isinstance(submodule, (nn.GroupNorm, nn.BatchNorm2d, nn.Conv2d, nn.Linear)):
            assert not submodule.training, f'{type(submodule).__name__} must stay in eval mode'


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_the_mc_spread_does_not_SWAMP_the_input_sensitivity(seed):
    """The bound that separates a useful MC ensemble from noise: the spread across members of ONE input must be smaller
    than the difference between two different inputs' ensemble means. If dropout noise exceeded the signal, the spread
    term of every ensemble metric would measure the dropout rate rather than the model's uncertainty.

    Parametrized over seeds deliberately. The gate this replaces bounded the pairwise member difference by the mean
    magnitude of the prediction, which holds at only 3 of 5 seeds — the ReLU output is mostly zeros, so it deflates the
    magnitude without deflating the difference. That bound passed at the gate's own seed and was not a property.
    """
    torch.manual_seed(seed)
    backbone = UNetBackbone(in_channels=3, unet_config={**UNET, 'dropout': 0.2}).eval()
    enable_mc_dropout(backbone)
    first, second = torch.randn(2, 3, 12, 16), torch.randn(2, 3, 12, 16)
    with torch.no_grad():
        members = torch.stack([backbone(first) for _ in range(16)])
        other = torch.stack([backbone(second) for _ in range(16)])

    within = (members - members.mean(dim=0)).abs().mean()
    between = (members.mean(dim=0) - other.mean(dim=0)).abs().mean()
    assert within > 0, 'the members must not be identical'
    assert within < between, f'dropout noise ({within:.4f}) swamps input sensitivity ({between:.4f})'


def test_mc_dropout_makes_two_forward_passes_differ(net):
    model = net(dropout=0.3)
    model.eval()
    x = torch.randn(2, IN_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        before_first, before_second = model(x), model(x)
        enable_mc_dropout(model)
        after_first, after_second = model(x), model(x)

    assert torch.allclose(before_first, before_second), 'eval() must be deterministic'
    assert not torch.allclose(after_first, after_second), 'enable_mc_dropout must make passes stochastic'


def test_a_zero_dropout_net_stays_deterministic_even_after_enabling(net):
    """There are no dropout layers to flip at p=0 (see below), so `enable_mc_dropout` is a no-op and every "member"
    is the same forward pass. Silent, hence the module-level raise on `dropout_p <= 0`."""
    model = net(dropout=0.0)
    model.eval()
    enable_mc_dropout(model)
    x = torch.randn(1, IN_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


# =====================================================================================================================
# The backbone
# =====================================================================================================================
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('upsampling', ['bilinear_conv', 'transposed_conv'])
def test_the_backbone_preserves_the_grid(depth, upsampling):
    """A U-net that returns a different resolution than it was given would break every score silently, since they are
    all computed on the arrays."""
    backbone = UNetBackbone(IN_CHANNELS, {**UNET, 'depth': depth, 'upsampling': upsampling}).eval()
    with torch.no_grad():
        output = backbone(torch.randn(2, IN_CHANNELS, HEIGHT, WIDTH))
    assert output.shape[-2:] == (HEIGHT, WIDTH)


def test_the_bare_backbone_REQUIRES_a_divisible_grid():
    """The backbone alone cannot take an arbitrary grid: each level halves with a floor, so 101 -> 50 -> 25 upsamples
    back to 100 and the skip-connection concat fails outright. Pinned because it is what makes the padding in
    DeterministicUnetNet load-bearing rather than defensive."""
    backbone = UNetBackbone(IN_CHANNELS, UNET).eval()
    with pytest.raises(RuntimeError, match='must match'):
        with torch.no_grad():
            backbone(torch.randn(1, IN_CHANNELS, 101, 149))


@pytest.mark.parametrize('depth', [3, 4, 5])
def test_the_NET_handles_the_real_101x149_grid_at_every_sampled_depth(depth):
    """The grid is 101 x 149 — both ODD — and the search space samples depth 3-5, so `2 ** depth` never divides it.
    The net pads to a multiple of `2 ** depth` with replicate padding and crops the output back.

    This is the test the Step 3 gates never ran: they all built modules on a 24 x 32 fixture, which is divisible by 8
    and so passes at any depth for the wrong reason.
    """
    model = DeterministicUnetNet(IN_CHANNELS, {**UNET, 'depth': depth}).eval()
    assert model.pad_multiple == 2 ** depth
    with torch.no_grad():
        output = model(torch.randn(1, IN_CHANNELS, 101, 149))
    assert output.shape[-2:] == (101, 149)


def test_bottleneck_attention_is_optional_and_changes_the_result():
    plain = UNetBackbone(IN_CHANNELS, {**UNET, 'bottleneck_attention': False}).eval()
    attended = UNetBackbone(IN_CHANNELS, {**UNET, 'bottleneck_attention': True}).eval()
    assert sum(p.numel() for p in attended.parameters()) > sum(p.numel() for p in plain.parameters())


def test_a_missing_unet_key_raises_rather_than_defaulting():
    """Six key mismatches between the config and this constructor were found during Step 3, each a ``KeyError`` at
    trial 0. Raising is right — a silent default would train a different architecture than the trials table records."""
    with pytest.raises(KeyError):
        UNetBackbone(IN_CHANNELS, {key: value for key, value in UNET.items() if key != 'kernel_size'})


def test_a_dropout_layer_is_emitted_ONLY_when_the_rate_is_positive():
    """At p=0 there is NO Dropout2d in the block at all — the layer is omitted, not merely inert. So MC-dropout with
    `dropout_p = 0` would have nothing to flip and would produce M identical members, a zero spread and NaN
    spread-skill, with no exception anywhere.

    That is the whole reason `MCDropoutModule` rejects `dropout_p <= 0` rather than tolerating it, and this is the
    other half of that argument (see mc_dropout_module_test.py).
    """
    def dropout_layers(rate):
        block = ConvBlock(4, 8, kernel_size=3, normalization='group', activation='relu', dropout=rate, blocks=1)
        return [m for m in block.modules() if isinstance(m, (nn.Dropout, nn.Dropout2d))]

    assert dropout_layers(0.0) == [], 'a zero rate must emit no dropout layer'
    assert dropout_layers(0.2), 'a positive rate must emit a dropout layer for enable_mc_dropout to flip'


@pytest.mark.parametrize('blocks', [1, 2, 3])
def test_blocks_per_level_stacks_that_many_convolutions(blocks):
    block = ConvBlock(4, 8, kernel_size=3, normalization='group', activation='relu', dropout=0.0, blocks=blocks)
    assert sum(1 for m in block.modules() if isinstance(m, nn.Conv2d)) == blocks


def test_the_fp32_upsample_keeps_the_dtype_and_doubles_the_grid():
    """Parameter-free, and therefore checkpoint-interchangeable with `nn.Upsample` — which is why warm-starting an
    MC-dropout net from a deterministic checkpoint is unaffected by A and D differing here."""
    upsample = Fp32BilinearUpsample()
    x = torch.randn(1, 4, 6, 8)
    output = upsample(x)
    assert output.shape[-2:] == (12, 16)
    assert output.dtype == x.dtype
    assert not list(upsample.parameters()), 'must be parameter-free for checkpoint interchangeability'


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
def test_the_fp32_upsample_round_trips_reduced_precision(dtype):
    """The reason it exists: bilinear interpolation lacks reduced-precision CUDA kernels on older torch builds, so the
    op is forced through float32 and cast back. The output dtype must still be the input's."""
    output = Fp32BilinearUpsample()(torch.randn(1, 4, 6, 8).to(dtype))
    assert output.dtype == dtype
    assert output.shape[-2:] == (12, 16)


def test_bottleneck_attention_preserves_its_input_shape():
    attention = BottleneckAttention(16).eval()
    x = torch.randn(2, 16, 6, 8)
    with torch.no_grad():
        assert attention(x).shape == x.shape


@pytest.mark.parametrize('kind', ['group', 'batch'])
def test_every_normalization_kind_builds(kind):
    assert isinstance(make_normalization(kind, 8), nn.Module)


def test_the_search_spaces_only_offer_group_normalization(search_spaces):
    """Group norm only, for all three families: batch statistics over a 99.93 %-zero field are dominated by the empty
    cells, and the warm start requires the upstream and the MC net to normalise identically."""
    for family, space in search_spaces.items():
        normalization = space['unet']['normalization'] if 'unet' in space else 'group'
        offered = normalization.get('choices', [normalization]) if isinstance(normalization, dict) \
            else [normalization]
        assert set(offered) == {'group'}, f'{family} offers {offered}'


@pytest.mark.parametrize('name', ['relu', 'silu', 'gelu'])
def test_every_activation_builds(name):
    assert isinstance(make_activation(name), nn.Module)


def test_an_unknown_normalization_or_activation_raises():
    with pytest.raises(ValueError):
        make_normalization('layer_maybe', 8)
    with pytest.raises(ValueError):
        make_activation('swish_ish')


# =====================================================================================================================
# PlattScaling — hourly calibration on a logit
# =====================================================================================================================
def test_platt_scaling_is_initialised_as_an_exact_no_op():
    """It is fitted in a separate phase, so at initialisation it must not perturb the head at all — otherwise phase 1's
    best checkpoint is degraded the moment the calibration phase starts."""
    platt = PlattScaling().eval()
    logits = torch.tensor([[-3.0, 0.0, 2.5]])
    with torch.no_grad():
        assert torch.allclose(platt(logits), logits, atol=1e-6)


def test_platt_scaling_is_an_affine_map_on_the_logit():
    platt = PlattScaling()
    with torch.no_grad():
        for name, parameter in platt.named_parameters():
            parameter.fill_(0.5 if 'w' in name or 'scale' in name else 0.25)
    logits = torch.tensor([[1.0, 2.0]])
    with torch.no_grad():
        calibrated = platt(logits)
    # affine in the logit: equal input spacing gives equal output spacing
    assert float(calibrated[0, 1] - calibrated[0, 0]) == pytest.approx(
        float(platt(torch.tensor([[3.0, 4.0]]))[0, 1] - platt(torch.tensor([[3.0, 4.0]]))[0, 0]), rel=1e-5
    )


def test_platt_scaling_has_exactly_two_learnable_scalars():
    """A slope and an intercept. More would make it a general warp rather than a calibration, and it is fitted on a
    validation split where overfitting is the whole risk."""
    assert sum(p.numel() for p in PlattScaling().parameters()) == 2


# =====================================================================================================================
# MonotoneCalibration — the daily hour warp
# =====================================================================================================================
def test_a_none_structure_is_rejected_by_the_LAYER(net):
    """`calibration.regression.structure: none` is handled by not BUILDING the layer — the module passes
    `regression_calibration=None` — rather than by a no-op structure inside it. Asserting both halves keeps the
    responsibility split visible."""
    with pytest.raises(ValueError):
        MonotoneCalibration('none', num_sigmoids=4)
    assert getattr(net(regression_calibration=None), 'regression_calibration', None) is None


@pytest.mark.parametrize('structure', list(REGRESSION_CALIBRATION_STRUCTURES))
def test_a_freshly_built_calibrator_is_APPROXIMATELY_the_identity(structure):
    """It is fitted in a separate phase after the backbone is trained, so at initialisation it must barely perturb the
    prediction — otherwise phase 1's best checkpoint is degraded the moment the calibration phase starts.

    `monotone_smooth` is the identity only approximately: its sum-of-sigmoids warp is initialised near-linear and
    deviates by ~0.2 % of the range at the top end. Well inside a rounding to whole lightning-hours, which is the scale
    that matters here, so this is a tolerance on a real property rather than a slackened exact one.
    """
    calibrator = MonotoneCalibration(structure, num_sigmoids=4).eval()
    prediction = torch.tensor([[0.0, 3.0, 12.0, 24.0]])
    with torch.no_grad():
        warped = calibrator(prediction)
    assert torch.allclose(warped, prediction, atol=0.05), f'{structure} deviates by {(warped - prediction).abs().max()}'


@pytest.mark.parametrize('structure', list(REGRESSION_CALIBRATION_STRUCTURES))
def test_every_calibration_structure_is_monotone_non_decreasing(structure):
    """The point of the layer: it may rescale predicted hours but must never reorder them, or it would destroy the
    ranking that ``average_precision_occurrence`` and ``roc_auc`` measure."""
    calibrator = MonotoneCalibration(structure, num_sigmoids=4).eval()
    with torch.no_grad():
        for parameter in calibrator.parameters():
            parameter.normal_(0.0, 0.5)
        rising = torch.linspace(0.0, 24.0, 60).unsqueeze(0)
        warped = calibrator(rising)
    differences = warped[0, 1:] - warped[0, :-1]
    assert float(differences.min()) >= -1e-5, f'{structure} is not monotone: min step {float(differences.min())}'


@pytest.mark.parametrize('structure', list(REGRESSION_CALIBRATION_STRUCTURES))
def test_every_calibration_structure_keeps_predictions_non_negative(structure):
    calibrator = MonotoneCalibration(structure, num_sigmoids=4).eval()
    with torch.no_grad():
        for parameter in calibrator.parameters():
            parameter.normal_(0.0, 0.5)
        warped = calibrator(torch.linspace(0.0, 24.0, 40).unsqueeze(0))
    assert float(warped.min()) >= -1e-5


def test_an_unknown_calibration_structure_raises():
    with pytest.raises(ValueError):
        MonotoneCalibration('spline_ish', num_sigmoids=4)


def test_softplus_one_is_the_softplus_inverse_anchor():
    """``SOFTPLUS_ONE`` is the value whose softplus is 1, used to initialise a monotone warp at the identity. A wrong
    constant would make the calibrator start as a silent rescaling."""
    assert float(torch.nn.functional.softplus(torch.tensor(SOFTPLUS_ONE))) == pytest.approx(1.0, abs=1e-6)


# =====================================================================================================================
# Block 5c — the two calibration parameter groups
#
# These exist for exactly one caller: ``UnetModuleBase.set_phase``, which freezes the whole backbone and re-enables
# only the group belonging to the phase being entered. An empty generator where parameters were expected produces a
# calibration phase that trains NOTHING — the fit runs, the checkpoint is written, and the calibrator stays at its
# identity initialisation. No error anywhere.
# =====================================================================================================================
@pytest.mark.parametrize('accessor,builder_kwarg', [
    ('regression_calibration_parameters', 'regression_calibration'),
    ('output_calibration_parameters', 'output_calibration'),
])
def test_a_calibration_group_is_EMPTY_when_its_layer_is_disabled(accessor, builder_kwarg):
    """Empty rather than raising, because the base class calls BOTH accessors on every fitting phase to freeze them —
    including in the mode where one of the two layers cannot exist."""
    disabled = {'regression_calibration': None, 'output_calibration': False}[builder_kwarg]
    net = DeterministicUnetNet(IN_CHANNELS, UNET, **{builder_kwarg: disabled})
    assert list(getattr(net, accessor)()) == []


def test_the_monotone_group_is_exactly_the_regression_calibrators_parameters(unet_trial):
    """Exactly, not a superset: ``set_phase`` freezes everything and unfreezes this group, so a group that leaked a
    backbone parameter would keep training the network during a phase meant to fit a scalar warp."""
    net = DeterministicUnetNet(IN_CHANNELS, UNET, regression_calibration={'structure': 'monotone_smooth'})

    group = list(net.regression_calibration_parameters())
    assert group, 'the monotone calibrator must expose parameters to fit'
    assert {id(parameter) for parameter in group} == \
        {id(parameter) for parameter in net.regression_calibration.parameters()}
    backbone = {id(parameter) for parameter in net.backbone.parameters()}
    assert not ({id(parameter) for parameter in group} & backbone)


def test_the_platt_group_is_exactly_the_platt_layers_parameters():
    net = DeterministicUnetNet(IN_CHANNELS, UNET, output_calibration=True)

    group = list(net.output_calibration_parameters())
    assert group
    assert {id(parameter) for parameter in group} == \
        {id(parameter) for parameter in net.output_calibration.parameters()}


def test_the_two_groups_are_DISJOINT_when_both_layers_somehow_exist():
    """They never coexist in practice — Platt is hourly-only and the monotone warp daily-only — but ``set_phase``
    unfreezes one and freezes the other unconditionally, so an overlap would leave a parameter frozen mid-phase."""
    net = DeterministicUnetNet(IN_CHANNELS, UNET, regression_calibration={'structure': 'monotone_smooth'},
                               output_calibration=True)

    monotone = {id(parameter) for parameter in net.regression_calibration_parameters()}
    platt = {id(parameter) for parameter in net.output_calibration_parameters()}
    assert monotone and platt and not (monotone & platt)
