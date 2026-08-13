"""Tests for src/utils/modeling/diffusion.py — the conditional flow-matching velocity network.

The file is byte-identical on both source branches and was taken as-is, so nothing here is a "port": neither branch
tested it. These pin the two things that would break silently.

The straight-path interpolant is the more important one. ``flow_matching_targets`` is three lines and its correctness
is invisible at training time: a wrong interpolant still produces a decreasing MSE, because the network happily learns
whatever target it is given. The failure only surfaces as samples that do not look like the data, several hours later.
"""
import numpy as np
import pytest
import torch

from src.utils.modeling.diffusion import (
    ConvDecoder, ConvStem, DiTBlock, FlowVelocityNet, TimestepEmbedder, build_2d_sincos_pos_embed,
    flow_matching_targets, modulate, sample, sinusoidal_embedding,
)

HEIGHT, WIDTH = 16, 24
COND_CHANNELS = 5


# =====================================================================================================================
# The straight (rectified-flow) interpolant
# =====================================================================================================================
def test_the_path_endpoints_are_the_noise_and_the_data():
    """``x_t = (1-t) z + t x1``: at t=0 the point IS the noise, at t=1 it IS the data. Getting the direction backwards
    is the classic flow-matching bug and it trains perfectly well."""
    x1 = torch.randn(3, HEIGHT, WIDTH)
    z = torch.randn(3, HEIGHT, WIDTH)

    at_zero, _ = flow_matching_targets(x1, torch.zeros(3), noise=z)
    at_one, _ = flow_matching_targets(x1, torch.ones(3), noise=z)

    assert torch.allclose(at_zero, z, atol=1e-6)
    assert torch.allclose(at_one, x1, atol=1e-6)


def test_the_velocity_target_is_the_constant_displacement():
    x1, z = torch.randn(3, HEIGHT, WIDTH), torch.randn(3, HEIGHT, WIDTH)
    _, velocity = flow_matching_targets(x1, torch.full((3,), 0.4), noise=z)
    assert torch.allclose(velocity, x1 - z, atol=1e-6)


def test_the_velocity_target_does_not_depend_on_t():
    """The consequence of a straight path, and why ONE forward pass trains the model: the regression target is the same
    at every time, so no time-dependent reweighting is needed."""
    x1, z = torch.randn(2, HEIGHT, WIDTH), torch.randn(2, HEIGHT, WIDTH)
    _, early = flow_matching_targets(x1, torch.full((2,), 0.1), noise=z)
    _, late = flow_matching_targets(x1, torch.full((2,), 0.9), noise=z)
    assert torch.allclose(early, late, atol=1e-6)


def test_the_interpolant_is_linear_in_t():
    x1, z = torch.randn(1, HEIGHT, WIDTH), torch.randn(1, HEIGHT, WIDTH)
    quarter, _ = flow_matching_targets(x1, torch.full((1,), 0.25), noise=z)
    half, _ = flow_matching_targets(x1, torch.full((1,), 0.5), noise=z)
    three_quarters, _ = flow_matching_targets(x1, torch.full((1,), 0.75), noise=z)
    assert torch.allclose(half - quarter, three_quarters - half, atol=1e-6)


def test_t_is_broadcast_per_sample_not_shared():
    """``t`` is ``[B]`` and must be broadcast over the spatial axes independently per sample — a shared scalar would
    make every item in the batch sit at the same time, quietly halving the time coverage per step."""
    x1 = torch.ones(2, 4, 4)
    z = torch.zeros(2, 4, 4)
    x_t, _ = flow_matching_targets(x1, torch.tensor([0.0, 1.0]), noise=z)
    assert torch.allclose(x_t[0], torch.zeros(4, 4), atol=1e-6)
    assert torch.allclose(x_t[1], torch.ones(4, 4), atol=1e-6)


def test_noise_is_drawn_when_not_supplied():
    x1 = torch.zeros(4, 8, 8)
    first, _ = flow_matching_targets(x1, torch.full((4,), 0.5))
    second, _ = flow_matching_targets(x1, torch.full((4,), 0.5))
    assert not torch.allclose(first, second), 'z must be resampled, not fixed'


# =====================================================================================================================
# The network: shapes, and the divisibility constraints the search space satisfies by construction
# =====================================================================================================================
@pytest.fixture
def net():
    """Built through ``DiffusionModule._net_config`` rather than a hand-written config dict, so the test exercises the
    same translation the module does — a drift between the two would otherwise be invisible here."""
    from src.utils.modeling.diffusion_module import DiffusionModule

    def build(hidden_dim=128, n_blocks=2, num_heads=4, patch_size=2, cond_channels=COND_CHANNELS):
        config = DiffusionModule._net_config({'hidden_dim': hidden_dim, 'n_blocks': n_blocks,
                                              'num_heads': num_heads, 'patch_size': patch_size})
        return FlowVelocityNet(cond_channels, config).eval()
    return build


def test_the_velocity_field_has_the_shape_of_the_target(net):
    """The network predicts a velocity per target cell, so its output must match the noisy-target map exactly — a
    patch-size change must not leak into the output resolution."""
    model = net()
    x_t = torch.randn(2, HEIGHT, WIDTH)
    condition = torch.randn(2, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        velocity = model(x_t, torch.rand(2), condition)
    assert velocity.shape == x_t.shape


@pytest.mark.parametrize('patch_size', [1, 2])
@pytest.mark.parametrize('hidden_dim', [128, 256])
def test_every_sampled_architecture_in_the_search_space_runs(patch_size, hidden_dim, net):
    """The search space offers ``patch_size in {1, 2}`` and ``hidden_dim in {128, 256}`` with ``num_heads`` FIXED at 4,
    and its comment claims the divisibility constraints (by num_heads, and by 4 for the 2-D positional embedding) hold
    by construction. This is that claim, executed over the whole grid."""
    model = net(hidden_dim=hidden_dim, patch_size=patch_size)
    with torch.no_grad():
        velocity = model(torch.randn(1, HEIGHT, WIDTH), torch.rand(1),
                         torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH))
    assert velocity.shape == (1, HEIGHT, WIDTH)
    assert torch.isfinite(velocity).all()


def test_the_residual_channel_count_is_accepted(net):
    """Residual mode appends the upstream prediction as the LAST conditioning channel, so the net must take 6."""
    model = net(cond_channels=COND_CHANNELS + 1)
    with torch.no_grad():
        velocity = model(torch.randn(1, HEIGHT, WIDTH), torch.rand(1),
                         torch.randn(1, COND_CHANNELS + 1, HEIGHT, WIDTH))
    assert velocity.shape == (1, HEIGHT, WIDTH)


def test_a_freshly_initialised_net_emits_exactly_zero_velocity(net):
    """adaLN-Zero, and a stronger property than "it runs": the final modulation and the decoder head are BOTH
    zero-initialised, so an untrained network is the exact identity flow rather than a random one. That is what makes
    the first training steps stable, and it is why the two sensitivity tests below have to break the zero-init first.

    ⚠️ It also means "the output changed when I changed the input" is vacuously false at initialisation — a trap for
    anyone writing a conditioning test against a fresh net.
    """
    model = net()
    with torch.no_grad():
        velocity = model(torch.randn(2, HEIGHT, WIDTH), torch.rand(2),
                         torch.randn(2, COND_CHANNELS, HEIGHT, WIDTH))
    assert float(velocity.abs().max()) == 0.0


@pytest.fixture
def perturbed_net(net):
    """A net with the adaLN-Zero gates broken, so it behaves like a partially trained one. Needed for any test of
    input sensitivity: at initialisation the output is identically zero (above)."""
    def build(**kwargs):
        model = net(**kwargs)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.numel() and float(parameter.abs().max()) == 0.0:
                    parameter.normal_(0.0, 0.05)
        return model.eval()
    return build


def test_the_conditioning_actually_changes_the_velocity(perturbed_net):
    """Otherwise the model is an unconditional generator wearing a conditional interface, and would sample plausible
    lightning fields unrelated to the day's ERA5 predictors — while training perfectly well."""
    model = perturbed_net()
    x_t, t = torch.randn(1, HEIGHT, WIDTH), torch.rand(1)
    with torch.no_grad():
        first = model(x_t, t, torch.zeros(1, COND_CHANNELS, HEIGHT, WIDTH))
        second = model(x_t, t, torch.ones(1, COND_CHANNELS, HEIGHT, WIDTH))
    assert not torch.allclose(first, second)


def test_the_time_conditioning_actually_changes_the_velocity(perturbed_net):
    model = perturbed_net()
    x_t, condition = torch.randn(1, HEIGHT, WIDTH), torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        early = model(x_t, torch.zeros(1), condition)
        late = model(x_t, torch.ones(1), condition)
    assert not torch.allclose(early, late)


def test_the_noisy_target_changes_the_velocity(perturbed_net):
    """The velocity must depend on ``x_t`` too — a field that ignores its own position on the path cannot integrate
    to anything."""
    model = perturbed_net()
    t, condition = torch.rand(1), torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        first = model(torch.zeros(1, HEIGHT, WIDTH), t, condition)
        second = model(torch.ones(1, HEIGHT, WIDTH), t, condition)
    assert not torch.allclose(first, second)


# =====================================================================================================================
# ODE sampling
# =====================================================================================================================
def test_sampling_returns_a_target_shaped_map(net):
    model = net()
    condition = torch.randn(2, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        drawn = sample(model, condition, (HEIGHT, WIDTH), num_steps=3)
    assert drawn.shape[-2:] == (HEIGHT, WIDTH)
    assert torch.isfinite(drawn).all()


def test_sampling_is_reproducible_under_a_generator(net):
    """The evaluation stage seeds per batch so a re-run of the same checkpoint reports the same numbers."""
    model = net()
    condition = torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        first = sample(model, condition, (HEIGHT, WIDTH), num_steps=3,
                       generator=torch.Generator().manual_seed(0))
        second = sample(model, condition, (HEIGHT, WIDTH), num_steps=3,
                        generator=torch.Generator().manual_seed(0))
    assert torch.allclose(first, second)


def test_different_seeds_give_different_draws(net):
    """The whole point of the family: the draws ARE the ensemble, so two seeds must not collapse to one member — that
    would give a zero spread and NaN spread-skill through ``ddof=1``."""
    model = net()
    condition = torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        first = sample(model, condition, (HEIGHT, WIDTH), num_steps=3,
                       generator=torch.Generator().manual_seed(0))
        second = sample(model, condition, (HEIGHT, WIDTH), num_steps=3,
                        generator=torch.Generator().manual_seed(1))
    assert not torch.allclose(first, second)


def test_an_untrained_net_integrates_to_its_initial_noise(net):
    """The consequence of adaLN-Zero at the sampling level: a zero velocity field means the Euler steps do not move the
    point, so the draw IS the initial noise. Worth pinning because it explains why an untrained checkpoint produces
    pure noise maps rather than an error."""
    model = net()
    condition = torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        two_steps = sample(model, condition, (HEIGHT, WIDTH), num_steps=2,
                           generator=torch.Generator().manual_seed(0))
        many_steps = sample(model, condition, (HEIGHT, WIDTH), num_steps=16,
                            generator=torch.Generator().manual_seed(0))
    assert torch.allclose(two_steps, many_steps)


def test_more_integration_steps_change_the_result(perturbed_net):
    """``flow.n_steps`` is searched over 8-32, which only means something if the step count actually matters once the
    velocity field is non-zero."""
    model = perturbed_net()
    condition = torch.randn(1, COND_CHANNELS, HEIGHT, WIDTH)
    with torch.no_grad():
        coarse = sample(model, condition, (HEIGHT, WIDTH), num_steps=2,
                        generator=torch.Generator().manual_seed(0))
        fine = sample(model, condition, (HEIGHT, WIDTH), num_steps=16,
                      generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(coarse, fine)


# =====================================================================================================================
# The building blocks
# =====================================================================================================================
def test_sinusoidal_embedding_is_deterministic_and_bounded():
    values = torch.tensor([0.0, 0.5, 1.0])
    embedding = sinusoidal_embedding(values, dim=32)
    assert embedding.shape == (3, 32)
    assert torch.allclose(embedding, sinusoidal_embedding(values, dim=32))
    assert float(embedding.abs().max()) <= 1.0 + 1e-6          # sin/cos


def test_sinusoidal_embedding_separates_nearby_times():
    close = sinusoidal_embedding(torch.tensor([0.10, 0.11]), dim=64)
    assert not torch.allclose(close[0], close[1])


def test_the_2d_positional_embedding_is_unique_per_cell():
    """A repeated position would make two grid cells indistinguishable to the attention, which is exactly the kind of
    bug that shows up as a spatially smeared sample rather than an error."""
    embedding = build_2d_sincos_pos_embed(6, 8, dim=32, device=torch.device('cpu'), dtype=torch.float32)
    flattened = embedding.reshape(-1, embedding.shape[-1])
    assert flattened.shape[0] == 48
    distinct = {tuple(np.round(row.numpy(), 6)) for row in flattened}
    assert len(distinct) == 48


def test_the_positional_embedding_dim_must_divide_by_four():
    """The 2-D sin/cos construction splits the width four ways, which is why the search space's ``hidden_dim`` choices
    are both multiples of 4."""
    for dim in (32, 64, 128, 256):
        assert build_2d_sincos_pos_embed(4, 4, dim=dim, device=torch.device('cpu'),
                                         dtype=torch.float32).shape[-1] == dim


def test_modulate_is_an_affine_scale_and_shift():
    """``shift`` and ``scale`` arrive as ``[B, hidden]`` (the adaLN projection's output) and are unsqueezed to broadcast
    over the TOKEN axis — so one modulation applies to every token of a sample, which is what makes it a per-sample
    conditioning rather than a per-token one."""
    x = torch.randn(2, 5, 8)
    shift, scale = torch.randn(2, 8), torch.randn(2, 8)
    expected = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    assert torch.allclose(modulate(x, shift, scale), expected, atol=1e-6)


def test_modulate_is_the_identity_at_zero():
    """adaLN-Zero: the blocks start as the identity so a freshly initialised network is a well-behaved starting point
    rather than noise."""
    x = torch.randn(2, 5, 8)
    assert torch.allclose(modulate(x, torch.zeros(2, 8), torch.zeros(2, 8)), x, atol=1e-6)


def test_the_timestep_embedder_maps_a_batch_of_times_to_the_hidden_width():
    embedder = TimestepEmbedder(128).eval()
    with torch.no_grad():
        embedded = embedder(torch.rand(4))
    assert embedded.shape == (4, 128)


def test_the_timestep_embedder_separates_nearby_times():
    embedder = TimestepEmbedder(128).eval()
    with torch.no_grad():
        embedded = embedder(torch.tensor([0.10, 0.11]))
    assert not torch.allclose(embedded[0], embedded[1])


def test_a_dit_block_preserves_the_token_shape():
    block = DiTBlock(128, num_heads=4, mlp_ratio=4.0, dropout=0.0).eval()
    tokens = torch.randn(2, 20, 128)
    with torch.no_grad():
        assert block(tokens, torch.randn(2, 128)).shape == tokens.shape


def test_the_stem_and_decoder_are_shape_inverses():
    """The decoder must undo the stem's patch stride exactly, or the velocity map comes back at the wrong resolution —
    which at ``patch_size: 1`` would be invisible and at 2 would silently halve the output grid."""
    stem = ConvStem(COND_CHANNELS + 1, stem_channels=32, hidden=128, patch_size=2,
                    normalization='group', activation='silu').eval()
    decoder = ConvDecoder(hidden=128, patch_size=2, normalization='group', activation='silu').eval()
    fused = torch.randn(2, COND_CHANNELS + 1, HEIGHT, WIDTH)
    with torch.no_grad():
        tokens = stem(fused)
        decoded = decoder(tokens)
    # the decoder restores the patch stride; FlowVelocityNet.forward then crops to the exact input size, which is why
    # a non-multiple-of-patch_size grid still comes back at [H, W]
    assert decoded.shape[-2:] >= (HEIGHT, WIDTH)
    assert decoded.shape[1] == 1, 'the decoder emits a single velocity channel'


def test_an_odd_grid_still_comes_back_at_the_input_size(net):
    """The real grid is 101 x 149 — both ODD, so at ``patch_size: 2`` the token grid does not divide the input evenly
    and the decoder overshoots. ``forward`` crops back, and this is the test that the crop exists."""
    model = net(patch_size=2)
    with torch.no_grad():
        velocity = model(torch.randn(1, 101, 149), torch.rand(1), torch.randn(1, COND_CHANNELS, 101, 149))
    assert velocity.shape == (1, 101, 149)


def test_a_non_power_of_two_patch_size_is_rejected():
    """The decoder halves the stride ``log2(patch_size)`` times, so anything else would silently round."""
    with pytest.raises(ValueError, match='power of two'):
        ConvDecoder(hidden=128, patch_size=3, normalization='group', activation='silu')
