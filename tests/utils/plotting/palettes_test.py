"""Tests for src/utils/plotting/palettes.py — the general-purpose IBM / Tol colour libraries.

Untouched plumber template code, byte-identical to branch A's (md5-checked), so nothing here is a port or an adaptation.
It is tested because the mirror requires it and because two of its behaviours are relied on elsewhere: the
``rcParams`` prop cycle it installs at import is what gives every line figure its colours without any figure asking,
and ``get_color`` is used as a pass-through resolver by ``reporting.py``.

⚠️ The 02a lightning colours are deliberately NOT here — they live in ``plotting/maps.py`` beside their only consumer,
``make_lightning_cmap``. This file holds the design libraries that reach the LINE figures; the lightning ramps define a
value axis. See Block 4 in the step-3 plan for the reasoning.
"""
import matplotlib
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                        # noqa: E402

from src.utils.plotting import palettes                                      # noqa: E402


# =====================================================================================================================
# The two design libraries
# =====================================================================================================================
def test_both_libraries_are_populated():
    assert len(palettes.ibm) == 6
    assert len(palettes.tol) == 8


@pytest.mark.parametrize('library', ['ibm', 'tol'])
def test_every_colour_is_a_full_length_hex_string(library):
    for name, value in getattr(palettes, library).items():
        assert value.startswith('#') and len(value) == 7, f'{library}.{name} = {value}'
        int(value[1:], 16)                                       # parses as hex


def test_the_named_orders_cover_their_libraries():
    """``ibm_colors`` and ``tol_colors`` are the ORDERS the prop cycle is built from, so a name missing from one would
    silently drop that colour out of the line-figure rotation."""
    assert set(palettes.ibm_colors) == set(palettes.ibm)
    assert set(palettes.tol_colors) == set(palettes.tol)


# =====================================================================================================================
# get_color — a resolver that passes unknown values through
# =====================================================================================================================
def test_a_library_name_resolves_to_its_hex():
    assert palettes.get_color('orange') == palettes.ibm['orange']
    assert palettes.get_color('sand') == palettes.tol['sand']


def test_an_unknown_name_passes_through_UNCHANGED():
    """This is what makes it usable as a general resolver: ``reporting.py`` hands it both library names and literal
    matplotlib colours, and the literals must survive. Raising instead would force every caller to know which is which."""
    for value in ('steelblue', '#ABCDEF', 'darkorange'):
        assert palettes.get_color(value) == value


def test_none_passes_through():
    """Matplotlib treats ``color=None`` as "use the prop cycle", so it must not be resolved to anything."""
    assert palettes.get_color(None) is None


def test_ibm_wins_when_a_name_exists_in_both_libraries():
    """``aqua`` is in both. The lookup order is the contract — a caller asking for it must always get the same colour."""
    assert 'aqua' in palettes.ibm and 'aqua' in palettes.tol
    assert palettes.get_color('aqua') == palettes.ibm['aqua']


# =====================================================================================================================
# The colormap factories
# =====================================================================================================================
def test_a_linear_palette_runs_from_white_to_its_colour():
    """Used for the confusion matrix, where the empty end of the scale has to read as absence rather than as a colour."""
    cmap = palettes.ibm_linear_palette_factory('purple')
    assert isinstance(cmap, LinearSegmentedColormap)
    assert tuple(round(channel, 3) for channel in cmap(0.0)[:3]) == (1.0, 1.0, 1.0)
    assert cmap(1.0)[:3] != cmap(0.0)[:3]


def test_a_diverging_palette_is_white_in_the_MIDDLE():
    """The residual diagnostics draw signed fields, so zero has to be the neutral point — a diverging map whose centre
    is not white would show "no correction" as a colour and read as a bias."""
    cmap = palettes.ibm_diverging_palette_factory('orange', 'purple')
    # sampled at the colormap's 256-step resolution, so the midpoint lands a fraction off the exact white node
    assert all(channel > 0.99 for channel in cmap(0.5)[:3]), cmap(0.5)
    assert cmap(0.0)[:3] != cmap(1.0)[:3]


def test_the_diverging_ends_are_the_two_requested_colours():
    cmap = palettes.ibm_diverging_palette_factory('orange', 'purple')
    hex_of = lambda rgb: '#{:02X}{:02X}{:02X}'.format(*(int(round(c * 255)) for c in rgb[:3]))
    assert hex_of(cmap(0.0)).upper() == palettes.ibm['orange'].upper()
    assert hex_of(cmap(1.0)).upper() == palettes.ibm['purple'].upper()


def test_a_linear_palette_accepts_a_custom_base_colour():
    cmap = palettes.ibm_linear_palette_factory('purple', basecolor=(0.0, 0.0, 0.0))
    assert tuple(round(channel, 3) for channel in cmap(0.0)[:3]) == (0.0, 0.0, 0.0)


def test_every_prebuilt_linear_palette_exists():
    assert set(palettes.ibm_linear_palettes) == {f'ibm_{name}s' for name in palettes.ibm_colors}


def test_the_prebuilt_diverging_palettes_exclude_the_identity_pairs():
    """A colour diverging to itself would be a flat map, so the product skips ``a == b``."""
    expected = len(palettes.ibm_colors) * (len(palettes.ibm_colors) - 1)
    assert len(palettes.ibm_diverging_palettes) == expected
    assert not any(name.split('_to_')[0].removeprefix('ibm_') == name.split('_to_')[1]
                   for name in palettes.ibm_diverging_palettes)


# =====================================================================================================================
# The global side effect
# =====================================================================================================================
def test_importing_the_module_installs_the_prop_cycle():
    """The reason ``reporting.py`` imports this module for its side effect: every line figure picks the palette up from
    ``rcParams`` without asking, so no figure has to name colours. A figure that DOES name them (the PSD curves' steelblue
    and darkorange) is opting out deliberately."""
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    assert len(cycle_colors) == len(palettes.ibm_colors) + len(palettes.tol_colors)
    assert cycle_colors[0] == palettes.ibm['orange'], 'the IBM order leads the cycle'


def test_the_cycle_holds_no_duplicate_colours():
    """Two series drawn in the same colour is the failure this prevents, and the two libraries share names (``aqua``)
    without sharing values."""
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    assert len(set(cycle_colors)) == len(cycle_colors)


@pytest.mark.source_invariant
def test_the_lightning_ramps_are_NOT_defined_here():
    """A deliberate departure from the step-3 plan's line 40. The warm/cool/grey ramps define the lightning-hours VALUE
    AXIS and have exactly one consumer, ``make_lightning_cmap``, in the same file. Moving them here would separate that
    function from its own data for no second caller."""
    import inspect

    source = inspect.getsource(palettes)
    for token in ('_BASE_COLORS_WARM', '_BASE_COLORS_COOL', 'lightning'):
        assert token not in source, token
