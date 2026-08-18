"""Tests for src/utils/plotting/maps.py — the 02a lightning map grammar.

The behavioural half of Block 4's verification gate, which is where the emphasis belongs: **every score in this repo is
computed on the arrays rather than on the rendered picture**, so a mirrored or mis-coloured map is invisible to every
metric. Nothing here was ported — neither source branch had a test for it.

The north-edge orientation is the headline. ``origin='upper'`` pairs with ``extent=GRID_EXTENT``, whose last element is
the NORTHERN latitude, and together they put array row 0 at the north edge. Flipping either mirrors the field about the
domain's mid-latitude, and the assertion cannot be read off the artifact: cartopy REGRIDS the array into the target CRS
and re-emits it as ``origin='lower'`` in projected metres, so the drawn object no longer carries the kwargs it was given.
Hence a rasterised probe.
"""
import ast
import inspect

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                              # noqa: E402
from cartopy.mpl.gridliner import Gridliner                                  # noqa: E402

from src.utils.plotting import maps                                          # noqa: E402

H, W = 101, 149                                                              # the real grid


@pytest.fixture(scope='module')
def context():
    return maps.geographic_context()


# =====================================================================================================================
# cartopy is a HARD requirement
# =====================================================================================================================
def test_geographic_context_returns_europp_over_platecarree(context):
    projection, data_crs = context
    assert type(projection).__name__ == 'EuroPP'
    assert type(data_crs).__name__ == 'PlateCarree'


def test_geographic_context_can_never_degrade_to_plain_axes():
    """cartopy is a hard dependency (``minimal_requirements.txt``), so a missing install raises at IMPORT. The
    alternative — silently dropping the projection — would emit figures in raw pixel indices that look plausible and are
    not maps, which is exactly what makes branch D's reporting unusable."""
    source = inspect.getsource(maps.geographic_context)
    tree = ast.parse(inspect.getsource(maps).split('def geographic_context')[0] + 'pass')
    assert 'try' not in source
    assert 'None' not in source
    assert maps.geographic_context() != (None, None)


@pytest.mark.source_invariant
def test_the_cartopy_unavailable_WARNING_PATH_is_gone_entirely():
    """Not just unused — absent. The optional-cartopy version logged a warning and carried on with plain axes, and a
    warning in a pipeline log is not a stop: the run completed and wrote figures in raw pixel indices. Removing the
    logger removes the only way that path could come back quietly."""
    import tokenize

    with open(maps.__file__, 'rb') as handle:
        identifiers = {token.string for token in tokenize.tokenize(handle.readline)
                       if token.type == tokenize.NAME}
    assert 'logger' not in identifiers
    assert 'logging' not in identifiers


def test_cartopy_is_imported_at_module_scope():
    tree = ast.parse(inspect.getsource(maps))
    module_level = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert 'cartopy.crs' in module_level


# =====================================================================================================================
# The extents, and the deliberate 55 N crop
# =====================================================================================================================
def test_the_grid_extent_is_the_metadata_domain():
    assert maps.GRID_EXTENT == (-12.0, 25.0, 35.0, 60.0)


def test_the_display_extent_lies_inside_the_data_extent():
    left, right, bottom, top = maps.DISPLAY_EXTENT
    assert maps.GRID_EXTENT[0] <= left and right <= maps.GRID_EXTENT[1]
    assert maps.GRID_EXTENT[2] <= bottom and top <= maps.GRID_EXTENT[3]


def test_the_display_deliberately_crops_the_northern_strip():
    """A recorded decision, not an oversight: the view stops at 55 N while the data runs to 60 N."""
    assert maps.DISPLAY_EXTENT[3] == 55.0 < maps.GRID_EXTENT[3]


def test_the_crop_hides_the_top_twenty_array_rows():
    """The consequence, stated so it cannot be mistaken for a bug: rows 0-20 of every ``[101, 149]`` field are never
    drawn. Real data lives there (southern Scandinavia, northern UK)."""
    degrees_per_row = (maps.GRID_EXTENT[3] - maps.GRID_EXTENT[2]) / (H - 1)
    hidden = int(round((maps.GRID_EXTENT[3] - maps.DISPLAY_EXTENT[3]) / degrees_per_row))
    assert 18 <= hidden <= 22, f'{hidden} rows hidden'


def test_europp_y_increases_northward():
    """The projection property that makes "origin='upper' + the north latitude last in extent" the correct pairing."""
    projection, data_crs = maps.geographic_context()
    north = projection.transform_point(6.5, maps.GRID_EXTENT[3], data_crs)[1]
    south = projection.transform_point(6.5, maps.GRID_EXTENT[2], data_crs)[1]
    assert north > south


# =====================================================================================================================
# THE north-edge claim, proved by rasterising
# =====================================================================================================================
def _render_lit_row(row, context):
    """Rasterise a field that is 24 h/day on one row and 0 elsewhere; return the reddish pixels' mean image row."""
    projection, data_crs = context
    field = np.zeros((H, W))
    field[row, :] = 24.0
    cmap, norm = maps.make_lightning_cmap(24.0)

    figure = plt.figure(figsize=(5, 5))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
    maps.draw_map(ax, field, f'row {row}', data_crs, cmap, norm)
    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].astype(int)
    bbox = ax.get_window_extent()
    plt.close(figure)

    red, green, blue = buffer[:, :, 0], buffer[:, :, 1], buffer[:, :, 2]
    lit = (red > 120) & (red - blue > 60) & (red - green > 60)     # the top of the warm ramp is #992015
    rows_lit = np.nonzero(lit)[0]
    # buffer row 0 is the TOP of the image; matplotlib's bbox y is measured from the BOTTOM
    axes_mid = buffer.shape[0] - (bbox.y0 + bbox.y1) / 2.0
    return rows_lit, axes_mid


@pytest.fixture(scope='module')
def orientation_probes(context):
    """Rows 25 (~53.8 N) and 95 (~36.3 N) — both INSIDE the displayed window, since the 55 N crop hides row 0."""
    return _render_lit_row(25, context), _render_lit_row(95, context)


def test_both_probes_actually_paint_pixels(orientation_probes):
    (north_rows, _), (south_rows, _) = orientation_probes
    assert north_rows.size > 50 and south_rows.size > 50


def test_ROW_ZERO_IS_NORTH(orientation_probes):
    """The whole point of this file. Image y grows DOWNWARD, so a northern row must have the SMALLER mean row."""
    (north_rows, _), (south_rows, _) = orientation_probes
    assert north_rows.mean() < south_rows.mean(), \
        f'northern row at y={north_rows.mean():.0f}, southern at y={south_rows.mean():.0f}'


def test_the_northern_row_lands_in_the_upper_half_of_the_axes(orientation_probes):
    (north_rows, axes_mid), _ = orientation_probes
    assert north_rows.mean() < axes_mid


def test_the_southern_row_lands_in_the_lower_half_of_the_axes(orientation_probes):
    _, (south_rows, axes_mid) = orientation_probes
    assert south_rows.mean() > axes_mid


@pytest.mark.parametrize('function_name', ['draw_map'])
def test_the_imshow_kwargs_are_right_at_the_SOURCE(function_name):
    """The layer an edit would break. It cannot be read back off the artifact — cartopy reprojects the array and
    re-emits it with ``origin='lower'`` in projected metres — so the kwargs are checked by AST."""
    tree = ast.parse(inspect.getsource(maps))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == function_name)
    call = next(node for node in ast.walk(function)
                if isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'imshow')
    kwargs = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}

    assert kwargs['origin'] == "'upper'"
    assert kwargs['extent'] == 'GRID_EXTENT'
    assert kwargs['transform'] == 'data_crs'


def test_cartopy_reprojects_the_field(context):
    """Pinned so nobody "fixes" the source to ``origin='lower'`` to match what they see on the drawn object."""
    projection, data_crs = context
    cmap, norm = maps.make_lightning_cmap(6.0)
    figure = plt.figure(figsize=(4, 4))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
    image = maps.draw_map(ax, np.zeros((H, W)), 't', data_crs, cmap, norm)
    assert image.origin == 'lower'
    assert tuple(image.get_extent()) != maps.GRID_EXTENT
    plt.close(figure)


# =====================================================================================================================
# The framing decorations
# =====================================================================================================================
@pytest.fixture
def framed_axis(context):
    def build(**kwargs):
        projection, data_crs = context
        cmap, norm = maps.make_lightning_cmap(6.0)
        figure = plt.figure(figsize=(4, 4))
        ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
        maps.draw_map(ax, np.zeros((H, W)), 't', data_crs, cmap, norm, **kwargs)
        return figure, ax
    return build


def test_map_axes_use_equal_aspect(framed_axis):
    figure, ax = framed_axis()
    assert ax.get_aspect() in ('equal', 1.0)
    plt.close(figure)


def test_exactly_one_labelled_gridliner_is_drawn(framed_axis):
    figure, ax = framed_axis()
    gridliners = [artist for artist in ax.artists if isinstance(artist, Gridliner)]
    assert len(gridliners) == 1
    assert not gridliners[0].top_labels and not gridliners[0].right_labels
    plt.close(figure)


@pytest.mark.parametrize('left_labels', [True, False])
def test_left_labels_are_honoured_in_both_directions(left_labels, framed_axis):
    """Latitude labels only on the LEFTMOST panel, so a 3-wide grid stays legible without repeating one axis thrice."""
    figure, ax = framed_axis(left_labels=left_labels)
    gridliner = next(artist for artist in ax.artists if isinstance(artist, Gridliner))
    assert bool(gridliner.left_labels) == left_labels
    plt.close(figure)


def test_the_framing_helper_applies_all_five_decorations_on_a_BARE_axis(context):
    """``frame_map_axis`` is called directly here rather than through ``draw_map``, because it is a separable step and
    one caller uses it that way: ``reporting._residual_map_panel`` hand-rolls its own extent + coastlines + title and
    deliberately does NOT call this, so the residual maps carry no gridlines. Pinning what the helper does keeps that
    difference legible as a choice rather than reading as an oversight.
    """
    projection, data_crs = context
    figure = plt.figure(figsize=(4, 4))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)

    maps.frame_map_axis(ax, 'a title', left_labels=True, data_crs=data_crs)

    assert ax.get_title() == 'a title'
    assert ax.get_aspect() in ('equal', 1.0)
    gridliners = [artist for artist in ax.artists if isinstance(artist, Gridliner)]
    assert len(gridliners) == 1 and gridliners[0].left_labels
    assert any('Feature' in type(artist).__name__ for artist in ax.artists + list(ax.collections))
    plt.close(figure)


def test_coastlines_are_drawn_and_borders_are_not(framed_axis):
    """The resolved decision: the coast is the geographic anchor, and political boundaries are left out because their
    line density would compete with a field that is 99.93 % zero. Both halves pinned so neither drifts back."""
    figure, ax = framed_axis()
    assert any('Feature' in type(artist).__name__ for artist in ax.artists + list(ax.collections))
    plt.close(figure)

    source = inspect.getsource(maps)
    assert 'coastlines' in source
    assert 'BORDERS' not in source and 'add_feature' not in source


# =====================================================================================================================
# The colour system: unit bins in lightning-hours
# =====================================================================================================================
def test_the_bounded_target_gives_twenty_five_bands():
    """``[0, 0.5)`` white + ``[0.5, 1)`` grey + one band per whole hour 1..24 = 25 bands over 26 boundaries. The bounded
    0-24 target has a finite natural binning, so there is no capped-top-bin handling to get wrong."""
    cmap, norm = maps.make_lightning_cmap(24.0)
    assert len(norm.boundaries) == 26 and cmap.N == 25
    assert list(norm.boundaries) == [0.0, 0.5] + list(range(1, 25))


def test_the_sub_one_interval_is_split_white_then_grey():
    """Near-zero is visually separated from a genuine low count: ``[0, 0.5)`` reads as "no lightning" and ``[0.5, 1)`` as
    "a prediction that rounds down to none"."""
    cmap, norm = maps.make_lightning_cmap(24.0)
    grey = tuple(round(int('9E9E9E'[i:i + 2], 16) / 255.0, 4) for i in (0, 2, 4))

    assert tuple(np.round(cmap(norm(0.2))[:3], 6)) == (1.0, 1.0, 1.0)
    assert tuple(np.round(cmap(norm(0.7))[:3], 4)) == grey
    assert cmap(norm(0.49)) != cmap(norm(0.51))
    assert tuple(np.round(cmap(norm(1.0))[:3], 4)) != grey


def test_whole_hours_get_distinct_darkening_colours():
    cmap, norm = maps.make_lightning_cmap(24.0)
    assert cmap(norm(3.0)) != cmap(norm(4.0))
    assert sum(cmap(norm(2.0))[:3]) > sum(cmap(norm(23.0))[:3])


@pytest.mark.parametrize('degenerate', [0.0, 0.3, -5.0, 1.0])
def test_a_degenerate_maximum_floors_to_a_one_hour_axis(degenerate):
    """A day with no lightning at all would otherwise build an empty colour axis."""
    _, norm = maps.make_lightning_cmap(degenerate)
    assert list(norm.boundaries) == [0.0, 0.5, 1.0]


def test_a_fractional_maximum_rounds_UP_to_a_whole_hour():
    _, norm = maps.make_lightning_cmap(6.2)
    assert list(norm.boundaries) == [0.0, 0.5] + list(range(1, 8))


def test_the_LEGACY_cool_palette_still_lands_on_the_warm_ones_value_axis():
    """``_BASE_COLORS_COOL`` outlives the diff encoding that used it (Step 4 block 4e), and this is what makes keeping
    it worth anything: fed to ``make_lightning_cmap`` it still produces a cmap/norm on a value axis IDENTICAL to the
    warm pair's, which is the whole property a second palette needed. Retaining a ramp that had quietly stopped being
    interchangeable would be retaining a trap.
    """
    warm_cmap, warm_norm = maps.make_lightning_cmap(9.0)
    cool_cmap, cool_norm = maps.make_lightning_cmap(9.0, maps._BASE_COLORS_COOL)

    assert list(warm_norm.boundaries) == list(cool_norm.boundaries)
    assert warm_cmap.N == cool_cmap.N
    assert warm_cmap(warm_norm(8.0)) != cool_cmap(cool_norm(8.0)), 'the two ramps must remain distinguishable'
    for cmap, norm in ((warm_cmap, warm_norm), (cool_cmap, cool_norm)):
        assert tuple(np.round(cmap(norm(0.1))[:3], 6)) == (1.0, 1.0, 1.0), 'both start at white below 0.5 h'


@pytest.mark.source_invariant
def test_the_cool_palette_is_FLAGGED_as_legacy():
    """It is unused by anything that ships, so the only thing standing between "deliberately retained" and "dead code
    nobody dared delete" is the note next to it. If the flag goes, the constant should go with it."""
    source = inspect.getsource(maps)
    marker = source.index('_BASE_COLORS_COOL = [')
    preamble = source[:marker]
    assert 'LEGACY' in preamble[-900:], 'the cool ramp must carry its legacy note'


def test_the_diff_map_MACHINERY_is_gone():
    """The removal, pinned. The palette stays; the apparatus built on it does not, and half-restoring one of these
    would give a figure a second palette with no colorbar to read it by."""
    for name in ('draw_diff_map', 'add_shared_diff_colorbars', 'make_lightning_scales', 'LightningScales'):
        assert not hasattr(maps, name), f'{name} came back without its counterparts'


def test_there_is_no_log_colour_scale():
    """``LogNorm`` / ``colorbar_scale: log`` existed for the heavy-tailed count field and is pointless on 0-24."""
    assert 'LogNorm' not in inspect.getsource(maps)


def test_the_kilometre_per_pixel_constant_is_a_quarter_degree():
    assert abs(maps.KM_PER_PIXEL - 27.75) < 0.5


@pytest.mark.parametrize('value,expected', [
    ('#FFFFFF', (1.0, 1.0, 1.0)),
    ('#000000', (0.0, 0.0, 0.0)),
    ('FFFFFF', (1.0, 1.0, 1.0)),                     # the leading '#' is optional
    ('#9E9E9E', (158 / 255, 158 / 255, 158 / 255)),  # _GREY, the [0.5, 1) band
])
def test_hex_colours_convert_to_the_zero_to_one_rgb_matplotlib_wants(value, expected):
    """The ramps are written as hex strings for readability against the 02a spec, but ``ListedColormap`` needs 0-1
    floats. A conversion that returned 0-255 would silently clip every colour to white."""
    assert tuple(round(channel, 6) for channel in maps._hex_to_rgb(value)) == \
        tuple(round(channel, 6) for channel in expected)


def test_every_stop_of_both_ramps_converts_to_a_valid_rgb_triple():
    """The ramps are hand-written constants; a typo (five digits, a stray character) would raise here rather than
    somewhere inside a figure build where the try/except only warns."""
    for ramp in (maps._BASE_COLORS_WARM, maps._BASE_COLORS_COOL, [maps._GREY]):
        for color in ramp:
            rgb = maps._hex_to_rgb(color)
            assert len(rgb) == 3 and all(0.0 <= channel <= 1.0 for channel in rgb), color


def test_both_ramps_start_at_WHITE():
    """The zero end of the axis is the 99.93 % of cells with no lightning; anything else makes the whole map a colour
    field with the signal invisible inside it."""
    assert maps._hex_to_rgb(maps._BASE_COLORS_WARM[0]) == (1.0, 1.0, 1.0)
    assert maps._hex_to_rgb(maps._BASE_COLORS_COOL[0]) == (1.0, 1.0, 1.0)




# =====================================================================================================================
# The single shared colorbar
# =====================================================================================================================
def test_ONE_detached_colorbar_is_added():
    """Every panel shares one palette and one value axis, so one bar serves the figure. The pair it replaced was
    redundant in VALUE — both bars carried identical boundaries — and differed only in palette."""
    cmap, norm = maps.make_lightning_cmap(6.0)
    figure = plt.figure(figsize=(6, 4))
    before = len(figure.axes)

    maps.add_lightning_colorbar(figure, cmap, norm, 0.1, 0.8)

    assert len(figure.axes) - before == 1
    plt.close(figure)


def test_the_colorbar_is_labelled_in_HOURS_PER_DAY():
    """Not by error direction any more: the panel titles say which field is which, and the bar says what the numbers
    mean. `h / day` is the unit of the bounded 0-24 target."""
    cmap, norm = maps.make_lightning_cmap(6.0)
    figure = plt.figure(figsize=(6, 4))
    maps.add_lightning_colorbar(figure, cmap, norm, 0.1, 0.8)

    labels = [ax.get_ylabel() for ax in figure.axes]
    assert 'h / day' in labels
    assert not any('obs' in label for label in labels), 'the over/under labelling went with the diff encoding'
    plt.close(figure)


def test_the_colorbar_CLEARS_the_ensemble_std_bar():
    """⚠️ The overlap this position exists to fix. On the 2x3 ensemble layout the std panel has its own viridis bar at
    x=0.80 whose `h / day` label sits to its right; at the old x=0.85 the shared bar was drawn on top of that label.

    Asserted as a gap rather than a magic number, so moving either bar deliberately still passes and moving one back
    on top of the other does not.
    """
    std_bar_x, std_bar_width = 0.80, 0.016
    assert maps.LIGHTNING_COLORBAR_X >= std_bar_x + std_bar_width + 0.05, 'no room for the std bar\'s label'
    assert maps.LIGHTNING_COLORBAR_X + std_bar_width < 1.0, 'the bar and its ticks must stay on the canvas'


def test_the_colorbar_x_position_is_actually_USED_by_default():
    """Anti-vacuity for the test above: a constant nothing reads would pass it for ever."""
    cmap, norm = maps.make_lightning_cmap(6.0)
    figure = plt.figure(figsize=(6, 4))
    maps.add_lightning_colorbar(figure, cmap, norm, 0.1, 0.8)

    bar = figure.axes[-1]
    assert abs(bar.get_position().x0 - maps.LIGHTNING_COLORBAR_X) < 1e-9
    plt.close(figure)


def test_the_panel_title_is_actually_DRAWN_not_just_SET(context):
    """🐛 A title that exists in the object model and never renders.

    Without an explicit ``y``, matplotlib's automatic title placement walks the axes' children to clear the tick
    labels — and a cartopy ``Gridliner`` with ``draw_labels=True`` reports NON-FINITE extents, so the position comes out
    ``inf`` and nothing is drawn. ``ax.get_title()`` returns the string either way, which is why every panel-title test
    in ``reporting_test.py`` passed while the rendered figures carried no panel titles at all.

    So this asserts the title has a FINITE window extent and sits above the axes — the two things ``get_title()`` cannot
    tell you. Same cartopy behaviour that cropped every saved map figure.
    """
    projection, data_crs = context
    cmap, norm = maps.make_lightning_cmap(6.0)
    figure = plt.figure(figsize=(5, 4))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
    maps.draw_map(ax, np.zeros((H, W)), 'observations', data_crs, cmap, norm)

    figure.canvas.draw()
    box = ax.title.get_window_extent(figure.canvas.get_renderer())
    axes_box = ax.get_window_extent(figure.canvas.get_renderer())

    assert np.isfinite([box.y0, box.y1, box.x0, box.x1]).all(), \
        'non-finite title bbox: the gridliner poisoned the automatic placement again'
    assert box.y0 >= axes_box.y1 - 2, 'the title must sit above the map, not inside it'
    assert ax.title.get_text() == 'observations'
    plt.close(figure)


@pytest.mark.source_invariant
def test_the_title_y_is_passed_EXPLICITLY():
    """The one-line guard the test above would catch behaviourally, named here so a "tidy-up" that drops the argument
    is understood as removing a fix rather than a redundant kwarg."""
    source = inspect.getsource(maps.frame_map_axis)
    assert 'y=1.02' in source, 'set_title needs an explicit y or the title is silently not drawn'
