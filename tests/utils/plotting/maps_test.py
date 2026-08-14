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
    scales = maps.make_lightning_scales(24.0)

    figure = plt.figure(figsize=(5, 5))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
    maps.draw_map(ax, field, f'row {row}', data_crs, scales.warm_cmap, scales.warm_norm)
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


@pytest.mark.parametrize('function_name', ['draw_map', 'draw_diff_map'])
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
    scales = maps.make_lightning_scales(6.0)
    figure = plt.figure(figsize=(4, 4))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
    image = maps.draw_map(ax, np.zeros((H, W)), 't', data_crs, scales.warm_cmap, scales.warm_norm)
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
        scales = maps.make_lightning_scales(6.0)
        figure = plt.figure(figsize=(4, 4))
        ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
        maps.draw_map(ax, np.zeros((H, W)), 't', data_crs, scales.warm_cmap, scales.warm_norm, **kwargs)
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


def test_the_warm_and_cool_palettes_share_an_identical_value_axis():
    """The comparability invariant the two diff colorbars rest on: both are built against the same ``max_val``, so the
    bars are directly readable against each other."""
    scales = maps.make_lightning_scales(9.0)
    assert list(scales.warm_norm.boundaries) == list(scales.cool_norm.boundaries)
    assert scales.warm_cmap.N == scales.cool_cmap.N
    assert scales.warm_cmap(scales.warm_norm(8.0)) != scales.cool_cmap(scales.cool_norm(8.0))
    for cmap, norm in ((scales.warm_cmap, scales.warm_norm), (scales.cool_cmap, scales.cool_norm)):
        assert tuple(np.round(cmap(norm(0.1))[:3], 6)) == (1.0, 1.0, 1.0)


def test_there_is_no_log_colour_scale():
    """``LogNorm`` / ``colorbar_scale: log`` existed for the heavy-tailed count field and is pointless on 0-24."""
    assert 'LogNorm' not in inspect.getsource(maps)


def test_the_kilometre_per_pixel_constant_is_a_quarter_degree():
    assert abs(maps.KM_PER_PIXEL - 27.75) < 0.5


# =====================================================================================================================
# draw_diff_map: over/under in one panel
# =====================================================================================================================
def _render_diff(prediction, observation, max_val, context):
    projection, data_crs = context
    figure = plt.figure(figsize=(5, 5))
    ax = maps.add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection)
    maps.draw_diff_map(ax, prediction, observation, 'diff', data_crs, maps.make_lightning_scales(max_val))
    n_layers = len(ax.get_images())
    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].astype(int)
    plt.close(figure)
    red, blue = buffer[:, :, 0], buffer[:, :, 2]
    return (red - blue > 40), (blue - red > 40), n_layers


def test_the_diff_map_draws_exactly_two_layers(context):
    observation = np.full((H, W), 4.0)
    _, _, n_layers = _render_diff(observation, observation, 4.0, context)
    assert n_layers == 2


def test_OVER_prediction_renders_WARM_and_UNDER_renders_COOL(context):
    """Tested behaviourally because the mask algebra cannot be read off the artifact: cartopy regrids the input, so
    ``get_array()`` returns the resampled image rather than the ``[101, 149]`` field.

    West of 7.5 E over-predicts by 2 h, east of it under-predicts by 2 h. Both halves sit inside the -5..20 E display
    window, so a swapped palette assignment flips which side is red.
    """
    observation = np.full((H, W), 4.0)
    prediction = np.where(np.arange(W)[None, :] < int((7.5 + 12) / 0.25), 6.0, 2.0) * np.ones((H, 1))
    warm, cool, _ = _render_diff(prediction, observation, 6.0, context)

    assert warm.sum() > 500 and cool.sum() > 500
    assert np.nonzero(warm)[1].mean() < np.nonzero(cool)[1].mean()
    assert not (warm & cool).any(), 'the two layers must not overlap'


def test_an_exactly_correct_forecast_renders_WARM(context):
    """The ``>=`` boundary. If ``pred == obs`` fell to the cool layer, a perfect forecast would read as
    under-prediction."""
    field = np.full((H, W), 3.0)
    warm, cool, _ = _render_diff(field, field, 3.0, context)
    assert warm.sum() > 500 and cool.sum() == 0


def test_the_two_masks_are_exact_complements_at_the_source():
    """``<`` and ``>=`` partition the grid, so every cell is painted by exactly one layer and neither hides the other."""
    source = inspect.getsource(maps.draw_diff_map)
    assert 'masked_where(prediction < observation, prediction)' in source
    assert 'masked_where(prediction >= observation, prediction)' in source
    assert source.count('masked_where') == 2


def test_both_layers_draw_the_PREDICTION():
    """The observation only decides which palette each cell is drawn in — drawing the observation in one layer would
    make the panel a comparison of two different fields."""
    source = inspect.getsource(maps.draw_diff_map)
    assert source.count(', prediction)') == 2


# =====================================================================================================================
# The shared colorbars
# =====================================================================================================================
def test_two_detached_diff_colorbars_are_added(context):
    projection, _ = context
    scales = maps.make_lightning_scales(6.0)
    figure = plt.figure(figsize=(6, 4))
    before = len(figure.axes)
    maps.add_shared_diff_colorbars(figure, scales, 0.1, 0.8)
    assert len(figure.axes) - before == 2
    plt.close(figure)


def test_the_colorbars_are_labelled_by_error_direction():
    source = inspect.getsource(maps.add_shared_diff_colorbars)
    assert 'obs' in source and 'h / day' in source
