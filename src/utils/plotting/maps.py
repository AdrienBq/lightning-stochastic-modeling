"""The lightning map grammar: projection, colour system and the diff-map encoding.

Implements the visual specification in ``.claude/plans/inventory-figures.md`` §1, whose design reference is
``notebooks/02a_visualize_val_event_diffusion.ipynb`` on the adrien-mc-dropout branch. The notebook itself is not
ported — this module is the non-interactive reproduction of its styling, which the report figures in
``src/utils/metrics/reporting.py`` draw with.

Three ideas carry the grammar:

1. **Unit bins in lightning-hours.** The daily target is an integer count of hours, so the colour axis is one band
   per whole hour rather than a continuous ramp. The sub-1 interval is split: ``[0, 0.5)`` is white and
   ``[0.5, 1)`` is grey, which separates "no lightning" from "a prediction that rounds down to none".

2. **Two structurally identical palettes.** A warm ramp (white -> yellow -> red) and a cool one
   (white -> blue -> navy), both built against the SAME ``max_val`` so their two ``BoundaryNorm``s span an
   identical range and their colorbars are directly comparable.

3. **The diff map.** One panel showing magnitude AND error direction at once, by drawing the prediction twice under
   complementary masks: warm where ``pred >= obs``, cool where ``pred < obs``. A single panel then answers both
   "how much lightning" and "over- or under-predicted", which two separate panels cannot.

The colour scale is observation-driven and PER DATE (``max_val = ceil(nanmax(obs))``), so every panel of one
figure shares a scale; panels from different days deliberately do not.
"""
import logging
from typing import NamedTuple, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

logger = logging.getLogger(__name__)

# The ERA5 domain, from metadata.json: 0.25 deg, 35-60 N / -12-25 E on a 101 x 149 grid, array row 0 = NORTH edge.
GRID_EXTENT = (-12.0, 25.0, 35.0, 60.0)              # (lon_min, lon_max, lat_min, lat_max) of the DATA
# The framing the maps are drawn at — a crop of the data domain onto the region that actually carries convective
# activity. Per inventory-figures.md §1. (The 02a notebook itself uses a lat range of 30-55, which puts 5 degrees of
# empty sea below the domain and drops 35-60's northern strip; the spec's 35-55 is the corrected version.)
DISPLAY_EXTENT = (-5.0, 20.0, 35.0, 55.0)

# 0.25 deg of latitude, for labelling spectral wavelengths in km rather than pixels
KM_PER_PIXEL = 27.75

_BASE_COLORS_WARM = [
    '#FFFFFF', '#FFF5A6', '#FFE37B', '#FACA57', '#F5AD37',
    '#F08C1E', '#E36C16', '#D24D17', '#B33117', '#992015',
]                                                    # white -> yellow -> red   (observed, and pred >= obs)
_BASE_COLORS_COOL = [
    '#FFFFFF', '#D6E6F5', '#A8CCE8', '#7FB0DB', '#5594CE',
    '#3576C0', '#1F5BA8', '#13428A', '#0A2E6B', '#06204D',
]                                                    # white -> blue -> navy    (pred < obs)
_GREY = '#9E9E9E'                                    # values in [0.5, 1) lightning-hours


def _hex_to_rgb(value: str) -> Tuple[float, ...]:
    value = value.lstrip('#')
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def make_lightning_cmap(max_val: float, base_colors: Sequence[str] = _BASE_COLORS_WARM):
    """Build the unit-binned lightning-hours colormap: white ``[0, 0.5)``, grey ``[0.5, 1)``, then one band per hour.

    Args:
        max_val: Upper end of the value axis, rounded up to a whole hour (``ceil``); at least 1. Pass
            ``nanmax(observation)`` so the scale is observation-driven and every panel of one figure shares it.
        base_colors: The 10-stop ramp to interpolate, ``_BASE_COLORS_WARM`` (default) or ``_BASE_COLORS_COOL``.

    Returns:
        Tuple ``(ListedColormap, BoundaryNorm)``. Because both palettes are interpolated onto the same number of
        intervals, a warm/cool pair built from one ``max_val`` shares an identical value axis.
    """
    max_val = max(int(np.ceil(float(max_val))), 1)
    n_intervals = max_val

    base_rgb = np.array([_hex_to_rgb(color) for color in base_colors])
    positions = np.linspace(0, len(base_colors) - 1, n_intervals)
    gradient = []
    for position in positions:
        low = int(np.floor(position))
        high = min(low + 1, len(base_colors) - 1)
        weight = position - low
        gradient.append(tuple((1 - weight) * base_rgb[low] + weight * base_rgb[high]))

    # split the first unit bin: [0, 0.5) stays white, [0.5, 1) is grey; the gradient covers values >= 1
    levels = [0.0, 0.5] + list(range(1, max_val + 1))
    colors = [gradient[0], _hex_to_rgb(_GREY)] + gradient[1:]

    cmap = ListedColormap(colors, 'lightnings')
    return cmap, BoundaryNorm(levels, cmap.N)


class LightningScales(NamedTuple):
    """The warm/cool colormap pair for one figure, both on the same value axis.

    Kept as one object because the pair is only meaningful together: the diff-map encoding and its two colorbars
    rely on the two ``BoundaryNorm``s spanning an identical range.
    """
    warm_cmap: ListedColormap
    warm_norm: BoundaryNorm
    cool_cmap: ListedColormap
    cool_norm: BoundaryNorm


def make_lightning_scales(max_val: float) -> LightningScales:
    """Both palettes for one figure, built against a single ``max_val`` (see :func:`make_lightning_cmap`)."""
    warm_cmap, warm_norm = make_lightning_cmap(max_val, _BASE_COLORS_WARM)
    cool_cmap, cool_norm = make_lightning_cmap(max_val, _BASE_COLORS_COOL)
    return LightningScales(warm_cmap, warm_norm, cool_cmap, cool_norm)


def geographic_context():
    """Return ``(axes projection, data transform)`` for the maps — cartopy ``EuroPP`` over ``PlateCarree`` data.

    Returns ``(None, None)`` when cartopy is unavailable, in which case the callers fall back to plain
    non-geographic axes so a missing optional dependency degrades the figures rather than losing the run.
    """
    try:
        import cartopy.crs as ccrs
    except Exception as error:                       # cartopy is an optional dependency
        logger.warning(f'cartopy unavailable ({error}); maps render without the CRS projection / basemap.')
        return None, None
    return ccrs.EuroPP(), ccrs.PlateCarree()


def add_map_axis(figure, spec, projection):
    """Add a map axis at a gridspec slot (a geographic GeoAxes when a projection is available)."""
    return figure.add_subplot(spec, projection=projection) if projection is not None \
        else figure.add_subplot(spec)


def frame_map_axis(ax, title: str, left_labels: bool, projection, data_crs) -> None:
    """Apply the 02a framing: display extent, coastlines, equal aspect and dashed labelled gridlines.

    Latitude labels are drawn only on the LEFTMOST panel of a row (``left_labels``); top and right labels are always
    off. That keeps a 3-wide panel grid legible without repeating the same axis three times.
    """
    ax.set_title(title, fontsize=11)
    if projection is None:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.set_extent(DISPLAY_EXTENT, crs=data_crs)
    ax.coastlines(linewidth=0.8)
    ax.set_aspect('equal')
    gridlines = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7, linewidth=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    if not left_labels:
        gridlines.left_labels = False


def draw_map(
        ax,
        data: np.ndarray,
        title: str,
        projection,
        data_crs,
        cmap,
        norm=None,
        left_labels: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
):
    """Draw one ``[H, W]`` field on a framed map axis. Returns the matplotlib image (for a colorbar).

    ``origin='upper'`` is not optional: array row 0 is the NORTHERN edge of the domain, so flipping it would
    silently mirror the field about the equator-ward axis.
    """
    if data_crs is not None:
        image = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                          origin='upper', transform=data_crs, extent=GRID_EXTENT)
    else:
        image = ax.imshow(data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, origin='upper')
    frame_map_axis(ax, title, left_labels, projection, data_crs)
    return image


def draw_diff_map(
        ax,
        prediction: np.ndarray,
        observation: np.ndarray,
        title: str,
        projection,
        data_crs,
        scales: LightningScales,
        left_labels: bool = False
) -> None:
    """Draw a prediction with the over/under encoding: warm where ``pred >= obs``, cool where ``pred < obs``.

    The prediction is drawn TWICE under complementary masks, both on the shared value axis, so one panel conveys
    magnitude and error direction simultaneously.
    """
    over = np.ma.masked_where(prediction < observation, prediction)          # pred >= obs -> warm
    under = np.ma.masked_where(prediction >= observation, prediction)        # pred <  obs -> cool
    for field, cmap, norm in (
            (over, scales.warm_cmap, scales.warm_norm),
            (under, scales.cool_cmap, scales.cool_norm),
    ):
        if data_crs is not None:
            ax.imshow(field, cmap=cmap, norm=norm, origin='upper', transform=data_crs, extent=GRID_EXTENT)
        else:
            ax.imshow(field, cmap=cmap, norm=norm, origin='upper')
    frame_map_axis(ax, title, left_labels, projection, data_crs)


def add_shared_diff_colorbars(figure, scales: LightningScales, bottom: float, height: float) -> None:
    """Two detached vertical colorbars on the right, one per diff palette, spanning the given figure rows."""
    from matplotlib.cm import ScalarMappable

    for x_position, cmap, norm, label in (
            (0.85, scales.warm_cmap, scales.warm_norm, 'pred ≥ obs  (h / day)'),
            (0.91, scales.cool_cmap, scales.cool_norm, 'pred < obs  (h / day)'),
    ):
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        figure.colorbar(mappable, cax=figure.add_axes([x_position, bottom, 0.016, height]), label=label)
