"""Fetch whatever Natural Earth data this repo's map figures actually need, so plotting works offline afterwards.

The repo already SHIPS what the current figures need — ``data/cartopy/shapefiles/natural_earth/physical/`` holds
``ne_50m_coastline.*`` (1.1 MB, ordinary git blobs, deliberately NOT LFS: see ``data/cartopy/README.md``), and
``src/__init__.py`` points ``CARTOPY_DATA_DIR`` at it. So on a normal clone this script has nothing to do and says so.

It exists for the case the bundle cannot cover: ``frame_map_axis`` calls ``ax.coastlines()`` with cartopy's default
``resolution='auto'``, which picks the shapefile from the AXES EXTENT. Change the projection or the domain and cartopy
may ask for ``10m`` or ``110m`` instead, which the bundle does not have — it then falls back to the per-user download
cache and reaches the network at PLOT time, on whatever machine is plotting. That is a compute node, often offline, and
the failure arrives at the end of a run rather than the start.

    python scripts/prewarm_cartopy.py            # report what the figures request and whether it is available
    python scripts/prewarm_cartopy.py --fetch    # download anything missing into the per-user cache (needs network)

⚠️ It discovers the requirement by RENDERING, not by listing filenames. A hardcoded list would drift from
``maps.py`` the moment the extent changed, which is the exact failure this script is here to catch.
"""
import os
import sys

# `src` first: the bootstrap sets CARTOPY_DATA_DIR, and cartopy reads it ONCE at import. Importing cartopy first would
# silently ignore the bundle -- see `use_bundled_cartopy_data`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src                                                                             # noqa: F401,E402

import matplotlib                                                                      # noqa: E402
matplotlib.use('Agg')

import cartopy                                                                         # noqa: E402
import cartopy.crs as ccrs                                                             # noqa: E402
from cartopy.io import shapereader                                                     # noqa: E402
from matplotlib import pyplot as plt                                                   # noqa: E402

from src.utils.io.environment import is_git_lfs_pointer                                 # noqa: E402
from src.utils.plotting.maps import add_map_axis, frame_map_axis                        # noqa: E402


def requested_datasets(projection=None):
    """Render one throwaway map through the repo's OWN framing and report every Natural Earth dataset it asks for.

    Args:
        projection: Cartopy CRS for the axes; defaults to ``EuroPP()``, which is what the figures use.

    Returns:
        Sorted list of ``(resolution, category, name)`` tuples, e.g. ``[('50m', 'physical', 'coastline')]``.
    """
    requested = []
    original = shapereader.natural_earth

    def record(resolution='110m', category='physical', name='coastline'):
        requested.append((resolution, category, name))
        return original(resolution=resolution, category=category, name=name)

    shapereader.natural_earth = record
    try:
        figure = plt.figure(figsize=(4, 4))
        axis = add_map_axis(figure, figure.add_gridspec(1, 1)[0, 0], projection or ccrs.EuroPP())
        frame_map_axis(axis, 'prewarm', left_labels=True, data_crs=ccrs.PlateCarree())
        figure.canvas.draw()                        # the features resolve lazily, at DRAW time
        plt.close(figure)
    finally:
        shapereader.natural_earth = original
    return sorted(set(requested))


def bundled_path(resolution, category, name):
    """Where a dataset would sit inside the repo bundle, following cartopy's own template."""
    directory = os.environ.get('CARTOPY_DATA_DIR')
    if not directory:
        return None
    return os.path.join(directory, 'shapefiles', 'natural_earth', category, f'ne_{resolution}_{name}.shp')


def unpulled_pointers():
    """Every shapefile in the bundle that is still a git-lfs pointer.

    ⚠️ Checked BEFORE anything renders, and that ordering is the point. Rendering reads the shapefile, so a pointer
    makes cartopy die inside its shapefile reader with `KeyError: 828781878` — the first four bytes of the word
    "version" read as a shape type. Diagnosing after the render means never diagnosing at all.
    """
    directory = os.environ.get('CARTOPY_DATA_DIR')
    if not directory:
        return []
    pointers = []
    for current, _directories, filenames in os.walk(directory):
        for filename in filenames:
            path = os.path.join(current, filename)
            if filename.endswith('.shp') and is_git_lfs_pointer(path):
                pointers.append(path)
    return sorted(pointers)


def main(fetch: bool = False) -> int:
    print(f'cartopy {cartopy.__version__}')
    print(f'bundle (CARTOPY_DATA_DIR) : {os.environ.get("CARTOPY_DATA_DIR") or "<unset — no bundle in this checkout>"}')
    print(f'download cache (data_dir) : {cartopy.config["data_dir"]}')
    print()

    pointers = unpulled_pointers()
    if pointers:
        print(f'{len(pointers)} bundled shapefile(s) are git-lfs POINTERS, not the data:')
        for path in pointers:
            print(f'  {path}')
        print('\nThe bundle is committed as ORDINARY blobs, so this should be impossible -- it means the\n'
              '`data/cartopy/shapefiles/** -filter` exemption was dropped from .gitattributes and something\n'
              'installed re-pointerised the file. Restore the exemption, then:\n'
              '    git lfs pull        # or check out the blob again\n'
              'Left as is, cartopy fails inside its shapefile reader (KeyError on a bogus shape type), which reads as '
              'corrupt data rather than a configuration mistake.')
        return 2

    datasets = requested_datasets()
    print(f'the map figures request {len(datasets)} dataset(s):')

    missing = []
    for resolution, category, name in datasets:
        bundle = bundled_path(resolution, category, name)
        if bundle and os.path.exists(bundle):
            if is_git_lfs_pointer(bundle):
                print(f'  ne_{resolution}_{name:20s} BUNDLED BUT AN LFS POINTER — run `git lfs pull`')
                missing.append((resolution, category, name))
            else:
                size = os.path.getsize(bundle)
                print(f'  ne_{resolution}_{name:20s} bundled ({size:,} bytes) — no network needed')
            continue
        print(f'  ne_{resolution}_{name:20s} NOT bundled — cartopy will use its download cache')
        missing.append((resolution, category, name))

    if not missing:
        print('\nNothing to do: every dataset the figures request is in the repo bundle.')
        return 0

    if not fetch:
        print(f'\n{len(missing)} dataset(s) would need the network at PLOT time. Re-run with --fetch on a machine that '
              f'has it, or add them to the bundle.')
        return 1

    print()
    for resolution, category, name in missing:
        path = shapereader.natural_earth(resolution=resolution, category=category, name=name)
        print(f'  fetched ne_{resolution}_{name} -> {path}')
    print('\nDone. Plotting on this machine no longer needs the network.')
    return 0


if __name__ == '__main__':
    from fire import Fire

    sys.exit(Fire(main))
