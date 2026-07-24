"""Matplotlib configuration and plotting utilities.

This module initialises matplotlib with the Inter font from the bundled data
directory and provides helper functions for displaying and persisting figures.
"""
import os

from typing import List, Union
from matplotlib import font_manager, pyplot as plt

from src import root_path

# Load Inter font from bundled data directory and apply globally.
font_dirs = [os.path.join(root_path, 'data/fonts/inter/')]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Inter'


markers = ['o', 's', 'v', 'P', 'h', 'X', '^', '<', '>']
linestyles = ['dashed', 'dashdot', 'dotted', 'solid']


def show_plot_and_save(
        *args: Union[str, None],
        fig: plt.Figure,
        output_dir: str,
        output_filename_pattern: str,
        show: bool = True,
        dpi: int = 300
) -> str:
    """Display and save a matplotlib figure to disk.

    Optionally displays the figure, then saves it to the specified output
    directory with a filename derived from the provided args. The figure
    is closed after saving to free memory.

    Args:
        *args: Positional strings to be joined with underscores and
            formatted into output_filename_pattern. Used to construct
            the final filename dynamically. "None" items in list are ignored
        fig: The matplotlib Figure object to display and save.
        output_dir: Directory path where the figure will be saved.
        output_filename_pattern: Format string for the output filename;
            should contain exactly one format placeholder (e.g.,
            'plot_{}.png'). The placeholder is filled with
            '_'.join(args).
        show: If True (default), display the figure before saving.
        dpi: Dots per inch for saved figure (default: 300).

    Returns:
        Absolute path to the saved figure file.
    """
    if show:
        plt.show()
    clean_args: List[str] = [a for a in args if a is not None]
    out_file = os.path.join(output_dir, output_filename_pattern.format('_'.join(clean_args)))
    fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_file
