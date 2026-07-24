import os
from itertools import product
from typing import Union, Tuple, List

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

from src.utils.plotting import *

# definition of design libraries
ibm = {
    'aqua': '#31C9B0',
    'light_blue': '#648FFF',
    'purple': '#785EF0',
    'hot_pink': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
}
tol = {
    'indigo': '#332288',
    'green': '#117733',
    'aqua': '#44AA99',
    'lightblue': '#88CCEE',
    'sand': '#DDCC77',
    'rose': '#CC6677',
    'orchid': '#AA4499',
    'violetred': '#882255'
}


def ibm_linear_palette_factory(
        ibm_color_1: str,
        basecolor: Union[str, Tuple[float, ...]] = (1.0, 1.0, 1.0),
        nodes: Tuple[float, float] = (0.0, 1.0)
) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f'ibm_{ibm_color_1}s',
        list(zip(nodes, (basecolor, ibm[ibm_color_1])))
    )


def ibm_diverging_palette_factory(
        ibm_color_0: str,
        ibm_color_1: str,
        basecolor: Union[str, Tuple[float, ...]] = (1.0, 1.0, 1.0),
        nodes: Tuple[float, float] = (0.0, 0.5, 1.0)
) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f'ibm_{ibm_color_0}_to_{ibm_color_1}',
        list(zip(nodes, (ibm[ibm_color_0], basecolor, ibm[ibm_color_1])))
    )


ibm_colors = ['orange', 'purple', 'light_blue', 'hot_pink', 'aqua', 'yellow']
tol_colors = ['sand', 'lightblue', 'violetred', 'aqua', 'orchid', 'rose', 'green', 'indigo']


def get_color(color: Union[str, None]) -> Union[str, None]:
    if color in ibm_colors:
        return ibm[color]
    elif color in tol_colors:
        return tol[color]
    else:
        return color


ibm_linear_palettes = {
    f'ibm_{color}s': ibm_linear_palette_factory(color)
    for color in ibm_colors
}
ibm_diverging_palettes = {
    f'ibm_{color_a}_to_{color_b}': ibm_diverging_palette_factory(ibm_color_0=color_a, ibm_color_1=color_b)
    for color_a, color_b in product(ibm_colors, ibm_colors)
    if color_a != color_b
}

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=list(map(ibm.get, ibm_colors)) + list(map(tol.get, tol_colors)))
