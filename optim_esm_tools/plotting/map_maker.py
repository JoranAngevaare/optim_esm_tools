import collections
import typing as ty
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from immutabledict import immutabledict
from matplotlib.colors import LogNorm

import optim_esm_tools as oet
from .plot import *
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.analyze.globals import _SECONDS_TO_YEAR


def plot_simple(
    ds,
    var,
    other_dim=None,
    show_std=False,
    std_kw=None,
    add_label=True,
    set_y_lim=True,
    **kw,
):
    if other_dim is None:
        other_dim = set(ds[var].dims) - {'time'}
    mean = ds[var].mean(other_dim)
    l = mean.plot(**kw)
    if show_std:
        std_kw = std_kw or {}
        for k, v in kw.items():
            std_kw.setdefault(k, v)
        std_kw.setdefault('alpha', 0.4)
        std_kw.pop('label', None)
        std = ds[var].std(other_dim)
        (mean - std).plot(color=l[0]._color, **std_kw)
        (mean + std).plot(color=l[0]._color, **std_kw)

    if set_y_lim:
        set_y_lim_var(var)
    if add_label:
        plt.ylabel(oet.plotting.plot.get_ylabel(ds, var))
    plt.title('')


def overlay_area_mask(ds_dummy, field='cell_area', ax=None):
    ax = ax or plt.gcf().add_subplot(
        1,
        2,
        2,
        projection=oet.plotting.plot.get_cartopy_projection(),
    )
    kw = dict(
        norm=LogNorm(),
        cbar_kwargs={
            **dict(orientation='horizontal', extend='both'),
            **dict(extend='neither', label='Sum of area [km$^2$]'),
        },
        transform=oet.plotting.plot.get_cartopy_transform(),
    )
    if field == 'cell_area':
        ds_dummy[field] /= 1e6
        tot_area = float(ds_dummy[field].sum(skipna=True))
        ds_dummy[field].values[ds_dummy[field] > 0] = tot_area
        kw.update(
            dict(
                vmin=1,
                vmax=510100000,
            ),  # type: ignore
        )
    ds_dummy[field].plot(**kw)
    ax.coastlines()
    if field == 'cell_area':
        exponent = int(np.log10(tot_area))  # type: ignore

        plt.title(f'Area ${tot_area/(10**exponent):.1f}\\times10^{{{exponent}}}$ km$^2$')  # type: ignore
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
