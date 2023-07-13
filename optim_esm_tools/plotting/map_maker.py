# -*- coding: utf-8 -*-
import optim_esm_tools as oet
import xarray as xr
import numpy as np

import typing as ty
import collections
from warnings import warn

import matplotlib.pyplot as plt

from immutabledict import immutabledict
from .plot import default_variable_labels
from matplotlib.colors import LogNorm

# import xrft

from optim_esm_tools.analyze.globals import _SECONDS_TO_YEAR
from optim_esm_tools.analyze import tipping_criteria


class MapMaker(object):
    data_set: xr.Dataset
    labels = tuple('i ii iii iv v vi vii viii ix x'.split())
    kw: ty.Mapping
    contitions: ty.Mapping

    def __init__(
        self,
        data_set: xr.Dataset,
        normalizations: ty.Union[None, ty.Mapping, ty.Iterable] = None,
        **conditions,
    ):
        self.data_set = data_set
        self.set_kw()
        self.set_conditions(**conditions)
        if normalizations is not None:
            self.set_normalizations(normalizations)

    def set_kw(self):
        import cartopy.crs as ccrs

        self.kw = immutabledict(
            fig=dict(dpi=200, figsize=(14, 10)),
            title=dict(fontsize=12),
            gridspec=dict(hspace=0.3),
            cbar=dict(orientation='horizontal', extend='both'),
            plot=dict(transform=ccrs.PlateCarree()),
            subplot=dict(
                projection=ccrs.PlateCarree(
                    central_longitude=0.0,
                ),
            ),
        )

    def set_conditions(self, **condition_kwargs):
        conditions = [
            cls(**condition_kwargs)
            for cls in [
                tipping_criteria.StartEndDifference,
                tipping_criteria.StdDetrended,
                tipping_criteria.MaxJump,
                tipping_criteria.MaxDerivitive,
                tipping_criteria.MaxJumpAndStd,
            ]
        ]

        self.conditions = {
            label: condition for label, condition in zip(self.labels, conditions)
        }
        self.labels = tuple(self.conditions.keys())

    normalizations: ty.Optional[ty.Mapping] = None

    _cache: bool = False

    def get_normalizations(self, normalizations=None):
        normalizations_start = (
            normalizations.copy() if normalizations is not None else None
        )

        if normalizations is None and self.normalizations is not None:
            # once set, they should be retrievable
            return self.normalizations

        if normalizations is None:
            normalizations = {i: [None, None] for i in self.conditions.keys()}
        elif isinstance(normalizations, collections.abc.Mapping):
            normalizations = normalizations
        elif isinstance(normalizations, collections.abc.Iterable):
            normalizations = {
                i: normalizations[j] for j, i in enumerate(self.conditions.keys())
            }

        def _incorrect_format():
            return (
                any(
                    not isinstance(v, collections.abc.Iterable)
                    for v in normalizations.values()
                )
                or any(len(v) != 2 for v in normalizations.values())
                or any(k not in normalizations for k in self.conditions)
            )

        if normalizations is None or _incorrect_format():
            raise TypeError(
                f'Normalizations should be mapping from'
                f'{self.conditions.keys()} to vmin, vmax, '
                f'got {normalizations} (from {normalizations_start})'
            )
        return normalizations

    def set_normalizations(
        self,
        normalizations: ty.Union[None, ty.Mapping, ty.Iterable] = None,
    ):
        # run even if we don't set to check if there are no errors
        norm = self.get_normalizations(normalizations)
        if normalizations is not None:
            self.normalizations = norm

    def plot(self, *a, **kw):
        print('Depricated use plot_all')
        return self.plot_all(*a, **kw)

    def plot_selected(self, items=('ii', 'iii'), nx=None, fig=None, **_gkw):
        from matplotlib.gridspec import GridSpec

        if nx is None:
            nx = len(items) if len(items) <= 3 else 2

        ny = np.ceil(len(items) / nx).astype(int)

        if fig is None:
            kw = self.kw['fig'].copy()
            # Defaults are set for a 2x2 matrix
            kw['figsize'] = kw['figsize'][0] / (2 / nx), kw['figsize'][1] / (2 / ny)
            fig = plt.figure(**kw)

        gs = GridSpec(ny, nx, **self.kw['gridspec'])
        plt_axes = []

        i = 0
        for i, label in enumerate(items):
            ax = fig.add_subplot(gs[i], **self.kw['subplot'])
            self.plot_i(label, ax=ax, **_gkw)
            plt_axes.append(ax)
        return plt_axes

    @oet.utils.timed()
    def plot_all(self, nx=2, **kw):
        return self.plot_selected(nx=nx, items=self.conditions.keys(), **kw)

    @oet.utils.timed()
    def plot_i(self, label, ax=None, coastlines=True, **kw):
        if ax is None:
            ax = plt.gca()
        if coastlines:
            ax.coastlines()

        prop = getattr(self, label)

        cmap = plt.get_cmap('viridis').copy()
        cmap.set_extremes(under='cyan', over='orange')
        x_label = prop.attrs.get('name', label)
        c_kw = self.kw['cbar'].copy()
        c_kw.setdefault('label', x_label)
        normalizations = self.get_normalizations()
        c_range_kw = {
            vm: normalizations[label][j] for j, vm in enumerate('vmin vmax'.split())
        }

        for k, v in {
            **self.kw['plot'],
            **c_range_kw,
            **dict(
                cbar_kwargs=c_kw,
                cmap=cmap,
            ),
        }.items():
            kw.setdefault(k, v)

        plt_ax = prop.plot(**kw)

        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        description = self.conditions[label].long_description
        ax.set_title(label.upper() + '\n' + description, **self.kw['title'])
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        return plt_ax

    def __getattr__(self, item):
        if item in self.conditions:
            condition = self.conditions[item]
            return self.data_set[condition.short_description]
        return self.__getattribute__(item)

    def _ts(
        self,
        variable,
        ds=None,
        running_mean=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        running_mean = (
            running_mean or oet.config.config['analyze']['moving_average_years']
        )
        if ds is None:
            ds = self.data_set
        if not only_rm:
            plot_kw['label'] = labels.get(variable, variable)
            plot_simple(ds, variable, **plot_kw)

        plot_kw['label'] = labels.get(
            f'{variable}_run_mean_{running_mean}',
            f'{variable} running mean {running_mean}',
        )
        plot_simple(ds, f'{variable}_run_mean_{running_mean}', **plot_kw)
        plt.legend()

    def _det_ts(
        self,
        variable,
        ds=None,
        running_mean=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        running_mean = (
            running_mean or oet.config.config['analyze']['moving_average_years']
        )
        if ds is None:
            ds = self.data_set
        if not only_rm:
            plot_kw['label'] = labels.get(
                f'{variable}_detrend', f'detrended {variable}'
            )
            plot_simple(ds, variable, **plot_kw)

        plot_kw['label'] = labels.get(
            f'{variable}_detrend_run_mean_{running_mean}',
            f'detrended {variable} running mean {running_mean}',
        )
        plot_simple(ds, f'{variable}_run_mean_{running_mean}', **plot_kw)
        plt.legend()

    def _ddt_ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=None,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        running_mean = (
            running_mean or oet.config.config['analyze']['moving_average_years']
        )
        if ds is None:
            ds = self.data_set
        variable_rm = f'{variable}_run_mean_{running_mean}'

        da = ds[variable]
        da_rm = ds[variable_rm]

        if other_dim:
            da = da.mean(other_dim)
            da_rm = da_rm.mean(other_dim)
        if not only_rm:
            # Dropna should take care of any nones in the data-array
            dy_dt = da.dropna(time).differentiate(time)
            dy_dt *= _SECONDS_TO_YEAR

            label = labels.get(variable, variable)
            dy_dt.plot(label=label, **plot_kw)

        dy_dt_rm = da_rm.dropna(time).differentiate(time)
        dy_dt_rm *= _SECONDS_TO_YEAR
        label = f"{labels.get(variable_rm, f'{variable} running mean {running_mean}')}"
        dy_dt_rm.plot(label=label, **plot_kw)

        plt.ylim(dy_dt_rm.min() / 1.05, dy_dt_rm.max() * 1.05)
        plt.ylabel(
            f'$\partial \mathrm{{{self.variable_name(variable)}}} /\partial t$ [{self.unit(variable)}/yr]'
        )
        plt.legend()
        plt.title('')

    @oet.utils.timed()
    def time_series(
        self,
        variable=None,
        time='time',
        other_dim=None,
        running_mean=None,
        interval=True,
        axes=None,
        **kw,
    ):
        ds = self.data_set
        variable = variable or self.variable
        other_dim = (
            other_dim
            if other_dim is not None
            else oet.config.config['analyze']['lon_lat_dim'].split(',')
        )
        running_mean = (
            running_mean or oet.config.config['analyze']['moving_average_years']
        )
        if interval is False:
            ds = ds.copy().mean(other_dim)
            other_dim = None

        plot_kw = dict(**kw)

        if axes is None:
            _, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw=dict(hspace=0.3))

        plt.sca(axes[0])
        self._ts(
            variable, ds=ds, running_mean=running_mean, other_dim=other_dim, **plot_kw
        )

        plt.sca(axes[1])
        self._det_ts(
            variable, ds=ds, running_mean=running_mean, other_dim=other_dim, **plot_kw
        )

        plt.sca(axes[2])
        self._ddt_ts(
            variable,
            ds=ds,
            time=time,
            running_mean=running_mean,
            other_dim=other_dim,
            **plot_kw,
        )

        return axes

    @property
    def dataset(self):
        warn(f'Calling {self.__class__.__name__}.data_set not .dataset')
        return self.data_set

    @property
    def title(self):
        return make_title(self.data_set)

    @property
    def variable(self):
        return self.data_set.attrs['variable_id']

    def variable_name(self, variable):
        return default_variable_labels().get(
            variable,
            variable,  # self.data_set[variable].attrs.get('long_name', variable)
        )

    def unit(self, variable):
        if 'units' not in self.data_set[variable].attrs:
            oet.config.get_logger().warning(
                f'No units for {variable} in {self.data_set}'
            )
            # raise ValueError( self.data_set.attrs, self.data_set[variable].attrs, variable)
        return self.data_set[variable].attrs.get('units', f'?').replace('%', '\%')


class HistoricalMapMaker(MapMaker):
    def __init__(self, *args, ds_historical=None, **kwargs):
        if ds_historical is None:
            raise ValueError('Argument ds_historical is required')
        self.ds_historical = ds_historical
        super().__init__(*args, **kwargs)

    @staticmethod
    def calculate_ratio_and_max(da, da_historical):
        result = da / da_historical
        ret_array = result.values
        if len(ret_array) == 0:
            raise ValueError(
                f'Empty ret array, perhaps {da.shape} and {da_historical.shape} don\'t match?'
                f'\nGot\n{ret_array}\n{result}\n{da}\n{da_historical}'
            )
        max_val = np.nanmax(ret_array)
        mask_divide_by_zero = (da_historical == 0) & (da > 0)
        # Anything clearly larger than the max val
        ret_array[mask_divide_by_zero.values] = 10 * max_val
        result.data = ret_array
        return result, max_val

    def set_norm_for_item(self, item, max_val):
        current_norm = self.get_normalizations()
        low, high = current_norm.get(item, [None, None])
        if high is None:
            oet.config.get_logger().debug(f'Update max val for {item} to {max_val}')
            current_norm.update({item: [low, max_val]})
        self.set_normalizations(current_norm)

    @staticmethod
    def add_meta_to_da(result, name, short, long):
        name = '$\\frac{\\mathrm{scenario}}{\\mathrm{picontrol}}$' + f' of {name}'
        result = result.assign_attrs(
            dict(short_description=short, long_description=long, name=name)
        )
        result.name = name
        return result

    def get_compare(self, item):
        """Get the ratio of historical and the current data set"""
        condition = self.conditions[item]

        da = self.data_set[condition.short_description]
        da_historical = self.ds_historical[condition.short_description]

        result, max_val = self.calculate_ratio_and_max(da, da_historical)
        self.set_norm_for_item(item, max_val)

        result = self.add_meta_to_da(
            result, da.name, condition.short_description, condition.long_description
        )
        return result

    def __getattr__(self, item):
        if item in self.conditions:
            return self.get_compare(item)
        return self.__getattribute__(item)


def get_range(var):
    r = (
        dict(oet.config.config['variable_ranges'].items())
        .get(var, 'None,None')
        .split(',')
    )
    return [(float(l) if l != 'None' else None) for l in r]


def set_range(var):
    d, u = get_range(var)
    cd, cu = plt.ylim()
    plt.ylim(
        cd if d is None else min(cd, d),
        cu if u is None else max(cu, u),
    )


def get_unit(ds, var):
    return ds[var].attrs.get('units', '?').replace('%', '\%')


def plot_simple(ds, var, other_dim=None, show_std=False, std_kw=None, **kw):
    if other_dim is None:
        other_dim = set(ds[var].dims) - {'time'}
    mean = ds[var].mean(other_dim)
    l = mean.plot(**kw)
    if show_std:
        std_kw = std_kw or dict()
        for k, v in kw.items():
            std_kw.setdefault(k, v)
        std_kw.setdefault('alpha', 0.4)
        std = ds[var].mean(other_dim)
        (mean - std).plot(color=l[0]._color, **std_kw)
        (mean + std).plot(color=l[0]._color, **std_kw)

    set_range(var)
    plt.ylabel(
        f'{oet.plotting.plot.default_variable_labels().get(var, var)} [{get_unit(ds, var)}]'
    )
    plt.title('')


def summarize_mask(data_set, one_mask, plot_kw=None, other_dim=None, plot='v'):
    import cartopy.crs as ccrs

    ds_sel = oet.analyze.region_finding.mask_xr_ds(data_set.copy(), one_mask)
    plot_kw = plot_kw or dict()
    mm_sel = MapMaker(ds_sel)

    mosaic = 'a.\nb.'
    fig, axes = plt.subplot_mosaic(
        mosaic,
        figsize=(17, 6),
        gridspec_kw=dict(width_ratios=[1, 1], wspace=0.1, hspace=0.05),
    )
    axes['b'].sharex(axes['a'])

    other_dim = other_dim or oet.config.config['analyze']['lon_lat_dim'].split(',')

    plt.sca(axes['a'])
    var = mm_sel.variable
    plot_simple(ds_sel, var, **plot_kw)

    plt.sca(axes['b'])
    var = f'{mm_sel.variable}_detrend'
    plot_simple(ds_sel, var, **plot_kw)

    if plot is None:
        ds_dummy = ds_sel.copy()
        ds_dummy['cell_area'] /= 1e6

        ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        tot_area = ds_dummy['cell_area'].sum()

        ds_dummy['cell_area'].values[ds_dummy['cell_area'] > 0] = tot_area
        ds_dummy['cell_area'].plot(
            vmin=1,
            vmax=510100000,
            norm=LogNorm(),
            cbar_kwargs={
                **mm_sel.kw.get('cbar', {}),
                **dict(extend='neither', label='Sum of area [km$^2$]'),
            },
        )
        ax.coastlines()
        exp = int(np.log10(tot_area))
        plt.title(f'Area ${tot_area/(10**exp):.1f}\\times10^{exp}$ km$^2$')
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        mm_sel.plot_i(label=plot, ax=ax, coastlines=True)
    plt.suptitle(mm_sel.title, y=0.97)
    axes = list(axes.values()) + [ax]
    plt.setp(axes[0].get_xticklabels(), visible=False)
    return axes


def make_title(ds):
    return '{institution_id} {source_id} {experiment_id} {variant_label} {variable_id} {version}'.format(
        **ds.attrs
    )
