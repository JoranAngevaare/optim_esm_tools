# -*- coding: utf-8 -*-
import optim_esm_tools as oet
import xarray as xr
import numpy as np

import typing as ty
import collections
from warnings import warn

import matplotlib.pyplot as plt

from immutabledict import immutabledict
from .plot import *
from matplotlib.colors import LogNorm

from optim_esm_tools.analyze.globals import _SECONDS_TO_YEAR
from optim_esm_tools.analyze import tipping_criteria


class MapMaker(object):
    data_set: xr.Dataset
    labels = tuple('i ii iii iv v vi vii viii ix x'.split())
    kw: ty.Mapping
    contitions: ty.Mapping
    normalizations: ty.Optional[ty.Mapping] = None
    _cache: bool = False

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
        self.kw = immutabledict(
            fig=dict(dpi=200, figsize=(14, 10)),
            title=dict(fontsize=12),
            gridspec=dict(hspace=0.3),
            cbar=dict(orientation='horizontal', extend='both'),
            plot=dict(transform=get_cartopy_projection()),
            subplot=dict(projection=get_cartopy_projection()),
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
        running_mean=None,
        no_rm=False,
        only_rm=False,
        labels=None,
        **kw,
    ):
        variable = variable or self.variable
        rm = running_mean or oet.config.config['analyze']['moving_average_years']
        _vars = [variable, f'{variable}_run_mean_{rm}']
        assert no_rm + only_rm < 2, 'Cannot show nothing?!'

        if isinstance(labels, ty.Mapping):
            labels = [labels.get(k, k) for k in _vars]

        if only_rm or no_rm:
            keep_idx = slice(0, 1) if no_rm else slice(-1, None)
            _vars = _vars[keep_idx]
            if labels is not None:
                labels = labels[keep_idx]

        self._ts_for_vars(_vars, labels=labels, **kw)

    def _ts_for_vars(self, variables, labels=None, **plot_kw):
        variables = oet.utils.to_str_tuple(variables)
        labels = labels or [None] * len(variables)
        if not len(variables) == len(labels):
            raise ValueError(
                f'Inconsistent number of vars and labels {variables, labels}'
            )
        plot_kw.setdefault('ds', self.data_set)
        assert 'ds' in plot_kw
        for v, l in zip(variables, labels):
            l = l or default_variable_labels().get(v, v)
            plot_simple(var=v, label=l, **plot_kw)

    def _det_ts(
        self,
        variable,
        **plot_kw,
    ):
        self._ts(
            f'{variable}_detrend',
            **plot_kw,
        )

    def _ddt_ts(
        self,
        variable,
        ds=None,
        time='time',
        running_mean=None,
        **plot_kw,
    ):
        ds = (ds or self.data_set).copy()

        rm = running_mean or oet.config.config['analyze']['moving_average_years']
        var_rm = f'{variable}_run_mean_{rm}'

        for var in variable, var_rm:
            # Dropna should take care of any nones in the data-array
            da = ds[var]
            # non_time_dim = set(da.dims) - {'time'}
            # if non_time_dim:
            #     da = da.mean(non_time_dim)
            dy_dt = da.differentiate(time)
            dy_dt *= _SECONDS_TO_YEAR
            ds[f'dt_{var}'] = dy_dt

        plot_kw['ds'] = ds
        # need to extract labels as derivatives have no defaults
        labels = plot_kw.pop('labels', None) or default_variable_labels()
        self._ts(
            f'dt_{variable}',
            labels=[labels.get(var, var) for var in (variable, var_rm)],
            **plot_kw,
        )

        plt.ylim(ds[f'dt_{var_rm}'].min() / 1.05, ds[f'dt_{var_rm}'].max() * 1.05)
        ylab = f'$\partial \mathrm{{{self.variable_name(variable)}}} /\partial t$ [{self.unit(variable)}/yr]'
        plt.ylabel(ylab)
        plt.legend()
        plt.title('')

    @oet.utils.timed()
    def time_series(
        self,
        variable=None,
        other_dim=None,
        running_mean=None,
        interval=None,
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
        rm = running_mean or oet.config.config['analyze']['moving_average_years']
        if interval is not None:
            oet.config.get_logger().warning('Interval kwarg is replaced by show_std')
            kw['show_std'] = interval

        plot_kw = dict(variable=variable, ds=ds, running_mean=rm, **kw)
        if axes is None:
            _, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw=dict(hspace=0.3))

        plt.sca(axes[0])
        self._ts(**plot_kw)

        plt.sca(axes[1])
        self._det_ts(**plot_kw)

        plt.sca(axes[2])
        self._ddt_ts(
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
        return get_unit(self.data_set, variable)


class HistoricalMapMaker(MapMaker):
    def __init__(self, *args, ds_historical=None, **kwargs):
        if ds_historical is None:
            raise ValueError('Argument ds_historical is required')
        self.ds_historical = ds_historical
        super().__init__(*args, **kwargs)

    @staticmethod
    def calculate_ratio_and_max(da, da_historical):
        mask_divide_by_zero = (da_historical == 0) & (da > 0)
        denominator = da_historical.values
        denominator[denominator == 0] = np.inf
        result = da / denominator
        ret_array = result.values
        if len(ret_array) == 0:
            raise ValueError(
                f'Empty ret array, perhaps {da.shape} and {da_historical.shape} don\'t match?'
                f'\nGot\n{ret_array}\n{result}\n{da}\n{da_historical}'
            )
        max_val = np.nanmax(ret_array)

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
        std_kw.pop('label', None)
        std = ds[var].std(other_dim)
        (mean - std).plot(color=l[0]._color, **std_kw)
        (mean + std).plot(color=l[0]._color, **std_kw)

    set_y_lim_var(var)
    plt.ylabel(
        f'{oet.plotting.plot.default_variable_labels().get(var, var)} [{get_unit(ds, var)}]'
    )
    plt.title('')


@oet.utils.check_accepts(accepts=dict(plot=('i', 'ii', 'iii', 'iv', 'v', None)))
def summarize_mask(
    data_set, one_mask, plot_kw=None, other_dim=None, plot='v', fig_kw=None
):
    plot_kw = plot_kw or dict(show_std=True)
    other_dim = other_dim or oet.config.config['analyze']['lon_lat_dim'].split(',')
    fig_kw = fig_kw or dict(
        mosaic='a.\nb.',
        figsize=(17, 6),
        gridspec_kw=dict(width_ratios=[1, 1], wspace=0.1, hspace=0.05),
    )

    ds_sel = oet.analyze.region_finding.mask_xr_ds(data_set.copy(), one_mask)
    mm_sel = MapMaker(ds_sel)
    fig, axes = plt.subplot_mosaic(**fig_kw)

    axes['b'].sharex(axes['a'])

    plt.sca(axes['a'])
    var = mm_sel.variable
    plot_simple(ds_sel, var, **plot_kw)

    plt.sca(axes['b'])
    var = f'{mm_sel.variable}_detrend'
    plot_simple(ds_sel, var, **plot_kw)

    if plot is None:
        ds_dummy = ds_sel.copy()
        ds_dummy['cell_area'] /= 1e6

        ax = fig.add_subplot(1, 2, 2, projection=get_cartopy_projection())
        tot_area = float(ds_dummy['cell_area'].sum(skipna=True))

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
        exponent = int(np.log10(tot_area))

        plt.title(f'Area ${tot_area/(10**exponent):.1f}\\times10^{exponent}$ km$^2$')
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax = fig.add_subplot(1, 2, 2, projection=get_cartopy_projection())
        mm_sel.plot_i(label=plot, ax=ax, coastlines=True)
    plt.suptitle(mm_sel.title, y=0.97)
    axes = list(axes.values()) + [ax]
    # Remove labels of top left axis.
    plt.setp(axes[0].get_xticklabels(), visible=False)
    return axes


def make_title(ds):
    return '{institution_id} {source_id} {experiment_id} {variant_label} {variable_id} {version}'.format(
        **ds.attrs
    )
