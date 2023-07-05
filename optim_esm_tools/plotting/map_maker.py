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

    @staticmethod
    def _ts_single(time_val, mean, std, plot_kw, fill_kw):
        if fill_kw is None:
            fill_kw = dict(alpha=0.4, step='mid')
        l = mean.plot(**plot_kw)

        if std is not None:
            # TODO, make this more elegant!
            # import cftime
            # plt.fill_between(   [cftime.real_datetime(dd.year, dd.month, dd.day) for dd in time_val], mean - std, mean+std, **fill_kw)

            (mean - std).plot(color=l[0]._color, alpha=0.4)
            (mean + std).plot(color=l[0]._color, alpha=0.4)

    def _ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=10,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        if ds is None:
            ds = self.data_set
        if not only_rm:
            mean, std = self._mean_and_std(ds, variable, other_dim)
            # return mean, std
            plot_kw['label'] = labels.get(variable, variable)
            self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        mean, std = self._mean_and_std(
            ds, f'{variable}_run_mean_{running_mean}', other_dim
        )
        plot_kw['label'] = labels.get(
            f'{variable}_run_mean_{running_mean}',
            f'{variable} running mean {running_mean}',
        )
        self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        plt.ylabel(f'{self.variable_name(variable)} [{self.unit(variable)}]')
        plt.legend()
        plt.title('')

    def _det_ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=10,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
        if ds is None:
            ds = self.data_set
        if not only_rm:
            mean, std = self._mean_and_std(ds, f'{variable}_detrend', other_dim)
            plot_kw['label'] = labels.get(
                f'{variable}_detrend', f'detrended {variable}'
            )
            self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        mean, std = self._mean_and_std(
            ds, f'{variable}_detrend_run_mean_{running_mean}', other_dim
        )
        plot_kw['label'] = labels.get(
            f'{variable}_detrend_run_mean_{running_mean}',
            f'detrended {variable} running mean {running_mean}',
        )
        self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)
        plt.ylabel(f'Detrend {self.variable_name(variable)} [{self.unit(variable)}]')
        plt.legend()
        plt.title('')

    def _ddt_ts(
        self,
        variable,
        ds=None,
        time='time',
        other_dim=(),
        running_mean=10,
        fill_kw=None,
        labels=dict(),
        only_rm=False,
        **plot_kw,
    ):
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
            # mean, std = self._mean_and_std(dy_dt, variable=None, other_dim=other_dim)
            # plot_kw['label'] = variable
            # self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)
            label = labels.get(variable, variable)
            dy_dt.plot(label=label, **plot_kw)

        dy_dt_rm = da_rm.dropna(time).differentiate(time)
        dy_dt_rm *= _SECONDS_TO_YEAR
        label = f"{labels.get(variable_rm, f'{variable} running mean {running_mean}')}"
        dy_dt_rm.plot(label=label, **plot_kw)
        # mean, std = self._mean_and_std(dy_dt_rm, variable=None, other_dim=other_dim)
        # plot_kw['label'] = variable
        # self._ts_single(ds[time].values, mean, std, plot_kw, fill_kw)

        plt.ylim(dy_dt_rm.min() / 1.05, dy_dt_rm.max() * 1.05)
        plt.ylabel(
            f'$\partial \mathrm{{{self.variable_name(variable)}}} /\partial t$ [{self.unit(variable)}/yr]'
        )
        plt.legend()
        plt.title('')

    @staticmethod
    def _mean_and_std(ds, variable, other_dim):
        if variable is None:
            da = ds
        else:
            da = ds[variable]
        if other_dim is None:
            return da, None
        return da.mean(other_dim), da.std(other_dim)

    @oet.utils.timed()
    def time_series(
        self,
        variable='tas',
        time='time',
        other_dim=('x', 'y'),
        running_mean=10,
        interval=True,
        axes=None,
        **kw,
    ):
        ds = self.data_set
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
    def ds(self):
        warn(
            f'Calling {self.__class__.__name__}.ds is depricated, use {self.__class__.__name__}.ds'
        )
        return self.data_set

    @property
    def dataset(self):
        warn(f'Calling {self.__class__.__name__}.data_set not .dataset')
        return self.data_set

    @property
    def title(self):
        return make_title(self.data_set)

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


def make_title(ds):
    return '{institution_id} {source_id} {experiment_id} {variant_label} {variable_id} {version}'.format(
        **ds.attrs
    )
