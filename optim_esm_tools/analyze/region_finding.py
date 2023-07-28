import optim_esm_tools as oet
from optim_esm_tools.plotting.map_maker import MapMaker, HistoricalMapMaker
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.analyze.globals import _CMIP_HANDLER_VERSION
from optim_esm_tools.analyze.cmip_handler import read_ds
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.analyze.clustering import build_weighted_cluster
from optim_esm_tools.analyze.xarray_tools import mask_xr_ds, mask_to_reduced_dataset
from optim_esm_tools.plotting.plot import setup_map, _show

import numpy as np
import matplotlib.pyplot as plt
import logging
import xarray as xr

import typing as ty
from functools import wraps
import inspect
import matplotlib.pyplot as plt
import immutabledict
import os


# >>> import scipy
# >>> scipy.stats.norm.cdf(3)
# 0.9986501019683699
# >> scipy.stats.norm.cdf(2)
# 0.9772498680518208
_two_sigma_percent = 97.72498680518208


def plt_show(*a):
    """Wrapper to disable class methods to follow up with show"""

    def somedec_outer(fn):
        @wraps(fn)
        def plt_func(*args, **kwargs):
            res = fn(*args, **kwargs)
            self = args[0]
            _show(getattr(self, 'show', False))
            return res

        return plt_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


def apply_options(*a):
    """If a function takes any arguments in self.extra_opt, apply it to the method"""

    def somedec_outer(fn):
        @wraps(fn)
        def timed_func(*args, **kwargs):
            self = args[0]
            takes = inspect.signature(fn).parameters
            kwargs.update({k: v for k, v in self.extra_opt.items() if k in takes})
            res = fn(*args, **kwargs)
            return res

        return timed_func

    if a and isinstance(a[0], ty.Callable):
        # Decorator that isn't closed
        return somedec_outer(a[0])
    return somedec_outer


class RegionExtractor:
    _logger: logging.Logger = None
    labels: tuple = tuple('ii iii'.split())
    show: bool = True

    criteria = (tipping_criteria.StdDetrended, tipping_criteria.MaxJump)
    extra_opt = None

    def __init__(
        self,
        variable=None,
        path=None,
        data_set=None,
        save_kw=None,
        extra_opt=None,
        read_ds_kw=None,
    ) -> None:
        read_ds_kw = read_ds_kw or dict()
        if path is None and data_set:
            self.data_set = data_set
        elif path:
            self.data_set = read_ds(path, **read_ds_kw)
        else:
            raise ValueError('Both path and data_set are None?!')

        save_kw = save_kw or dict(
            save_in='./',
            file_types=(
                'png',
                'pdf',
            ),
            skip=False,
            sub_dir=None,
        )
        extra_opt = extra_opt or dict(show_basic=True)
        extra_opt.update(dict(read_ds_kw=read_ds_kw))
        self.extra_opt = extra_opt
        self.save_kw = save_kw
        self.variable = variable or self.data_set.attrs.get('variable_id', 'NO_VAR_ID!')

    @property
    def log(self):
        if self._logger is None:
            self._logger = oet.config.get_logger()
        return self._logger

    @apply_options
    def workflow(self, show_basic=True):
        if show_basic:
            self.plot_basic_map()
        masks_and_clusters = self.get_masks()
        masks_and_clusters = self.filter_masks_and_clusters(masks_and_clusters)

        self.plot_masks(masks_and_clusters)
        self.plot_mask_time_series(masks_and_clusters)

    @plt_show
    def plot_basic_map(self):
        self._plot_basic_map()
        self.save(f'{self.title_label}_global_map')

    def _plot_basic_map(self):
        raise NotImplementedError(f'{self.__class__.__name__} has no _plot_basic_map')

    def save(self, name):
        assert self.__class__.__name__ in name
        oet.utils.save_fig(name, **self.save_kw)

    @property
    def title(self):
        return MapMaker(self.data_set).title

    @property
    def title_label(self):
        return self.title.replace(' ', '_') + f'_{self.__class__.__name__}'

    def mask_area(self, mask):
        try:
            if mask is None or not np.sum(mask):
                return 0
        except Exception as e:
            print(mask)
            raise ValueError(
                mask,
            ) from e
        self.check_shape(mask)
        return self.data_set['cell_area'].values[mask]

    def check_shape(
        self, data: ty.Union[np.ndarray, xr.DataArray], compare_with='cell_area'
    ):
        shape_should_be = self.data_set[compare_with].shape
        if data.shape == shape_should_be:
            return
        error_message = f'Got {data.shape}, expected {shape_should_be}'
        if name := getattr(data, 'name', False):
            error_message = f'For {name}: ' + error_message
        if dims := getattr(data, 'dims', False):
            error_message = f'{error_message}. Dims are {dims}, expected'
        error_message += f'for {self.data_set[compare_with].dims}'
        raise ValueError(error_message)

    @apply_options
    def mask_is_large_enough(self, mask, min_area_sq=0):
        return self.mask_area(mask).sum() >= min_area_sq

    def filter_masks_and_clusters(self, masks_and_clusters):
        if not len(masks_and_clusters[0]):
            return [], []
        ret_m = []
        ret_c = []
        for m, c in zip(*masks_and_clusters):
            if self.mask_is_large_enough(m):
                ret_m.append(m)
                ret_c.append(c)

        self.log.info(f'Keeping {len(ret_m)}/{len(masks_and_clusters[0])} of masks')
        return ret_m, ret_c

    @plt_show
    def summarize_mask(self, mask, m_i):
        self._summarize_mask(mask)
        plt.suptitle(self.title + f' cluster {m_i}')
        self.save(f'{self.title_label}_cluster-{m_i}')

        self.store_mask(mask, m_i)

    @apply_options
    def store_mask(self, mask, m_i, store_masks=True):
        if not store_masks:
            return
        save_in = self.save_kw['save_in']
        store_in_dir = os.path.join(save_in, 'masks')
        os.makedirs(store_in_dir, exist_ok=True)
        ds_masked = mask_to_reduced_dataset(self.data_set, mask)
        ds_masked.to_netcdf(
            os.path.join(
                store_in_dir,
                f'{self.title_label}_cluster-{m_i}_v{_CMIP_HANDLER_VERSION}.nc',
            )
        )

    def _summarize_mask(self, mask, plot=None):
        axes = oet.plotting.map_maker.summarize_mask(self.data_set, mask, plot=plot)
        return axes

    def make_mask_figures(self, masks):
        for m_i, mask in enumerate(masks):
            if np.sum(mask) == 0:
                continue
            self.summarize_mask(mask, m_i)


class MaxRegion(RegionExtractor):
    def get_masks(self) -> dict:
        """Get mask for max of ii and iii and a box around that"""

        def _val(label):
            return self.data_set[label].values

        def _max(label):
            return _val(label)[~np.isnan(_val(label))].max()

        masks = [_val(label) == _max(label) for label in self._labels]
        return masks, [None for _ in range(len(masks))]

    @apply_options
    def filter_masks_and_clusters(self, masks_and_clusters, min_area_km_sq=0):
        """Wrap filter to work on dicts"""
        if min_area_km_sq:
            message = f'Calling {self.__class__.__name__}.filter_masks_and_clusters is nonsensical as masks are single grid cells'
            self.log.warning(message)
        return masks_and_clusters

    @plt_show
    def plot_masks(self, masks, ax=None, legend=True):
        masks = masks[0]
        self._plot_masks(masks=masks, ax=ax, legend=legend)
        self.save(f'{self.title_label}_map_maxes_{"-".join(self.labels)}')

        self.make_mask_figures(masks)

    @apply_options
    def _plot_masks(self, masks, ax=None, legend=True):
        points = {}
        for key, mask_2d in zip(self._labels, masks):
            points[key] = self._mask_to_coord(mask_2d)
        if ax is None:
            oet.plotting.plot.setup_map()
            ax = plt.gca()
        for i, (label, xy) in enumerate(zip(self.labels, points.values())):
            ax.scatter(*xy, marker='oxv^'[i], label=f'Maximum {label}')
        if legend:
            ax.legend(**oet.utils.legend_kw())
        plt.suptitle(self.title, y=0.95)
        plt.ylim(-90, 90)
        plt.xlim(-180, 180)

    def _mask_to_coord(self, mask_2d):
        arg_mask = np.argwhere(mask_2d)[0]
        x = self.data_set.lon[arg_mask[1]]
        y = self.data_set.lat[arg_mask[0]]
        return x, y

    def _plot_basic_map(self):
        mm = MapMaker(self.data_set)
        axes = mm.plot_selected(items=self.labels)
        masks = self.get_masks()
        for ax in axes:
            self._plot_masks(masks[0], ax=ax, legend=False)
        plt.suptitle(self.title, y=0.95)

    @plt_show
    @apply_options
    def plot_mask_time_series(self, masks, time_series_joined=True):
        res = self._plot_mask_time_series(masks, time_series_joined=time_series_joined)
        if time_series_joined:
            self.save(f'{self.title_label}_time_series_maxes_{"-".join(self.labels)}')
        return res

    @apply_options
    def _plot_mask_time_series(
        self,
        masks_and_clusters,
        time_series_joined=True,
        only_rm=False,
        axes=None,
        _ma_window=None,
    ):
        _ma_window = _ma_window or oet.config.config['analyze']['moving_average_years']
        masks = masks_and_clusters[0]
        legend_kw = oet.utils.legend_kw(
            loc='upper left', bbox_to_anchor=None, mode=None, ncol=2
        )
        for label, mask_2d in zip(self._labels, masks):
            x, y = self._mask_to_coord(mask_2d)
            plot_labels = {
                f'{self.variable}': f'{label} at {x:.1f}:{y:.1f}',
                f'{self.variable}_detrend': f'{label} at {x:.1f}:{y:.1f}',
                f'{self.variable}_detrend_run_mean_{_ma_window}': f'$RM_{{{_ma_window}}}$ {label} at {x:.1f}:{y:.1f}',
                f'{self.variable}_run_mean_{_ma_window}': f'$RM_{{{_ma_window}}}$ {label} at {x:.1f}:{y:.1f}',
            }
            argwhere = np.argwhere(mask_2d)[0]
            ds_sel = self.data_set.isel(lat=argwhere[0], lon=argwhere[1])
            mm_sel = MapMaker(ds_sel)
            axes = mm_sel.time_series(
                variable=self.variable,
                other_dim=(),
                show_std=False,
                labels=plot_labels,
                axes=axes,
                only_rm=only_rm,
            )
            if time_series_joined is False:
                axes = None
                plt.suptitle(f'Max. {label} {self.title}', y=0.95)
                self.save(f'{self.title_label}_time_series_max_{label}')
                _show(self.show)
        if not time_series_joined:
            return

        for ax in axes:
            ax.legend(**legend_kw)
        plt.suptitle(f'Max. {"-".join(self.labels)} {self.title}', y=0.95)

    @property
    def _labels(self):
        return [crit.short_description for crit in self.criteria]


class Percentiles(RegionExtractor):
    @oet.utils.check_accepts(
        accepts=immutabledict.immutabledict(cluster_method=('weighted', 'masked'))
    )
    @apply_options
    def get_masks(self, cluster_method='masked') -> dict:
        if cluster_method == 'weighted':
            masks, clusters = self._get_masks_weighted()
        else:
            masks, clusters = self._get_masks_masked()
        if len(masks):
            self.check_shape(masks[0])
        return masks, clusters

    @apply_options
    def _get_masks_weighted(
        self,
        min_weight=0.95,
        lon_lat_dim=('lon', 'lat'),
    ):
        labels = [crit.short_description for crit in self.criteria]

        sums = []
        for lab in labels:
            vals = self.data_set[lab].values
            vals = tipping_criteria.rank2d(vals)
            vals[np.isnan(vals)] = 0
            sums.append(vals)

        tot_sum = np.zeros_like(sums[0])
        for s in sums:
            tot_sum += s
        tot_sum /= len(sums)

        self.check_shape(tot_sum)

        masks, clusters = build_weighted_cluster(
            weights=tot_sum,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
            threshold=min_weight,
        )
        return masks, clusters

    @apply_options
    def _get_masks_masked(
        self,
        percentiles=_two_sigma_percent,
        lon_lat_dim=('lon', 'lat'),
    ):
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        for lab in labels:
            arr = self.data_set[lab].values
            arr_no_nan = arr[~np.isnan(arr)]
            thr = np.percentile(arr_no_nan, percentiles)
            masks.append(arr >= thr)

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m

        self.check_shape(all_mask)

        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters

    @plt_show
    def plot_masks(self, masks_and_clusters, ax=None, legend=True):
        if not len(masks_and_clusters[0]):
            self.log.warning('No clusters found!')
            return
        self._plot_masks(
            masks_and_clusters=masks_and_clusters,
            ax=ax,
            legend=legend,
        )
        self.save(f'{self.title_label}_map_clusters_{"-".join(self.labels)}')

        self.make_mask_figures(masks_and_clusters[0])

    @apply_options
    def _plot_masks(
        self,
        masks_and_clusters,
        scatter_medians=True,
        ax=None,
        legend=True,
        mask_cbar_kw=None,
        cluster_kw=None,
    ):
        ds_dummy = self.data_set.copy()
        masks, clusters = masks_and_clusters
        all_masks = np.zeros(ds_dummy['cell_area'].shape, np.float64)
        all_masks[:] = np.nan
        for m, _ in zip(masks, clusters):
            self.check_shape(m)
            all_masks[m] = self.mask_area(m).sum()

        if ax is None:
            setup_map()
            ax = plt.gca()

        # Remember, cell_area is in m^2, hence the 1e6 below. Should we get this from the attrs?
        if mask_cbar_kw is None:
            mask_cbar_kw = dict(extend='neither', label='Area per cluster [km$^2$]')
        mask_cbar_kw.setdefault('orientation', 'horizontal')

        ds_dummy['area_square'] = (ds_dummy['cell_area'].dims, all_masks / 1e6)

        ds_dummy['area_square'].plot(cbar_kwargs=mask_cbar_kw, vmin=0, extend='neither')
        plt.title('')
        if scatter_medians:
            if cluster_kw is None:
                cluster_kw = dict()
            for m_i, cluster in enumerate(clusters):
                lat, lon = np.median(cluster, axis=0)
                ax.scatter(lon, lat, label=f'cluster {m_i}', **cluster_kw)
            if legend:
                plt.legend(**oet.utils.legend_kw())
        plt.suptitle(f'Clusters {self.title}', y=0.97 if len(masks) < 4 else 0.99)
        return ax

    def _plot_basic_map(self):
        mm = MapMaker(self.data_set)
        axes = mm.plot_selected(items=self.labels)
        plt.suptitle(self.title, y=0.95)
        return axes

        # Could add some masked selection on top

    #         masks, _ = self.get_masks()

    #         all_masks = masks[0]
    #         for m in masks[1:]:
    #             all_masks &= m
    #         ds_masked = mask_xr_ds(self.data_set.copy(), all_masks)
    #         mm_sel = MapMaker(ds_masked)
    #         for label, ax in zip(mm.labels, axes):
    #             plt.sca(ax)
    #             mm_sel.plot_i(label, ax=ax, coastlines=False)

    @plt_show
    @apply_options
    def plot_mask_time_series(self, masks_and_clusters, time_series_joined=True):
        if not len(masks_and_clusters[0]):
            self.log.warning('No clusters found!')
            return
        res = self._plot_mask_time_series(
            masks_and_clusters, time_series_joined=time_series_joined
        )
        if time_series_joined and masks_and_clusters:
            self.save(f'{self.title_label}_time_series_all_clusters')
        return res

    @apply_options
    def _plot_mask_time_series(
        self,
        masks_and_clusters,
        time_series_joined=True,
        only_rm=None,
        axes=None,
        _ma_window=None,
    ):
        if only_rm is None:
            only_rm = (
                True
                if (len(masks_and_clusters[0]) > 1 and time_series_joined)
                else False
            )
        _ma_window = _ma_window or oet.config.config['analyze']['moving_average_years']
        masks, clusters = masks_and_clusters
        legend_kw = oet.utils.legend_kw(
            loc='upper left', bbox_to_anchor=None, mode=None, ncol=4
        )
        for m_i, (mask, cluster) in enumerate(zip(masks, clusters)):
            lat, lon = np.median(cluster, axis=0)
            plot_labels = {
                f'{self.variable}': f'Cluster {m_i} near ~{lat:.1f}:{lon:.1f}',
                f'{self.variable}_detrend': f'Cluster {m_i} near ~{lat:.1f}:{lon:.1f}',
                f'{self.variable}_detrend_run_mean_{_ma_window}': f'Cluster {m_i} $RM_{{{_ma_window}}}$ near ~{lat:.1f}:{lon:.1f}',
                f'{self.variable}_run_mean_{_ma_window}': f'Cluster {m_i} $RM_{{{_ma_window}}}$ near ~{lat:.1f}:{lon:.1f}',
            }
            ds_sel = mask_xr_ds(self.data_set.copy(), mask)
            mm_sel = MapMaker(ds_sel)
            axes = mm_sel.time_series(
                variable=self.variable,
                show_std=True,
                labels=plot_labels,
                axes=axes,
                only_rm=only_rm,
            )
            if time_series_joined == False:
                axes = None
                plt.suptitle(f'Cluster. {m_i} {self.title}', y=0.95)
                self.save(f'{self.title_label}_time_series_cluster_{m_i}')
                _show(self.show)
        if not time_series_joined:
            return

        if axes is not None:
            for ax in axes:
                ax.legend(**legend_kw)
        plt.suptitle(f'Clusters {self.title}', y=0.95)


class PercentilesHistory(Percentiles):
    @apply_options
    def get_masks(
        self,
        percentiles_historical=_two_sigma_percent,
        read_ds_kw=None,
        lon_lat_dim=('lon', 'lat'),
    ) -> dict:
        read_ds_kw = read_ds_kw or dict()
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)

        historical_ds = self.get_historical_ds(read_ds_kw=read_ds_kw)
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        for lab in labels:
            arr = self.data_set[lab].values
            arr_historical = historical_ds[lab].values
            thr = np.percentile(
                arr_historical[~np.isnan(arr_historical)], percentiles_historical
            )
            masks.append(arr >= thr)

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m

        self.check_shape(all_mask)
        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters

    @apply_options
    def find_historical(
        self,
        match_to='piControl',
        look_back_extra=0,
        query_updates=None,
        search_kw=None,
    ):
        path = self.data_set.attrs['path']
        return oet.analyze.find_matches.associate_historical(
            path=path,
            data_set=None,
            match_to=match_to,
            look_back_extra=look_back_extra,
            query_updates=query_updates,
            search_kw=search_kw,
        )

    @apply_options
    def get_historical_ds(self, read_ds_kw=None, **kw):
        read_ds_kw = read_ds_kw or dict()
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)
        historical_path = self.find_historical(**kw)[0]
        return read_ds(historical_path, **read_ds_kw)


class ProductPercentiles(Percentiles):
    labels = ('ii', 'iii', 'v')

    @oet.utils.check_accepts(
        accepts=immutabledict.immutabledict(cluster_method=('weighted', 'masked'))
    )
    @apply_options
    def get_masks(self, cluster_method='masked') -> dict:
        """Get mask for max of ii and iii and a box around that"""
        if cluster_method == 'weighted':
            masks, clusters = self._get_masks_weighted()
        else:
            masks, clusters = self._get_masks_masked()
        if len(masks):
            self.check_shape(masks[0])
        return masks, clusters

    @apply_options
    def _get_masks_weighted(self, min_weight=0.95, lon_lat_dim=('lon', 'lat')):
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        ds = self.data_set.copy()
        combined_score = np.ones_like(ds[labels[0]].values)

        for label in labels:
            combined_score *= tipping_criteria.rank2d(ds[label].values)
        self.check_shape(combined_score)
        masks, clusters = build_weighted_cluster(
            weights=combined_score,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
            threshold=min_weight,
        )
        return masks, clusters

    @apply_options
    def _get_masks_masked(
        self, product_percentiles=_two_sigma_percent, lon_lat_dim=('lon', 'lat')
    ) -> dict:
        """Get mask for max of ii and iii and a box around that"""
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        ds = self.data_set.copy()
        combined_score = np.ones_like(ds[labels[0]].values)
        for label in labels:
            combined_score *= tipping_criteria.rank2d(ds[label].values)

        # Combined score is fraction, not percent!
        all_mask = combined_score > (product_percentiles / 100)

        self.check_shape(all_mask)

        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters


class LocalHistory(PercentilesHistory):
    @apply_options
    def get_masks(
        self, n_times_historical=4, read_ds_kw=None, lon_lat_dim=('lon', 'lat')
    ) -> dict:
        read_ds_kw = read_ds_kw or dict()
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)

        historical_ds = self.get_historical_ds(read_ds_kw=read_ds_kw)
        labels = [crit.short_description for crit in self.criteria]
        masks = []

        for lab in labels:
            arr = self.data_set[lab].values
            arr_historical = historical_ds[lab].values

            # If arr_historical is 0, the division is going to get a nan assigned,
            # despite this being the most interesting region (no historical
            # changes, only in the scenario's)!
            mask_no_std = arr_historical == 0
            mask_divide = np.zeros_like(mask_no_std)
            mask_divide[~mask_no_std] = (
                arr[~mask_no_std] / arr_historical[~mask_no_std] > n_times_historical
            )
            masks.append(mask_divide | (mask_no_std & (arr != 0)))

        all_mask = np.ones_like(masks[0])
        for m in masks:
            all_mask &= m

        self.check_shape(all_mask)

        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters

    @apply_options
    def _plot_basic_map(self, historical_normalizations=None, read_ds_kw=None):
        read_ds_kw = read_ds_kw or dict()
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)
        ds_historical = self.get_historical_ds(read_ds_kw=read_ds_kw)

        mm = HistoricalMapMaker(
            self.data_set,
            ds_historical=ds_historical,
            normalizations=historical_normalizations,
        )
        mm.plot_selected()
        plt.suptitle(self.title, y=0.95)
        return mm
