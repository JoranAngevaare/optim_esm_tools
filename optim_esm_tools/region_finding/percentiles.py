from ._base import (
    RegionExtractor,
    _two_sigma_percent,
    apply_options,
    plt_show,
    _mask_cluster_type,
)
import optim_esm_tools as oet

from optim_esm_tools.plotting.map_maker import MapMaker
from optim_esm_tools.analyze.clustering import (
    build_cluster_mask,
    build_weighted_cluster,
)
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.plotting.plot import setup_map, _show
from optim_esm_tools.analyze.xarray_tools import mask_xr_ds
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import typing as ty
import immutabledict
import os


class Percentiles(RegionExtractor):
    @oet.utils.check_accepts(
        accepts=immutabledict.immutabledict(cluster_method=('weighted', 'masked'))
    )
    @apply_options
    def get_masks(self, cluster_method='masked') -> _mask_cluster_type:
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
    def plot_masks(self, masks_and_clusters: _mask_cluster_type, ax=None, legend=True):
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
        masks_and_clusters: _mask_cluster_type,
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

        ds_dummy['area_square'].plot(
            cbar_kwargs=mask_cbar_kw,
            vmin=0,
            extend='neither',
            transform=oet.plotting.plot.get_cartopy_transform(),
        )  # type: ignore
        plt.title('')
        if scatter_medians:
            cluster_kw = cluster_kw or {}
            for m_i, cluster in enumerate(clusters):
                lat, lon = np.median(cluster, axis=0)
                ax.scatter(
                    lon,
                    lat,
                    label=f'cluster {m_i}',
                    **cluster_kw,
                    transform=oet.plotting.plot.get_cartopy_transform(),
                )
            if legend:
                plt.legend(**oet.utils.legend_kw())
        plt.suptitle(f'Clusters {self.title}', y=0.97 if len(masks) < 4 else 0.99)
        return ax

    def _plot_basic_map(self) -> ty.List[plt.Axes]:
        mm = MapMaker(self.data_set)
        axes = mm.plot_selected(items=self.labels)
        plt.suptitle(self.title, y=0.95)
        return axes

    @plt_show
    @apply_options
    def plot_mask_time_series(
        self, masks_and_clusters: _mask_cluster_type, time_series_joined=True
    ):
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
        masks_and_clusters: _mask_cluster_type,
        time_series_joined=True,
        only_rm=None,
        axes=None,
        _ma_window=None,
    ):
        if only_rm is None:
            only_rm = bool((len(masks_and_clusters[0]) > 1 and time_series_joined))
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
