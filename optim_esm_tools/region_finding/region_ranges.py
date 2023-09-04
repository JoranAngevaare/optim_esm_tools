from abc import ABC

import numpy as np
import xarray as xr

import optim_esm_tools as oet
from ._base import _mask_cluster_type
from ._base import apply_options
from .local_history import LocalHistory
from .percentiles import Percentiles
from .product_percentiles import ProductPercentiles
from optim_esm_tools.analyze.clustering import build_cluster_mask


class _ThresholdIterator(ABC):
    data_set: xr.Dataset

    def _get_masks_weighted(self, *a):
        raise NotImplementedError

    def _get_masks_masked(
        self,
        iterable_range=dict(percentiles=(85, 100, 7)),
        lon_lat_dim=('lon', 'lat'),
        _mask_method='not_specified',
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        already_seen = None
        masks, clusters = [], []
        iter_key, iter_values = list(iterable_range.items())[0]
        pbar = oet.utils.tqdm(np.linspace(*iter_values)[::-1])
        for value in pbar:
            pbar.desc = f'{iter_key} = {value:.1e}'
            pbar.display()
            all_mask = self._build_combined_mask(  # type: ignore
                method=_mask_method,
                **{iter_key: value},
            )

            if already_seen is not None:
                all_mask[already_seen] = False

            these_masks, these_clusters = build_cluster_mask(
                all_mask,
                lon_coord=self.data_set[lon_lat_dim[0]].values,
                lat_coord=self.data_set[lon_lat_dim[1]].values,
            )
            for m, c in zip(these_masks, these_clusters):
                if self.mask_area(m).sum() >= iter_mask_min_area:  # type: ignore
                    masks.append(m)
                    clusters.append(c)
                    if already_seen is None:
                        already_seen = m
                    already_seen[m] = True
        pbar.close()
        return masks, clusters


class IterProductPercentiles(_ThresholdIterator, ProductPercentiles):
    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(percentiles=(85, 100, 7)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            _mask_method='all_pass_percentile',
        )


class IterLocalHistory(_ThresholdIterator, LocalHistory):
    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(n_times_historical=(3, 6, 7)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            _mask_method='all_pass_historical',
        )


class IterPercentiles(_ThresholdIterator, Percentiles):
    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(percentiles=(85, 100, 7)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            _mask_method='all_pass_percentile',
        )
