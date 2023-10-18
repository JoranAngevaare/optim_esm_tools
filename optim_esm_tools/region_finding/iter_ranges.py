import typing as ty
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
from optim_esm_tools.region_finding._base import _mask_cluster_type


class _ThresholdIterator:
    data_set: xr.Dataset
    _tqmd: bool = False

    def _get_masks_weighted(self, *a):
        raise NotImplementedError

    def _get_masks_masked(
        self,
        iterable_range: ty.Dict[str, ty.Iterable] = dict(percentiles=(99.5, 97.5, 90)),
        lon_lat_dim=('lon', 'lat'),
        _mask_method='not_specified',
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        """The function `_get_masks_masked` builds masks and clusters based on
        specified parameters and returns them.

        :param iterable_range: The `iterable_range` parameter is a dictionary that specifies the range
        of values for which the function will iterate. It has a single key-value pair, where the key
        represents the name of the parameter being iterated over, and the value is an iterable (such as
        a list or numpy array)
        :type iterable_range: ty.Dict[str, ty.Iterable]
        :param lon_lat_dim: The `lon_lat_dim` parameter is a tuple that specifies the names of the
        longitude and latitude dimensions in the dataset. These dimensions are used to extract the
        corresponding coordinate values for building the cluster mask
        :param _mask_method: The `_mask_method` parameter is used to specify the method for building the
        combined mask. It is currently set to `'not_specified'`, which means that the method is not
        specified and needs to be provided, defaults to not_specified (optional)
        :param iter_mask_min_area: The parameter `iter_mask_min_area` represents the minimum area that a
        mask must have in order to be considered. Masks with an area less than `iter_mask_min_area` will
        be excluded from the final result
        :return: two lists: `masks` and `clusters`.
        """
        already_seen = None
        masks, clusters = [], []
        iter_key, iter_values = list(iterable_range.items())[0]
        pbar = oet.utils.tqdm(iter_values, disable=not self._tqmd)
        for value in pbar:
            pbar.desc = f'{iter_key} = {value:.3g}'

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
                        already_seen = m.copy()
                    already_seen[m] = True
        pbar.close()
        return masks, clusters


class IterProductPercentiles(_ThresholdIterator, ProductPercentiles):
    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(product_percentiles=(99.5, 97.5, 90)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            _mask_method='product_rank_past_threshold',
        )


class IterLocalHistory(_ThresholdIterator, LocalHistory):
    @apply_options
    def get_masks(
        self,
        iterable_range=dict(n_times_historical=(6, 5, 4, 3)),
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
        iterable_range=dict(percentiles=(99.5, 97.5, 90)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            _mask_method='all_pass_percentile',
        )
