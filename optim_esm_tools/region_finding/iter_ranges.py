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
from optim_esm_tools.analyze import tipping_criteria
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.region_finding._base import _mask_cluster_type


class _ThresholdIterator(ABC):
    _tqmd: bool = False

    data_set: xr.Dataset

    _get_mask_function_and_kw: ty.Callable
    _force_continuity: ty.Callable
    mask_area: ty.Callable

    def _get_masks_masked(
        self,
        iterable_range: ty.Dict[str, ty.Iterable] = dict(percentiles=(99.5, 97.5, 90)),
        lon_lat_dim: ty.Tuple[str, str] = ('lon', 'lat'),
        _mask_method: str = 'not_specified',
        iter_mask_min_area: float = 1e12,
        iter_mask_max_area: float = 999e12,
        force_continuity: bool = False,
    ) -> _mask_cluster_type:
        """The function `_get_masks_masked` builds masks and clusters based on
        a given iterable range and a combination of masks, and returns the
        masks and clusters that meet certain size criteria.

        :param iterable_range: A dictionary that specifies the range of values for the iteration. The
        keys represent the name of the parameter being iterated over, and the values represent the
        iterable range of values for that parameter
        :type iterable_range: ty.Dict[str, ty.Iterable]
        :param lon_lat_dim: The `lon_lat_dim` parameter is a tuple that specifies the names of the
        longitude and latitude dimensions in your dataset. These dimensions are used to extract the
        corresponding coordinate values for building the mask
        :param _mask_method: The `_mask_method` parameter is a string that specifies the method to be
        used for building the combined mask. It is not specified in the code snippet provided, so its
        value is not known, defaults to not_specified (optional)
        :param iter_mask_min_area: The parameter `iter_mask_min_area` represents the minimum area
        threshold for a mask. It is used to filter out masks that have a size smaller than this
        threshold
        :param iter_mask_max_area: The parameter `iter_mask_max_area` represents the maximum area that a
        mask can have. Masks with an area greater than or equal to `iter_mask_max_area` will raise an
        error
        :return: two lists: `masks` and `clusters`.
        """
        already_seen = None
        masks, clusters = [], []

        _, filtered_kwargs = self._get_mask_function_and_kw(
            method=_mask_method,
            **iterable_range,
        )
        iter_key, iter_values = list(filtered_kwargs.items())[0]

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
                force_continuity=force_continuity,
            )
            for m, c in zip(these_masks, these_clusters):
                size = self.mask_area(m).sum()
                if size >= iter_mask_min_area and size < iter_mask_max_area:
                    masks.append(m)
                    clusters.append(c)
                    if already_seen is None:
                        already_seen = m.copy()
                    already_seen[m] = True
                elif size >= iter_mask_max_area:
                    raise ValueError(
                        f'Got {size/iter_mask_min_area:.1%} target size for {value}',
                    )

        pbar.close()
        return masks, clusters

    def _get_masks_weighted(self, *a, **kw):
        raise NotImplementedError


class IterProductPercentiles(_ThresholdIterator, ProductPercentiles):
    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(product_percentiles=(99.5, 97.5, 90)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
        iter_mask_max_area=999e12,
        force_continuity=False,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            iter_mask_max_area=iter_mask_max_area,
            force_continuity=force_continuity,
            _mask_method='product_rank_past_threshold',
        )


class IterLocalHistory(_ThresholdIterator, LocalHistory):
    @apply_options
    def get_masks(
        self,
        iterable_range=dict(n_times_historical=(6, 5, 4, 3)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
        iter_mask_max_area=999e12,
        force_continuity=False,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            iter_mask_max_area=iter_mask_max_area,
            force_continuity=force_continuity,
            _mask_method='all_pass_historical',
        )


class IterPercentiles(_ThresholdIterator, Percentiles):
    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(percentiles=(99.5, 97.5, 90)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
        iter_mask_max_area=999e12,
        force_continuity=False,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            iter_mask_max_area=iter_mask_max_area,
            force_continuity=force_continuity,
            _mask_method='all_pass_percentile',
        )


class IterStartEnd(_ThresholdIterator, ProductPercentiles):
    labels = ('i',)
    criteria: ty.Tuple = (tipping_criteria.StartEndDifference,)

    @apply_options
    def _get_masks_masked(
        self,
        iterable_range=dict(product_percentiles=np.linspace(99.9, 85, 41)),
        lon_lat_dim=('lon', 'lat'),
        iter_mask_min_area=1e12,
        iter_mask_max_area=999e12,
        force_continuity=False,
    ) -> _mask_cluster_type:
        return super()._get_masks_masked(
            iterable_range=iterable_range,
            lon_lat_dim=lon_lat_dim,
            iter_mask_min_area=iter_mask_min_area,
            iter_mask_max_area=iter_mask_max_area,
            force_continuity=force_continuity,
            _mask_method='product_rank_past_threshold',
        )


class IterSNR(IterStartEnd):
    labels = ('i',)
    criteria: ty.Tuple = (tipping_criteria.SNR,)


class IterStd(IterStartEnd):
    labels = ('ii',)
    criteria: ty.Tuple = (tipping_criteria.StdDetrended,)


class IterMJ(IterStartEnd):
    labels = ('iii',)
    criteria: ty.Tuple = (tipping_criteria.MaxJump,)
