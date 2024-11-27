import itertools
import typing as ty

import numpy as np
import regionmask

from ._base import _mask_cluster_type
from ._base import RegionExtractor
from ._base import apply_options


class _NamedRegions(RegionExtractor):
    region_database = regionmask.defined_regions.ar6.all

    _default_regions: ty.Tuple[str, ...]

    @apply_options
    def get_masks(
        self,
        select_regions: ty.Optional[ty.Tuple[str, ...]] = None,
    ) -> _mask_cluster_type:
        if select_regions is None:
            select_regions = self._default_regions
        mask_2d = ~self.data_set[self.variable].isnull().all(dim='time')
        region_map = self.region_database.mask(self.data_set.lon, self.data_set.lat)
        mask_values = mask_2d.values

        masks = []
        for i, b in enumerate(self.region_database.names):
            if b not in select_regions:
                continue

            masks.append((region_map == i).values & mask_values)

        coords = []

        for m in masks:
            # Format to lat,lon instead of lon,lat
            coords.append(np.array(self.mask_to_lon_lat(m)))
        return masks, coords

    def filter_masks_and_clusters(
        self,
        masks_and_clusters: _mask_cluster_type,
    ) -> _mask_cluster_type:
        return masks_and_clusters


class Medeteranian(_NamedRegions):
    _default_regions: ty.Tuple[str, ...] = (
        'S.W.South-America',
        'W.North-America',
        'N.Central-America',
        'Mediterranean',
        'S.Australia',
        'W.Southern-Africa',
        'E.Southern-Africa',
    )


class Asia(_NamedRegions):
    _default_regions: ty.Tuple[str, ...] = (
        'Russian-Far-East',
        'E.Asia',
        'S.E.Asia',
        'N.Australia',
    )
