import itertools
import typing as ty

import numpy as np
import regionmask

from ._base import _mask_cluster_type
from ._base import RegionExtractor


class NamedRegions(RegionExtractor):
    region_database = regionmask.defined_regions.ar6.all
    # TODO this should be a config setting
    select_regions: ty.Tuple[str, ...] = (
        'S.W.South-America',
        'W.North-America',
        'N.Central-America',
        'Mediterranean',
        'S.Australia',
        'W.Southern-Africa',
        'E.Southern-Africa',
    )

    def get_masks(self) -> _mask_cluster_type:  # pragma: no cover
        mask_2d = ~self.data_set[self.variable].isnull().all(dim='time')
        region_map = self.region_database.mask(self.data_set.lon, self.data_set.lat)
        mask_values = mask_2d.values

        masks = []
        for i, b in enumerate(self.region_database.names):
            if b not in self.select_regions:
                continue

            masks.append((region_map == i).values & mask_values)
        lats = self.data_set['lat'].values
        lons = self.data_set['lon'].values
        coords = []

        # TODO this is so blunt, not even sure if it is completely correct
        for m in masks:
            this_coords = []
            for i, j in itertools.product(
                range(mask_2d.shape[0]),
                range(mask_2d.shape[1]),
            ):
                if m[i][j]:
                    this_coords.append(
                        np.array(
                            [
                                [
                                    lats[i],
                                    lons[j],
                                ],
                            ],
                        ),
                    )
            coords.append(np.array(this_coords))
        return masks, coords

    def filter_masks_and_clusters(
        self,
        masks_and_clusters: _mask_cluster_type,
    ) -> _mask_cluster_type:
        return masks_and_clusters


class Asia(NamedRegions):
    select_regions: ty.Tuple[str, ...] = (
        'Russian-Far-East',
        'E.Asia',
        'S.E.Asia',
        'N.Australia',
    )
