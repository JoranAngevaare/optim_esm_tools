from ._base import RegionExtractor, _mask_cluster_type, apply_options
import itertools
import numpy as np
import typing as ty
import xarray as xr
import optim_esm_tools as oet


class MaskAll(RegionExtractor):
    @apply_options
    def get_masks(
        self,
        step_size: int = 5,
        force_continuity: bool = True,
    ) -> _mask_cluster_type:  # pragma: no cover
        mask_2d: xr.DataArray = ~self.data_set[self.variable].isnull().all(dim='time')
        mask_values: np.ndarray = mask_2d.values
        lats: np.ndarray = self.data_set['lat'].values
        lons: np.ndarray = self.data_set['lon'].values
        masks: ty.List[np.ndarray] = []
        coords: ty.List[np.ndarray] = []

        for i, j in itertools.product(
            range(0, mask_2d.shape[0], step_size),
            range(0, mask_2d.shape[1], step_size),
        ):
            this_mask = np.zeros_like(mask_values)
            this_coords = []
            for ii in range(i, i + step_size):
                if ii >= len(lats):
                    continue
                for jj in range(j, j + step_size):
                    if jj >= len(lons):
                        continue
                    if mask_values[ii][jj]:
                        this_coords.append([lats[ii], lons[jj]])
                        this_mask[ii][jj] = True
            if this_mask.sum():
                coords.append(np.array(this_coords))
                masks.append(this_mask)
        if force_continuity:
            masks = oet.analyze.clustering._split_to_continuous(masks=masks)
            lat, lon = np.meshgrid(lats, lons)
            coords = [
                oet.analyze.clustering._find_lat_lon_values(m, lats=lat.T, lons=lon.T)
                for m in masks
            ]
        return masks, coords

    def filter_masks_and_clusters(
        self,
        masks_and_clusters: _mask_cluster_type,
    ) -> _mask_cluster_type:
        return masks_and_clusters
