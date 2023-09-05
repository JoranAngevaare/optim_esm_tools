import abc
import typing as ty

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import xarray as xr

import optim_esm_tools as oet
from ._base import _mask_cluster_type
from ._base import apply_options
from optim_esm_tools.analyze.clustering import build_cluster_mask
from optim_esm_tools.plotting.map_maker import HistoricalMapMaker
from optim_esm_tools.plotting.map_maker import MapMaker
from optim_esm_tools.region_finding.percentiles import Percentiles
from optim_esm_tools.utils import check_accepts


class _HistroricalLookup(abc.ABC):
    data_set: xr.Dataset

    @apply_options
    def find_historical(
        self,
        match_to: str = 'piControl',
        look_back_extra: int = 0,
        query_updates: ty.Optional[ty.Mapping] = None,
        search_kw: ty.Optional[ty.Mapping] = None,
    ) -> ty.Optional[ty.List[str]]:
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
    def get_historical_ds(
        self,
        read_ds_kw: ty.Optional[ty.Mapping] = None,
        **kw,
    ) -> xr.Dataset:
        read_ds_kw = read_ds_kw or {}

        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)  # type: ignore
        historical_path = self.find_historical(**kw)
        if historical_path is None:
            raise ValueError('No match found')  # pragma: no cover
        res = oet.read_ds(historical_path[0], **read_ds_kw)
        assert isinstance(res, xr.Dataset)
        return res


class LocalHistory(Percentiles, _HistroricalLookup):
    def _all_pass_historical(
        self,
        labels: ty.List[str],
        n_times_historical: ty.Union[float, int],
        read_ds_kw: ty.Optional[ty.Mapping] = None,
    ) -> npt.NDArray[np.bool_]:
        read_ds_kw = read_ds_kw or {}
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)  # type: ignore

        historical_ds = self.get_historical_ds(read_ds_kw=read_ds_kw)

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
        return all_mask

    @apply_options
    def get_masks(
        self,
        n_times_historical: ty.Union[int, float] = 4,
        read_ds_kw: ty.Optional[ty.Mapping] = None,
        lon_lat_dim: ty.Tuple[str, str] = ('lon', 'lat'),
    ) -> _mask_cluster_type:
        all_mask = self._build_combined_mask(
            method='all_pass_historical',
            n_times_historical=n_times_historical,
            read_ds_kw=read_ds_kw,
        )
        masks, clusters = build_cluster_mask(
            all_mask,
            lon_coord=self.data_set[lon_lat_dim[0]].values,
            lat_coord=self.data_set[lon_lat_dim[1]].values,
        )
        return masks, clusters

    @apply_options
    def _plot_basic_map(
        self,
        historical_normalizations=None,
        read_ds_kw: ty.Optional[ty.Mapping] = None,
    ) -> MapMaker:
        read_ds_kw = read_ds_kw or {}
        for k, v in dict(min_time=None, max_time=None).items():
            read_ds_kw.setdefault(k, v)  # type: ignore
        ds_historical = self.get_historical_ds(read_ds_kw=read_ds_kw)

        mm = HistoricalMapMaker(
            self.data_set,
            ds_historical=ds_historical,
            normalizations=historical_normalizations,
        )
        mm.plot_selected()
        plt.suptitle(self.title, y=0.95)
        return mm
