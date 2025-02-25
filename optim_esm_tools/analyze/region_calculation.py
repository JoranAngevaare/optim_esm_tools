import optim_esm_tools as oet
import numpy as np

import typing as ty


import xarray as xr
import functools
from optim_esm_tools.analyze.tools import (
    weighted_mean_array,
    _weighted_mean_array_numba,
    running_mean,
    running_mean_array,
)
import scipy


class RegionPropertyCalculator:
    """
    For a given region, calculate properties that can be used to assess how special a given region is.

    For this purpose, three essential elements are required:
     - The dataset of interest (ds_global)
     - The pi-control dataset of interest (ds_pi)
     - The region of interest (mask)
    """

    def __init__(
        self,
        ds_global: xr.Dataset,
        ds_pi: xr.Dataset,
        mask: ty.Union[np.ndarray, xr.DataArray],
        field: ty.Optional[str] = None,
        ds_local: ty.Optional[xr.Dataset] = None,
        ds_pi_local: ty.Optional[xr.Dataset] = None,
        _tropic_lat: ty.Union[int, float] = float(
            oet.config.config["analyze"]["tropics_latitude"],
        ),
        _rm_years: int = int(oet.config.config["analyze"]["moving_average_years"]),
    ):
        """

        Args:
            ds_global (xr.Dataset): dataset of interest
            ds_pi (xr.Dataset): pi-control dataset corresponding to ds_global
            mask (ty.Union[np.ndarray, xr.DataArray]): region of interest in the dataset of interest
            field (ty.Optional[str], optional): variable_id in the dataset to extract. Defaults to None in which case we read it from ds_global.
            ds_local (ty.Optional[xr.Dataset], optional): Masked ds_global. Defaults to None, and we compute the masked dataset using ds_global and mask.
            ds_pi_local (ty.Optional[xr.Dataset], optional): Masked ds_pi. Defaults to, similar to ds_local for the pi-control dataset.
            _tropic_lat (ty.Union[int, float], optional): Value of tropics to use. Defaults to float( oet.config.config["analyze"]["tropics_latitude"], ).
            _rm_years (int, optional): Number of years for running mean-calculations. Defaults to int(oet.config.config["analyze"]["moving_average_years"]).
        """
        self._tropic_lat = _tropic_lat
        self._rm_years = _rm_years
        self.ds_global = ds_global
        self.ds_pi = ds_pi
        self.mask = mask
        assert mask.dtype == "bool"
        self.field = field or ds_global.variable_id

        da_mask = self.mask_to_da(mask)
        self.da_mask: xr.DataArray = da_mask
        kw = dict(mask=da_mask, add_global_mask=False)
        _common_keys = [
            self.field_rm,
            "time",
            "cell_area",
            *oet.config.config["analyze"]["lon_lat_dim"].split(","),
        ]
        keep_keys_sc = [self.field, *_common_keys]
        keep_keys_pi = [self.field_detrend_rm, f"{self.field}_detrend", *_common_keys]

        self.ds_local = ds_local or oet.analyze.xarray_tools.mask_to_reduced_dataset(
            self.ds_global,
            keep_keys=keep_keys_sc,
            **kw,
        )
        self.ds_pi_local = (
            ds_pi_local
            or oet.analyze.xarray_tools.mask_to_reduced_dataset(
                self.ds_pi,
                keep_keys=keep_keys_pi,
                **kw,
            )
        )
        self._cache: ty.Dict[str, ty.Any] = dict()

    @property
    def field_detrend_rm(self) -> str:
        return f"{self.field}_detrend_run_mean_{self._rm_years}"

    def mask_to_da(
        self,
        mask: ty.Union[np.ndarray, xr.DataArray],
        fall_back_field: str = "cell_area",
    ) -> xr.DataArray:
        """Handle masks of either numpy or xarray types and make sure an xarray.DataArray is returned

        Args:
            mask (ty.Union[np.ndarray, xr.DataArray]): _description_
            fall_back_field (str, optional): _description_. Defaults to "cell_area".

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(mask, xr.DataArray):
            return mask
        if isinstance(mask, np.ndarray):
            mask_da = self.ds_global[fall_back_field].astype(np.bool_)
            mask_da.data = mask
            return mask_da
        raise ValueError

    @property
    def field_rm(self) -> str:
        return f"{self.field}_run_mean_{self._rm_years}"

    def weigthed_mean_cached(
        self,
        field: str,
        data_set: str = "ds_local",
    ) -> np.ndarray:
        k = f"{data_set}-weighted_mean_time-{field}"
        if k not in self._cache:
            _ds = getattr(self, data_set, "NO DATA")
            assert isinstance(
                _ds,
                xr.Dataset,
            ), f"{_ds} is not an xr.Dataset or not available at {data_set}"

            self._cache[k] = weighted_mean_array(_ds, field=field)
        return self._cache[k]

    def _calc_max_end(self) -> float:
        """Calculte the difference between the end and the maximum calue in the default moving average filtered time series"""
        _m = self.weigthed_mean_cached(self.field_rm, "ds_local")
        _rm_years = int(self._rm_years)
        end = _m[-_rm_years // 2]
        return max(np.abs(np.nanmax(_m) - end), np.abs(np.nanmin(_m) - end))

    def _calc_std_trop(self, **kw) -> float:
        """Calculte the average standard deviation in the scenario dataset"""
        return self._calc_std_trop_inner(**kw, values_from="scenario")

    def _calc_pi_std_trop(self, **kw) -> float:
        """Calculte the average standard deviation in the pi-control dataset"""
        return self._calc_std_trop_inner(**kw, values_from="pi")

    @oet.utils.check_accepts(accepts=dict(values_from=["scenario", "pi"]))
    def _calc_std_zone(
        self,
        std_field: str = "std detrended",
        values_from: str = "scenario",
        remove_zero_std: bool = False,
        zone_name: str = "tropics",
    ) -> float:
        ds_values = dict(pi=self.ds_pi, scenario=self.ds_global)[values_from]
        mask = self.mask

        _, lat = np.meshgrid(self.ds_global.lon.values, self.ds_global.lat.values)
        if not isinstance(mask, np.ndarray):
            mask = mask.values
        assert np.all(
            np.diff(self.ds_global.lat.values) < 0,
        ), "Lats should be in descending order"

        lats = lat[mask.astype(np.bool_)].flatten()
        use_mask = self._mask_by_zone(lats, zone_name=zone_name)
        if remove_zero_std:
            log = oet.get_logger()
            log.debug(f"Use {use_mask.sum()} datapoints, remove zero std")
            use_mask &= ds_values[std_field].values > 0
            log.debug(f"{use_mask.sum()} datapoints left")

        data = ds_values[std_field].values[use_mask]
        weights = ds_values["cell_area"].values[use_mask]

        dinom = _weighted_mean_array_numba(
            data=data,
            weights=weights,
            has_time_dim=False,
        )

        return float(dinom)

    @oet.utils.check_accepts(accepts=dict(values_from=["scenario", "pi"]))
    def _calc_std_trop_inner(
        self,
        min_frac: float = 0.5,
        std_field: str = "std detrended",
        values_from: str = "scenario",
        remove_zero_std: bool = False,
    ) -> float:
        _tropic_lat = self._tropic_lat
        ds_global = self.ds_global
        ds_values = dict(pi=self.ds_pi, scenario=ds_global)[values_from]
        mask = self.mask
        log = oet.get_logger()
        _, lat = np.meshgrid(ds_global.lon.values, ds_global.lat.values)
        if not isinstance(mask, np.ndarray):
            mask = mask.values
        assert np.all(
            np.diff(ds_global.lat.values) < 0,
        ), "Lats should be in descending order"

        lats = lat[mask.astype(np.bool_)].flatten()
        mask_trop_2d = (lat < _tropic_lat) & (lat > -_tropic_lat)

        if (
            np.sum([(lats > -_tropic_lat) & (lats < _tropic_lat)]) / len(lats)
            > min_frac
        ):
            log.debug("Is tropical")
            is_trop = np.zeros_like(mask)
            is_trop[mask_trop_2d] = True
            use_mask = is_trop
        else:
            log.debug("Is extra tropical en een annanas")
            extra_trop = np.ones_like(mask)
            extra_trop[mask_trop_2d] = False
            use_mask = extra_trop
        if remove_zero_std:
            log.debug(f"Use {use_mask.sum()} datapoints, remove zero std")
            use_mask &= ds_values[std_field].values > 0
            log.debug(f"{use_mask.sum()} datapoints left")
        dinom = _weighted_mean_array_numba(
            data=ds_values[std_field].values[use_mask],
            weights=ds_values["cell_area"].values[use_mask],
            has_time_dim=False,
        )

        return float(dinom)

    def _calc_start_end(self, offset: ty.Optional[int] = None) -> float:
        offset = offset or self._rm_years // 2
        assert isinstance(offset, int)
        data = self.ds_local[self.field_rm].values
        weights = self.ds_local["cell_area"].values
        se = data[-offset] - data[offset]
        # # make sure that there is at least one point in time that is not the same
        # mask = np.any(data[1:]!=data[0], axis=0)
        res = _weighted_mean_array_numba(data=se, weights=weights, has_time_dim=False)
        assert isinstance(res, float)
        return res

    def _calc_siconc_norm(self, calculate_vars=("siconc", "siconca")) -> float:
        return float(
            (
                calculate_norm(
                    ds_global=self.ds_global,
                    mask=self.da_mask,
                    field=self.field,
                )
                if self.field in calculate_vars
                else np.nan
            ),
        )

    def _calc_rho_frac(self, rm_0: int, rm_1: ty.Optional[int] = None) -> float:
        ds = self.ds_local
        field = self.field
        ar = ds[field].values
        if len(ar.shape) > 1:
            ar = ds[field].mean("lat lon".split()).values
        ar_0 = running_mean(ar, rm_0)
        ar_1 = ar if rm_1 is None else running_mean(ar, rm_1)
        return np.nanstd(ar_1 - ar_0)

    def _calc_pi_std(self) -> float:
        return float(
            np.nanstd(
                self.weigthed_mean_cached(
                    f"{self.field}_detrend_run_mean_{self._rm_years}",
                    "ds_pi_local",
                ),
            ),
        )

    def _calc_pi_std_rmx(self, rm_years) -> float:
        return float(
            np.nanstd(
                running_mean(
                    self.weigthed_mean_cached(
                        f"{self.field}_detrend",
                        "ds_pi_local",
                    ),
                    rm_years,
                ),
            ),
        )

    def _calc_psymmetry(self) -> np.float64:
        values = self.weigthed_mean_cached(self.field, "ds_local")
        return oet.analyze.time_statistics.calculate_symmetry_test(values=values)

    def _calc_psymmetry_cached(self) -> np.float64:
        values = self.weigthed_mean_cached(self.field, "ds_local")
        values_as_tuple = tuple(values)
        return cache_psym_test(values=values_as_tuple)

    def _calc_pdip(self) -> float:
        values = self.weigthed_mean_cached(self.field, "ds_local")
        return oet.analyze.time_statistics.calculate_dip_test(values=values)

    def _calc_j2(self, *a, do_raise=False, **k) -> float:
        values = self.weigthed_mean_cached(self.field_rm, "ds_local")
        try:
            return np.divide(*_max_and_second_jump(values, *a, **k))
        except ValueError as e:
            if do_raise:
                raise e
            return np.nan

    def _calc_jump_n_years(
        self,
        n_years: int = 10,
        rm_years: ty.Optional[int] = None,
    ) -> float:
        a = self.weigthed_mean_cached(self.field, data_set="ds_local")
        rm_years = rm_years or self._rm_years
        a_rm = running_mean(a, rm_years)
        return np.nanmax(np.abs(a_rm[n_years:] - a_rm[:-n_years]))

    def _calc_max_end_rmx(self, rm_alt=50, apply_max=True) -> float:
        assert rm_alt % 2 == 0, rm_alt
        _m = self.weigthed_mean_cached(self.field, "ds_local")
        rm_x = running_mean(_m, rm_alt)
        end = rm_x[-rm_alt // 2]

        if not apply_max:
            return np.nanmax(rm_x) - end
        return max(np.abs(np.nanmax(rm_x) - end), np.abs(np.nanmin(rm_x) - end))

    def _add_std_rm_alt_to_ds(
        self,
        ds_values: xr.Dataset,
        set_field: str,
        rm_alt: int,
    ) -> None:
        yearly_means = ds_values[f"{self.field}_detrend"].values
        vals_2d = ds_values["cell_area"].copy()
        vals_2d.attrs["units"] = "std"

        rm_50_3d = running_mean_array(yearly_means, rm_alt)

        vals_2d.data = np.nanstd(rm_50_3d, axis=0)
        ds_values[set_field] = vals_2d

    def _sigma_trop_rmx(self, values_from: str = "scenario", rm_alt: int = 50) -> float:
        cache_to_field = f"std_{self.field}_detrend_rm_{rm_alt}"
        ds_values = dict(pi=self.ds_pi, scenario=self.ds_global)[values_from]
        if cache_to_field not in ds_values:
            self._add_std_rm_alt_to_ds(ds_values, cache_to_field, rm_alt)

        return self._calc_std_trop_inner(
            remove_zero_std=True,
            values_from=values_from,
            std_field=cache_to_field,
        )

    def _mask_to_lats(self) -> np.ndarray:
        assert np.all(
            np.diff(self.ds_global.lat.values) < 0,
        ), "Lats should be in descending order"
        _, lat = np.meshgrid(self.ds_global.lon.values, self.ds_global.lat.values)
        lats = lat[self.mask.astype(np.bool_)].flatten()
        return lats

    def _get_named_zone(self, zone_name: str) -> np.ndarray:
        lats = self._mask_to_lats()
        return self._mask_by_zone(zone_name=zone_name, lats=lats)

    def _mask_by_zone(self, lats: np.ndarray, zone_name: str, **kw) -> np.ndarray:
        zone_kw = dict(
            tropics=dict(zone_bounds=np.array([-90, self._tropic_lat]), reflect=True),
        )
        if zone_name == "tropics_land_or_water":
            oet.get_logger().warning(f"Replaced {zone_name} with tropics")
            zone_name = "tropics"
        return self._mask_by_zone_inner(lats, **zone_kw[zone_name])  # type: ignore

    def _mask_by_zone_inner(
        self,
        lats: np.ndarray,
        zone_bounds: np.ndarray,
        reflect: bool = False,
    ) -> np.ndarray:
        _, lat = np.meshgrid(self.ds_global.lon.values, self.ds_global.lat.values)
        if reflect:
            zone_bounds = np.concatenate([-zone_bounds, zone_bounds])
        assert np.all(
            np.diff(self.ds_global.lat.values) < 0,
        ), "Lats should be in descending order"
        zone_bounds = np.sort(zone_bounds)[::-1]
        zone_ranges = np.vstack([zone_bounds[:-1], zone_bounds[1:]])
        overlaps = []
        for z_left, z_right in zip(*zone_ranges):
            this_n = np.sum((lats <= z_left) & (lats > z_right))
            if reflect:
                this_n += np.sum((lats >= -z_left) & (lats < -z_right))
            overlaps.append(this_n)
        zone_i = np.argmax(overlaps)
        use_mask = (lat <= zone_ranges[0][zone_i]) & (lat > zone_ranges[1][zone_i])
        if reflect:
            use_mask |= (lat >= -zone_ranges[0][zone_i]) & (
                lat < -zone_ranges[1][zone_i]
            )
        return use_mask

    def calculate_mse_trop_rmx(
        self,
        start_or_max: str = "start",
        zone_name: str = "tropics",
        rm: int = 10,
        field: ty.Optional[str] = None,
    ):
        mask = self._get_named_zone(zone_name)
        data = self.ds_global[field or self.field].where(mask, drop=False).values
        weights = self.ds_global["cell_area"].values

        trop_avg = _weighted_mean_array_numba(data, weights)
        if rm is not None:
            trop_avg = running_mean(trop_avg, rm)
        offset = int(rm // 2 if rm else 0)
        if start_or_max == "start":
            se = trop_avg[-1 - offset] - trop_avg[offset]
            return se
        elif start_or_max == "max":
            end = trop_avg[offset]
            return max(
                np.abs(np.nanmax(trop_avg) - end),
                np.abs(np.nanmin(trop_avg) - end),
            )
        raise NotImplementedError(f"Method {start_or_max} not available")

    def max_rmx(self, rm_alt=50, field=None, values_from="ds_local"):
        _m = self.weigthed_mean_cached(field or self.field, values_from)
        rm_x = running_mean(_m, rm_alt)
        return np.nanmax(rm_x)

    def calculate(self) -> ty.Dict[str, ty.Union[str, int, float, bool]]:  # type: ignore
        doc = dict(
            p_sym_mi=self._calc_psymmetry_cached(),
            p_dip=self._calc_pdip(),
            se=abs(self._calc_start_end()),
            J2_rm=self._calc_j2(),
            J2_min0_rm=self._calc_j2(force_same_sign=False, min_distance=0),
            area_sum=float(self.ds_local["cell_area"].sum()),
            j=self._calc_jump_n_years(),
            j_50=self._calc_jump_n_years(50),
            variable_id=self.field,
            _pi_std=self._calc_pi_std(),
            _pi_std_rm50=self._calc_pi_std_rmx(50),
            _frac_norm=self._calc_siconc_norm(),
            _frac_rho10_50=self._calc_rho_frac(rm_0=50, rm_1=10),
            me_rm50=self._calc_max_end_rmx(rm_alt=50),
            me_rm50_signed=self._calc_max_end_rmx(rm_alt=50, apply_max=False),
            std_pi_trop_rm50=self._sigma_trop_rmx(values_from="pi", rm_alt=50),
        )
        doc.update(
            dict(
                max_jump=np.divide(doc["j"], doc["_pi_std"]),
                max_jump_rm50=np.divide(doc["j_50"], doc["_pi_std_rm50"]),
                e_s=np.divide(doc["se"], doc["_frac_norm"]),
                rho10_50=np.divide(doc["se"], doc["_frac_rho10_50"]),
                me_std_pi_trop_rm50=np.divide(doc["me_rm50"], doc["std_pi_trop_rm50"]),
            ),
        )
        zone = 'tropics'
        doc["se_std_trop_domain"] = np.divide(
            doc["se"],
            self._calc_std_zone(remove_zero_std=True, zone_name=zone),
        )
        doc["se_pi_std_trop_domain"] = np.divide(
            doc["se"],
            self._calc_std_zone(remove_zero_std=True, zone_name=zone, values_from="pi"),
        )
        doc["mj_std_trop_domain"] = np.divide(
            doc["j"],
            self._calc_std_zone(remove_zero_std=True, zone_name=zone),
        )
        doc["mj_pi_std_trop_domain"] = np.divide(
            doc["j"],
            self._calc_std_zone(remove_zero_std=True, zone_name=zone, values_from="pi"),
        )
        doc["_se_trop"] = self.calculate_mse_trop_rmx("start", rm=10)
        doc["_me_trop_rm50"] = self.calculate_mse_trop_rmx("max", rm=50)

        doc["se_vs_se_trop"] = np.divide(self._calc_start_end(), doc["_se_trop"])
        doc["me_vs_me_trop_rm50"] = np.divide(doc["me_rm50"], doc["_me_trop_rm50"])
        doc["me_max_rm50"] = np.divide(doc["me_rm50"], self.max_rmx(50))
        doc['me_pi_std_rm50'] = np.divide(doc['me_rm50'], doc['_pi_std_rm50'])
        doc['se_pi_std'] = np.divide(doc['se'], doc['_pi_std'])
        return {k: (v if v is not None else np.nan) for k, v in doc.items()}


def summarize_stats(
    ds_global: xr.Dataset,
    ds_pi: xr.Dataset,
    mask: ty.Union[np.ndarray, xr.DataArray],
    field: str,
) -> ty.Dict[str, ty.Union[str, int, float, bool]]:
    return RegionPropertyCalculator(
        ds_global=ds_global,
        ds_pi=ds_pi,
        mask=mask,
        field=field,
    ).calculate()


def _max_and_second_jump(
    values,
    n_years_difference: int = 10,
    min_distance: int = 10,
    force_same_sign: bool = True,
):
    difference = values[n_years_difference:] - values[:-n_years_difference]

    max_index = np.nanargmax(np.abs(difference))

    slice_disabled = slice(
        np.max(
            [int(max_index) - min_distance, 0],
        ),
        np.min([int(max_index) + min_distance, len(difference)]),
    )

    delta_nanned = difference.copy()
    delta_nanned[slice_disabled] = np.nan
    if (n_nan := np.sum(np.isnan(delta_nanned))) < min_distance:
        raise RuntimeError(
            f"Should have at least {min_distance} nans, got {n_nan}",
        )
    sign = np.sign(difference[max_index])
    sign = sign if force_same_sign else -sign
    second_jump_index = np.nanargmax(sign * delta_nanned)

    return difference[max_index], difference[second_jump_index]


def find_max_in_equal_area(
    ds,
    target_area,
    time_index=0,
    field=None,
    tqdm=False,
    step_size=1.0,
    max_multiplier=1000,
):
    field = field or ds.attrs["variable_id"]
    field = f"{field}_run_mean_10"

    # Shift running mean
    time_index = 5 if time_index == 0 else time_index
    time_index = -5 if time_index == -1 else time_index

    data = ds[field].isel(time=time_index)
    area = ds["cell_area"].load()
    vals = data.values.copy()
    nona_vals = vals.copy()
    nona_vals[np.isnan(vals)] = np.nanmin(vals)
    if not target_area:
        return dict(
            area_m2=target_area,
            max_in_sel=nona_vals.max(),
        )
    da_m = data.copy()
    da_m.data[:] = np.nan

    max_distance_km = oet.analyze.clustering.infer_max_step_size(
        data.lat.values,
        data.lon.values,
    )

    area_perc = target_area / float(area.sum())
    iter_perc = 1 - area_perc * step_size * (
        np.linspace(1, max_multiplier, max_multiplier)
    )
    iter_perc = iter_perc[(iter_perc > 0) & (iter_perc < 1)]
    prev_mask = None
    n_mask = 1
    for i, threshold in enumerate(oet.utils.tqdm(iter_perc, disable=not tqdm)):
        mask = vals > np.percentile(nona_vals, threshold * 100)
        if np.sum(mask) / n_mask < 1.03 and i < len(iter_perc) - 1:
            # Skip steps that are too close together
            continue
        n_mask = np.sum(mask)
        masks, _ = oet.analyze.clustering.build_cluster_mask(
            mask,
            max_distance_km=max_distance_km,
            lat_coord=data.lat.values,
            lon_coord=data.lon.values,
        )
        for m in masks[:1]:
            da_m.data = m

            area_m2 = float(area.where(da_m).sum())
            if area_m2 < target_area and i < len(iter_perc) - 1:
                if prev_mask is None:
                    prev_mask = da_m.copy()
                elif da_m.sum() > prev_mask.sum():
                    prev_mask = da_m.copy()
                continue

            da_sel = data.where(da_m)

            max_in_sel = oet.analyze.tools._weighted_mean_2d_numba(
                da_sel.where(da_m).values,
                weights=area.values,
            )
            if prev_mask is None or target_area > area_m2:
                return dict(
                    area_m2=area_m2,
                    max_in_sel=max_in_sel,
                )

            prev_in_sel = oet.analyze.tools._weighted_mean_2d_numba(
                da_sel.where(prev_mask).values,
                weights=area.values,
            )
            prev_area = float(area.where(prev_mask).sum())

            itp_max = scipy.interpolate.interp1d(
                [prev_area, area_m2],
                [prev_in_sel, max_in_sel],
            )(target_area)
            return dict(
                area_m2=target_area,
                max_in_sel=itp_max,
            )
    # At some point, you just are going to try finding larger areas than possible with this kind of dataset. E.g. when you require half of the area surface of the earth for sea-ice. Which will fail because only a fraction is coverered by a portion of sea ice
    return dict(
        area_m2=1e-10,
        max_in_sel=1e-10,
    )


@functools.lru_cache(maxsize=int(1e9))
def cache_psym_test(values: np.ndarray, **kw) -> np.float64:
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    return oet.analyze.time_statistics.calculate_symmetry_test(values=values, **kw)


def calculate_norm(
    ds_global: xr.Dataset,
    mask: ty.Union[np.ndarray, xr.DataArray],
    field: str,
) -> float:
    if hasattr(mask, "values"):
        mask = mask.values
    use_mask = mask.copy()
    use_mask &= np.any(ds_global[f"{field}_run_mean_10"].values > 0, axis=0)
    target = float(ds_global["cell_area"].where(use_mask).sum())
    kw = dict(
        ds=ds_global,
        target_area=target,
        field=field,
    )
    t0 = find_max_in_equal_area(
        **kw,
        time_index=0,
    )
    t1 = find_max_in_equal_area(
        **kw,
        time_index=-1,
    )
    return max(t0["max_in_sel"], t1["max_in_sel"])


def jump_n_years(
    field: str,
    ds_local: xr.Dataset,
    n_years: int = 10,
    moving_average_years: ty.Optional[int] = None,
) -> np.float64:
    ma = moving_average_years or int(
        oet.config.config["analyze"]["moving_average_years"],
    )
    use_field = f"{field}_run_mean_{ma}"
    a = ds_local[use_field].mean("lat lon".split()).values

    return np.nanmax(np.abs(a[n_years:] - a[:-n_years]))
