# -*- coding: utf-8 -*-
import numpy as np
import typing as ty
import xarray as xr
from functools import wraps


def _native_date_fmt(time_array: np.array, date: ty.Tuple[int, int, int]):
    """Create date object using the date formatting from the time-array"""

    if isinstance(time_array, xr.DataArray):
        return _native_date_fmt(time_array=time_array.values, date=date)

    if not len(time_array):
        raise ValueError(f'No values in dataset?')

    # Support cftime.DatetimeJulian, cftime.DatetimeGregorian, cftime.DatetimeNoLeap and similar
    _time_class = time_array[0].__class__
    return _time_class(*date)


def apply_abs(apply=True, add_abs_to_name=True, _disable_kw='apply_abs'):
    """Apply np.max() to output of function (if apply=True)
    Disable in the function kwargs by using the _disable_kw argument

    Example:
        ```
        @apply_abs(apply=True, add_abs_to_name=False)
        def bla(a=1, **kw):
            print(a, kw)
            return a
        assert bla(-1, apply_abs=True) == 1
        assert bla(-1, apply_abs=False) == -1
        assert bla(1) == 1
        assert bla(1, apply_abs=False) == 1
        ```
    Args:
        apply (bool, optional): apply np.abs. Defaults to True.
        _disable_kw (str, optional): disable with this kw in the function. Defaults to 'apply_abs'.
    """

    def somedec_outer(fn):
        @wraps(fn)
        def somedec_inner(*args, **kwargs):
            response = fn(*args, **kwargs)
            do_abs = kwargs.get(_disable_kw)
            if do_abs or (do_abs is None and apply):
                if add_abs_to_name and isinstance(getattr(response, 'name'), str):
                    response.name = f'Abs. {response.name}'
                return np.abs(response)
            return response

        return somedec_inner

    return somedec_outer


def _remove_any_none_times(da, time_dim, drop=True):
    data_var = da.copy()
    time_null = data_var.isnull().all(dim=set(data_var.dims) - {time_dim})
    if np.all(time_null):
        # If we take a running mean of 10 (the default), and the array is shorter than
        # 10 years we will run into issues here because a the window is longer than the
        # array. Perhaps we should raise higher up.
        raise ValueError(
            f'This array only has NaN values, perhaps array too short ({len(time_null)} < 10)?'
        )

    if np.any(time_null):
        try:
            # For some reason only alt_calc seems to work even if it should be equivalent to the data_var
            # I think there is some fishy indexing going on in pandas <-> dask
            # Maybe worth raising an issue?
            alt_calc = xr.where(~time_null, da, np.nan)
            if drop:
                alt_calc = alt_calc.dropna(time_dim)
            data_var = data_var.load().where(~time_null, drop=drop)
            assert np.all((alt_calc == data_var).values)
        except IndexError as e:
            from optim_esm_tools.config import get_logger

            get_logger().error(e)
            return alt_calc
    return data_var


def mask_xr_ds(data_set, da_mask, masked_dims=None, drop=False):
    # Modify the ds in place - make a copy!
    data_set = data_set.copy()
    if masked_dims is None:
        from optim_esm_tools.config import config

        masked_dims = config['analyze']['lon_lat_dim'].split(',')

    ds_start = data_set.copy()
    func_by_drop = {True: _drop_by_mask, False: _mask_xr_ds}[drop]
    data_set = func_by_drop(data_set, masked_dims, ds_start, da_mask)
    data_set = data_set.assign_attrs(ds_start.attrs)
    return data_set


def _drop_by_mask(data_set, masked_dims, ds_start, da_mask):
    """Drop values with masked_dims dimensions.
    Unfortunately, data_set.where(da_mask, drop=True) sometimes leads to bad results,
    for example for time_bnds (time, bnds) being dropped by (lon, lat). So we have to do
    some funny bookkeeping of which data vars we can drop with data_set.where.
    """

    dropped = []
    for k, data_array in data_set.data_vars.items():
        if not all(dim in list(data_array.dims) for dim in masked_dims):
            dropped += [k]

    data_set = data_set.drop_vars(dropped)

    data_set = data_set.where(da_mask, drop=True)

    # Restore ignored variables and attributes
    for k in dropped:
        data_set[k] = ds_start[k]
    return data_set


def _mask_xr_ds(data_set, masked_dims, ds_start, da_mask):
    """Rebuild data_set for each variable that has all masked_dims"""
    for k, data_array in data_set.data_vars.items():
        if all(dim in list(data_array.dims) for dim in masked_dims):
            # First dim is time?
            if (
                'time' == data_array.dims[0] and data_array.shape[1:] == da_mask.T.shape
            ) or data_array.shape == da_mask.T.shape:
                raise ValueError(f'Please make "{k}" lat, lon, now "{data_array.dims}"')
            da = data_set[k].where(da_mask, drop=False)
            da = da.assign_attrs(ds_start[k].attrs)
            data_set[k] = da

    return data_set
