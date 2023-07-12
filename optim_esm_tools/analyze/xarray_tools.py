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
