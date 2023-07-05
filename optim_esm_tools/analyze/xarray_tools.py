# -*- coding: utf-8 -*-
import numpy as np
import typing as ty
import xarray as xr
from functools import wraps
import numba


def _native_date_fmt(time_array: np.array, date: ty.Tuple[int, int, int]):
    """Create date object using the date formatting from the time-array"""

    if isinstance(time_array, xr.DataArray):
        return _native_date_fmt(time_array=time_array.values, date=date)

    if not len(time_array):
        raise ValueError(f'No values in dataset?')

    # Support cftime.DatetimeJulian, cftime.DatetimeGregorian, cftime.DatetimeNoLeap and similar
    _time_class = time_array[0].__class__
    return _time_class(*date)


def _mask2d_to_xy_slice(mask: np.array, cyclic: bool = False) -> np.array:
    """Lazy alias for doing a box-cut"""
    where = np.argwhere(mask)
    slices = np.zeros((len(mask), 2, 2), dtype=np.int64)
    n_slices = 1
    slices[0][0][0] = where[0][0]
    slices[0][0][1] = where[0][0] + 1
    slices[0][1][0] = where[0][1]
    slices[0][1][1] = where[0][1] + 1

    for x, y in where[1:]:
        # x1 and y1 are EXLCUSIVE!
        for s_i, ((x0, x1), (y0, y1)) in enumerate(slices[:n_slices]):
            if x0 <= x <= x1 and y0 <= y <= y1:
                if x == x1:
                    slices[s_i][0][1] += 1
                if y == y1:
                    slices[s_i][1][1] += 1
                if cyclic:
                    raise ValueError
                break
        else:
            slices[n_slices][0][0] = x
            slices[n_slices][0][1] = x + 1

            slices[n_slices][1][0] = y
            slices[n_slices][1][1] = y + 1

            n_slices += 1
    return slices[:n_slices]


_n_mask2d_to_xy_slice = numba.njit(_mask2d_to_xy_slice)


def mask2d_to_xy_slice(*args, **kwargs):
    return _n_mask2d_to_xy_slice(*args, **kwargs)


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


def detrend(
    data_array: xr.DataArray, dimension: str, use_xrft=False, detrend_type='linear'
) -> xr.DataArray:
    """Wrapper for detrending using two very similar methods.

    xrft is generally faster than the crude - but the simple method in detrend_dim is
    more robust for nan values

    Args:
        data_array (xr.DataArray): data array
        dimension (str): time dimension
        use_xrft (bool, optional): Use the xrft implementation. Defaults to True.
        detrend_type (str, optional): Method of detrending either linear or constant. Defaults to 'linear'.

    Raises:
        TypeError: If a dataset is provided instead of a data array
        NotImplementedError: if a non existing detrend_type is provided

    Returns:
        xr.DataArray: detrended xr.DataArray
    """
    if isinstance(data_array, xr.Dataset):
        raise TypeError('First argument is a dataset instead of a data array.')
    if use_xrft:
        import xrft

        return xrft.detrend(data_array, dimension, detrend_type=detrend_type)
    methods = dict(linear=1, constant=0)
    if detrend_type not in methods:
        raise NotImplementedError(
            'Currently not implemented to do polinominal fits of order > 1'
        )
    return detrend_dim(data_array, dim=dimension, polynome_order=methods[detrend_type])


def detrend_dim(da: xr.DataArray, dim: int, polynome_order=1):
    # Source: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=polynome_order)
    fit = xr.polyval(da[dim], p['polyfit_coefficients'])
    return da - fit
