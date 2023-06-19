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

def mask2d_to_xy_slice(*args, **kwargs):
    try:
        import numba
    except (ImportError, ModuleNotFoundError) as exeception:
        raise ModuleNotFoundError('Numba is an optional dependency, please try pip install numba') from e
    return numba.njit(_mask2d_to_xy_slice)(*args, **kwargs)


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
