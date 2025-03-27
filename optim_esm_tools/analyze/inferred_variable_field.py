import numpy as np
import numba
import xarray as xr
import dask
import typing as ty

import dask.array as dask_array


def detrend_array(x, y, is_known_nan: ty.Optional[np.ndarray] = None):
    assert len(x) == len(y), f'Mismatch {len(x)}!={len(y)}'
    assert len(y.shape) == 3, f'Only works for 3d-array'
    if is_known_nan is None:
        is_known_nan = np.all(np.all(np.isnan(y), axis=1), axis=1)
    assert all(
        isinstance(a, np.ndarray) for a in (x, y, is_known_nan)
    ), f'One or more unknown types ({[type(a) for a in (x,y,is_known_nan)]})'
    assert (
        len(is_known_nan) == len(x) == len(y)
    ), f'Mismatch {len(x)}, {len(y)}, {len(is_known_nan)}'

    if np.any(is_known_nan):
        res_full = np.empty_like(y)
        res_full[is_known_nan] = np.nan

        res_full[~is_known_nan] = detrend_array_nb(x[~is_known_nan], y[~is_known_nan])

        return res_full
    return detrend_array_nb(x, y)


@numba.njit
def running_mean(a: np.ndarray, window: int) -> np.ndarray:
    res = np.zeros_like(a)
    res[:] = np.nan
    half_win = window // 2
    mean = 0
    for i, v in enumerate(a):
        mean += v
        if i >= window:
            mean -= a[i - window]
        if i >= (window - 1):
            res[i - half_win + 1] = mean / window
    return res


@numba.njit
def running_mean_array(a: np.ndarray, window: int) -> np.ndarray:
    _, len_x, len_y = a.shape
    res = np.zeros_like(a)
    res[:] = np.nan
    for i in range(len_x):
        for j in range(len_y):
            res[:, i, j] = running_mean(a[:, i, j], window)
    return res


@numba.njit
def mean(x, n):
    s = 0
    for xx in x:
        s += xx / n
    return s


@numba.njit
def slope(x, x_mean, y, y_mean):
    return sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean) ** 2)


@numba.njit
def offset(x_mean, y_mean, a):
    return y_mean - a * x_mean


@numba.njit
def detrend(x, y):
    n = len(x)
    x_mean = mean(x, n)
    y_mean = mean(y, n)

    a = slope(x, x_mean, y, y_mean)
    b = offset(x_mean, y_mean, a)
    return y - (a * x + b)


@numba.njit
def detrend_array_nb(x, y):
    res = np.empty_like(y)
    _, ii, jj = y.shape
    for i in range(ii):
        for j in range(jj):
            res[:, i, j] = detrend(x, y[:, i, j])
    return res


@dask.delayed
def _dask_get_values(a: xr.DataArray):
    return a.values


def _dask_to_da(func: ty.Callable, args: tuple, da_original: xr.DataArray):
    return xr.DataArray(
        dask_array.from_delayed(
            dask.delayed(
                func,
            )(*args),
            shape=da_original.shape,
            meta='f8',
        ),
        coords=da_original.coords,
        dims=da_original.dims,
    )


def inferred_fields_to_dataset(
    ds,
    field=None,
    do_detrend=True,
    do_running_mean=True,
    _rm=None,
    use_dask=True,
):
    field = field or ds.variable_id
    from optim_esm_tools.config import config

    _rm = _rm or int(config['analyze']['moving_average_years'])

    t_year = np.array(
        [
            (t if isinstance(t, (int, np.integer)) else t.year)
            for t in ds['time'].values
        ],
    )
    if not use_dask:
        v = ds[field].values
        xr_kw = dict(dims=ds[field].dims, coords=ds[field].coords)
        if do_running_mean:
            # We keep this one as an array since we need it below.
            field_rm_np = running_mean_array(v, _rm)
        if do_detrend:
            field_det = xr.DataArray(detrend_array(t_year, v), **xr_kw)
        if do_running_mean and do_detrend:
            field_det_rm = xr.DataArray(detrend_array(t_year, field_rm_np), **xr_kw)
        field_rm = xr.DataArray(field_rm_np, **xr_kw)
    else:
        v_da = ds[field]
        v = _dask_get_values(v_da)
        if do_running_mean:
            field_rm = _dask_to_da(
                func=running_mean_array,
                args=(v, _rm),
                da_original=v_da,
            )
        if do_detrend:
            field_det = _dask_to_da(
                func=detrend_array,
                args=(t_year, v),
                da_original=v_da,
            )
        if do_running_mean and do_detrend:
            field_det_rm = _dask_to_da(
                func=detrend_array,
                args=(t_year, _dask_get_values(field_rm)),
                da_original=v_da,
            )
    if do_running_mean:
        ds[f'{field}_run_mean_{_rm}'] = field_rm
    if do_detrend:
        ds[f'{field}_detrend'] = field_det
    if do_running_mean and do_detrend:
        ds[f'{field}_detrend_run_mean_{_rm}'] = field_det_rm
    return ds
