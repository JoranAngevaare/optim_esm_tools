import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm
import typing as ty
import xarray as xr
import numba
import json


def _dinfo(a):
    try:
        return np.iinfo(a.dtype)
    except ValueError:
        return np.finfo(a.dtype)


def rank2d(a):
    nan_mask = np.isnan(a)
    a_flat = a[~nan_mask].flatten().astype(np.float64)
    dtype_info = _dinfo(a_flat)
    # Clip infinite from values - they will get ~0 or ~1 for -np.inf and np.inf respectively
    a_flat = np.clip(a_flat, dtype_info.min, dtype_info.max)

    # This is equivalent to (but much faster than)
    # from scipy.stats import percentileofscore
    # import optim_esm_tools as oet
    # pcts = [[percentileofscore(a_flat, i, kind='mean') / 100 for i in aa]
    #         for aa in oet.utils.tqdm(a)]
    # return pcts
    a_sorted, count = np.unique(a_flat, return_counts=True)
    # One value can occur more than once, get the center x value for that case
    cumsum_high = (np.cumsum(count) / len(a_flat)).astype(np.float64)
    cumsum_low = np.zeros_like(cumsum_high)
    cumsum_low[1:] = cumsum_high[:-1]
    cumsum = (cumsum_high + cumsum_low) / 2
    itp = interp1d(a_sorted, cumsum, bounds_error=True, kind='linear')

    result = np.empty_like(a, dtype=np.float32)
    result[:] = np.nan
    result[~nan_mask] = itp(a_flat)
    return result


def smooth_lowess(
    *a: ty.Union[
        ty.Tuple[np.ndarray, np.ndarray],
        ty.Tuple[np.ndarray,],
        ty.Tuple[xr.DataArray, xr.DataArray],
        ty.Tuple[xr.DataArray,],
    ],
    **kw,
) -> ty.Union[xr.DataArray, np.ndarray]:

    if len(a) == 2:
        x, y = a
        ret_slice = slice(None, None)
    elif len(a) == 1:
        y = a[0]
        x = np.arange(len(y))

        ret_slice = slice(1, None)
    else:
        raise ValueError(len(a), a)
    input_type = 'xr' if isinstance(y, xr.DataArray) else 'np'
    assert isinstance(y, (xr.DataArray, np.ndarray)), f'{type(x)} not supported'
    if input_type == 'xr':
        _y = y.values
        _x = x if isinstance(x, np.ndarray) else x.values
    else:
        _x, _y = x, y
    assert isinstance(_y, type(_x)), f'{type(_x)} is not {type(_y)}'

    res = _smooth_lowess(_x, _y, ret_slice, **kw)
    if input_type == 'np':
        return res

    ret_y = y.copy()
    if len(a) == 1:
        ret_y.data = res
        return ret_y

    ret_x = x.copy()
    ret_x.data, ret_y.data = res
    return ret_x, ret_y


smooth_lowess.__doc__ = """wrapper for statsmodels.api.nonparametric.lowess. For kwargs read\n\n: {doc}""".format(
    doc=sm.nonparametric.lowess.__doc__,
)


def _smooth_lowess(x: np.ndarray, y: np.ndarray, ret_slice: slice, **kw) -> np.ndarray:
    kw = kw.copy()
    if 'window' in kw:
        assert 'frac' not in kw, 'Provide either frac or window, not both!'
        window = kw.pop('window')
        assert window > 0 and window <= len(y)
        kw['frac'] = window / len(y)

    kw.setdefault('frac', 0.1)
    kw.setdefault('missing', 'raise')

    smoothed = sm.nonparametric.lowess(exog=x, endog=y, **kw)
    return smoothed.T[ret_slice].squeeze()


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


def _weighted_mean_array_xarray(
    data: xr.DataArray,
    weights: xr.DataArray,
) -> xr.DataArray:
    res = data * weights
    if "time" in res.dims:
        res = res.sum("lat lon".split())
        mask_time = data.isnull().all(dim="lat").all(dim="lon")
        res[mask_time] = np.nan

        mask_lat_lon = ~data.isnull().all(dim="time")
        area_mask = weights.where(mask_lat_lon)
    else:
        mask_lat_lon = ~data.isnull()
        res = res.values[mask_lat_lon].sum()
        area_mask = weights.where(mask_lat_lon)

    return res / (area_mask.sum())


def _weighted_mean_array_numpy(
    data: np.ndarray,
    weights: np.ndarray,
    has_time_dim: bool = True,
    _dtype=np.float64,
) -> np.ndarray:
    res = data * weights
    if has_time_dim:
        na_array = np.isnan(res)

        mask_time = na_array.all(axis=1).all(axis=1)

        # This is slightly confusing. We used to ignore any time step where there is a nan, but this is problematic if the nans pop in and out at a given grid cell
        # So instead we remove those grid cells.

        # Used to follow this logic: throw away all data where each cell is always nan in time, and keep data without any nan in time.
        # mask_lat_lon = ~na_array.all(axis=0)
        # no_na_vals = res[~mask_time][:, mask_lat_lon].sum(axis=1)

        # However, it's better to exclude those grid cells that are nan at least somewhere in time
        # Individual grid cells might have nans, but are at least not consistent in time.
        mask_lat_lon = ~na_array[~mask_time].any(axis=0)
        no_na_vals = np.nansum(res[~mask_time][:, mask_lat_lon], axis=1)

        res = np.zeros(len(data), dtype=_dtype)
        res[mask_time] = np.nan
        res[~mask_time] = no_na_vals
        area_mask = weights[mask_lat_lon]
    else:
        mask_lat_lon = ~np.isnan(data)
        res = np.nansum(res[mask_lat_lon])
        area_mask = weights[mask_lat_lon]

    return res / (area_mask.sum())


def _weighted_mean_array_numba(
    data: np.ndarray,
    weights: np.ndarray,
    has_time_dim: bool = True,
) -> ty.Union[float, np.ndarray]:
    if has_time_dim:
        assert len(data.shape) == 3, data.shape

    else:
        if len(data.shape) == 1:
            return _weighted_mean_1d_numba(data, weights)
        assert len(data.shape) == 2, data.shape
        return _weighted_mean_2d_numba(data, weights)
    return _weighted_mean_3d_numba(data, weights)


@numba.njit
def _weighted_mean_2d_numba(data, weights):
    tot = 0.0
    weight = 0.0
    x, y = data.shape
    for i in range(x):
        for j in range(y):
            if np.isnan(data[i][j]):
                continue
            tot += data[i][j] * weights[i][j]
            weight += weights[i][j]
    if tot == 0.0:
        return np.nan
    return tot / weight


@numba.njit
def _weighted_mean_1d_numba(data: np.ndarray, weights: np.ndarray) -> float:
    tot = 0.0
    weight = 0.0
    for i in range(len(data)):
        if np.isnan(data[i]):
            continue
        tot += data[i] * weights[i]
        weight += weights[i]
    if tot == 0.0:
        return np.nan
    return tot / weight


@numba.njit
def _weighted_mean_3d_numba(data, weights, _dtype=np.float64):
    t, x, y = data.shape
    is_nan_xy = np.zeros((x, y), dtype=np.bool_)
    weight = 0.0
    # Not sure if we should allow anything but np.float64 since you get overflows get quickly!
    tot = np.zeros(t, dtype=_dtype)

    # First, check which time steps are always nan
    for k in range(t):
        do_break = False
        for i in range(x):
            for j in range(y):
                if ~np.isnan(data[k][i][j]):
                    do_break = True
                    break
            if do_break:
                break
        is_nan_for_all_i_j = not do_break
        if is_nan_for_all_i_j:
            tot[k] = np.nan

    # Then, check which lat,lon coords are always nan
    for k in range(t):
        if np.isnan(tot[k]):
            continue
        for i in range(x):
            for j in range(y):
                if np.isnan(data[k][i][j]):
                    is_nan_xy[i][j] = True

    # Now sum all gridcells which are never nan in time, or lat+lon
    for i in range(x):
        for j in range(y):
            if is_nan_xy[i][j]:
                continue
            for k in range(t):
                if np.isnan(tot[k]):
                    continue
                tot[k] = tot[k] + data[k][i][j] * weights[i][j]
            weight += weights[i][j]
    return tot / weight


def weighted_mean_array(
    _ds: xr.Dataset,
    field: str = "std detrended",
    area_field: str = "cell_area",
    return_values: bool = True,
    method: str = "numpy",
    time_field: str = "time",
) -> ty.Union[np.ndarray, float, xr.DataArray]:
    if method == "xarray":
        res_da = _weighted_mean_array_xarray(_ds[field], _ds[area_field])
        return res_da.values if return_values else res_da

    da_sel = _ds[field]
    has_time_dim = time_field in da_sel.dims

    data = da_sel.values
    weights = _ds[area_field].values
    kw = dict(data=data, weights=weights, has_time_dim=has_time_dim)
    if method == "numba":
        res_arr = _weighted_mean_array_numba(**kw)  # type: ignore
    elif method == "numpy":
        res_arr = _weighted_mean_array_numpy(**kw)  # type: ignore
    else:
        raise ValueError(f"Unknown method {method}")

    if return_values or not has_time_dim:
        return res_arr
    res_da = xr.DataArray(res_arr, dims="time")
    res_da.attrs.update(da_sel.attrs)
    return res_da


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    # Thanks https://stackoverflow.com/a/49677241/18280620
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
