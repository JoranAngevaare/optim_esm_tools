import numpy as np
from scipy.interpolate import interp1d
import statsmodels.api as sm
import typing as ty
import xarray as xr


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
    doc=sm.nonparametric.lowess.__doc__
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
