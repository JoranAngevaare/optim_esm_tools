import numpy as np
from scipy.interpolate import interp1d


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
