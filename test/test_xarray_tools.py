import optim_esm_tools as oet
import numpy as np


def test_remove_nan():
    ds = oet._test_utils.minimal_xr_ds(len_x=8, len_y=9, len_time=10)
    var = ds['var'].values.astype(np.float64)
    var[:3][:] = np.nan
    ds['var'] = (ds['var'].dims, var)
    time = ds['time'].values.astype(np.float64)
    time[:3] = np.nan
    ds['time'] = time
    oet.analyze.xarray_tools._remove_any_none_times(ds['var'], 'time')
    try:
        oet.analyze.xarray_tools._remove_any_none_times(ds['var'], 'time', drop=False)
    except AssertionError:
        # This one is mentioned in the comment of _remove_any_none_times
        pass
