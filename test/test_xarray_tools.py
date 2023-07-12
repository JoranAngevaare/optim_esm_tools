import optim_esm_tools as oet
import numpy as np


def test_remove_nan():
    ds = oet._test_utils.minimal_xr_ds(len_x=8, len_y=9, len_time=10)
    time = ds['time'].values.astype(np.float64)
    time[:3] = np.nan
    ds['time'] = time

    oet.analyze.xarray_tools._remove_any_none_times(ds['var'], 'time')
    oet.analyze.xarray_tools._remove_any_none_times(ds['var'], 'time', drop=False)
