import contextlib
import optim_esm_tools as oet
import unittest
import numpy as np
import xarray as xr

from hypothesis import given
from hypothesis.extra.numpy import arrays


def test_remove_nan():
    ds = oet._test_utils.minimal_xr_ds(len_x=8, len_y=9, len_time=10)
    var = ds['var'].values.astype(np.float64)
    var[:3][:] = np.nan
    ds['var'] = (ds['var'].dims, var)
    time = ds['time'].values.astype(np.float64)
    time[:3] = np.nan
    ds['time'] = time
    oet.analyze.xarray_tools._remove_any_none_times(ds['var'], 'time')
    with contextlib.suppress(AssertionError):
        oet.analyze.xarray_tools._remove_any_none_times(ds['var'], 'time', drop=False)


def test_global_mask():
    ds = oet._test_utils.minimal_xr_ds(len_x=8, len_y=9, len_time=10)
    ds['var'].data = np.random.randint(1, 10, size=ds['var'].shape)
    mask = ds['var'] > 5

    renamed_mask = oet.analyze.xarray_tools.rename_mask_coords(mask.copy())
    assert mask.dims != renamed_mask.dims, (
        mask.dims,
        renamed_mask.dims,
    )

    rev_renamed_mask = oet.analyze.xarray_tools.reverse_name_mask_coords(
        renamed_mask.copy(),
    )
    assert mask.dims == rev_renamed_mask.dims, (
        mask.dims,
        rev_renamed_mask.dims,
    )


class TestDrop(unittest.TestCase):
    def test_drop_by_mask(self):
        ds = oet._test_utils.minimal_xr_ds(len_x=8, len_y=9, len_time=10)
        ds['var'].data = np.random.randint(1, 10, size=ds['var'].shape)
        mask = ds['var'].isel(time=0).drop_vars('time') > 5
        kw = dict(
            data_set=ds,
            da_mask=mask,
            masked_dims=list(mask.dims),
            drop=True,
            keep_keys=None,
        )
        ds['cell_area'] = mask.astype(np.int64)
        dropped_nb = oet.analyze.xarray_tools.mask_xr_ds(
            **kw,
            drop_method='numba',
        )
        dropped_xr = oet.analyze.xarray_tools.mask_xr_ds(
            **kw,
            drop_method='xarray',
        )
        v_xr = dropped_xr['var'].values
        v_nb = dropped_nb['var'].values
        self.assertTrue(np.array_equal(v_xr[~np.isnan(v_xr)], v_nb[~np.isnan(v_nb)]))
        self.assertTrue(np.array_equal(np.isnan(v_xr), np.isnan(v_nb)))
        with self.assertRaises(ValueError):
            oet.analyze.xarray_tools.mask_xr_ds(
                **kw,
                drop_method='numpy_or_somthing',
            )





@given(arrays(np.float16, shape=(2, 100)))
def test_smooth_lowess_2d(a):
    x, y = a
    x_da = xr.DataArray(x)
    y_da = xr.DataArray(y)

    try:
        res = oet.analyze.tools.smooth_lowess(x, y)
    except ValueError as e:
        if np.any(np.isnan(y) | np.isnan(x) | ~np.isfinite(x) | ~np.isfinite(y)):
            # This is fine, the data is not in the proper format!
            return
        raise e
    assert all(isinstance(z, np.ndarray) for z in res)
    try:
        res_da = oet.analyze.tools.smooth_lowess(x_da, y_da)
    except ValueError as e:
        if np.any(np.isnan(y) | np.isnan(x) | ~np.isfinite(x) | ~np.isfinite(y)):
            return
        raise e

    assert all(isinstance(z, xr.DataArray) for z in res_da)

    assert np.array(res).shape == a.shape

    assert np.array_equal(res[0], res_da[0].values)
    assert np.array_equal(res[1], res_da[1].values)


@given(arrays(np.float16, shape=(100)))
def test_smooth_lowess_1d(y):
    y_da = xr.DataArray(y)

    try:
        res = oet.analyze.tools.smooth_lowess(y)
    except ValueError as e:
        if np.any(np.isnan(y) | ~np.isfinite(y)):
            # This is fine, the data is not in the proper format!
            return
        raise e
    assert isinstance(res, np.ndarray)
    try:
        res_da = oet.analyze.tools.smooth_lowess(y_da)
    except ValueError as e:
        if np.any(np.isnan(y) | ~np.isfinite(y)):
            return
        raise e

    assert isinstance(res_da, xr.DataArray)

    assert res.shape == y.shape
    assert res_da.shape == y.shape

    assert np.array_equal(res, res_da.values)

def test_set_time_int():
    ds = oet._test_utils.minimal_xr_ds(len_x=2, len_y=2, len_time=2)
    assert not isinstance(ds['time'].values[0], int)
    oet.analyze.xarray_tools.set_time_int(ds)
    assert isinstance(ds['time'].values[0], int)
    
    ds2 = oet.analyze.xarray_tools.set_time_int(ds)
    assert isinstance(ds2['time'].values[0], int)
    