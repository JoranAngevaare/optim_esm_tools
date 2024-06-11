import contextlib

import numpy as np

import optim_esm_tools as oet
import unittest


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
        mask = ds['var'].isel(time=0).drop('time') > 5
        kw = dict(
            data_set=ds,
            da_mask=mask,
            masked_dims=list(mask.dims),
            drop=True,
            keep_keys=None,
        )
        dropped_xr = oet.analyze.xarray_tools.mask_xr_ds(
            **kw,
            drop_method='xarray',
        )
        dropped_nb = oet.analyze.xarray_tools.mask_xr_ds(
            **kw,
            drop_method='numba',
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
