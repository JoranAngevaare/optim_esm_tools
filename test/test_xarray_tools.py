import contextlib

import numpy as np
import pandas as pd
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

class TestYearlyAverage(unittest.TestCase):
    """From ChatGPT"""
    def setUp(self):
        # Create a time range with monthly data over 3 years
        time = pd.date_range('2000-01-01', '2002-12-31', freq='M')
        time_bnds = xr.DataArray(
            np.array([pd.date_range(start, periods=2, freq='MS') for start in time]),
            dims=['time', 'bnds']
        )

        # Create mock data variables (tas, pr) resembling CMIP6 data
        tas_data = np.random.rand(len(time), 2, 2) * 300  # temperature in K
        pr_data = np.random.rand(len(time), 2, 2) * 10  # precipitation in mm/day

        # Create a dataset
        self.ds = xr.Dataset({
            'tas': (('time', 'lat', 'lon'), tas_data),
            'pr': (('time', 'lat', 'lon'), pr_data),
            'time_bnds': (('time', 'bnds'), time_bnds)
        }, coords={
            'time': time,
            'lat': [10.0, 20.0],
            'lon': [30.0, 40.0]
        })

    def test_yearly_average(self):
        # Run the yearly averaging function
        ds_yearly = yearly_average(self.ds, time_dim='time')

        # Check if the output dataset has a 'year' dimension instead of 'time'
        self.assertIn('year', ds_yearly.dims)
        self.assertNotIn('time', ds_yearly.dims)
        
        # Check that the shape of the yearly averaged data is correct
        expected_shape = (3, 2, 2)  # 3 years, 2 lat, 2 lon
        self.assertEqual(ds_yearly['tas'].shape, expected_shape)
        self.assertEqual(ds_yearly['pr'].shape, expected_shape)
        
        # Additional checks could include verifying the correctness of the values
        # But for simplicity, we're focusing on dimensional checks here.

    def test_skip_non_numeric_variable(self):
        # Add a non-numeric variable
        self.ds['string_var'] = (('time',), np.array(['a'] * len(self.ds['time'])))

        # Run the yearly averaging function
        ds_yearly = yearly_average(self.ds, time_dim='time')

        # Ensure the non-numeric variable was skipped
        self.assertNotIn('string_var', ds_yearly)

    def test_with_time_bounds(self):
        # Check if time bounds are correctly handled (in the setup data)
        self.ds['time_bounds'] = self.ds['time_bnds']  # Alias for testing

        # Run the yearly averaging function
        ds_yearly = yearly_average(self.ds, time_dim='time')

        # Check if the output dataset has a 'year' dimension instead of 'time'
        self.assertIn('year', ds_yearly.dims)
        self.assertNotIn('time', ds_yearly.dims)
        self.assertNotIn('time_bnds', ds_yearly)
        self.assertNotIn('time_bounds', ds_yearly)
