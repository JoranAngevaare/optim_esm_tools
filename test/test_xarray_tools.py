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
    """From ChatGPT."""
    def setUp(self):
        # Create a time range with monthly data over 3 years
        self.time = pd.date_range('2000-01-01', '2002-12-31', freq='M')
        self.lat = [10.0, 20.0]
        self.lon = [30.0, 40.0]

        # Create mock data variables (tas, pr) resembling CMIP6 data
        self.tas_data = np.random.rand(len(self.time), len(self.lat), len(self.lon)) * 300  # temperature in K
        self.pr_data = np.random.rand(len(self.time), len(self.lat), len(self.lon)) * 10  # precipitation in mm/day

    def create_dataset(self, with_time_bounds=True):
        if with_time_bounds:
            # Create time bounds assuming each time point represents a full month
            time_bnds = xr.DataArray(
                np.array([pd.date_range(start, periods=2, freq='MS') for start in self.time]),
                dims=['time', 'bnds']
            )

            return xr.Dataset({
                'tas': (('time', 'lat', 'lon'), self.tas_data),
                'pr': (('time', 'lat', 'lon'), self.pr_data),
                'time_bnds': (('time', 'bnds'), time_bnds)
            }, coords={
                'time': self.time,
                'lat': self.lat,
                'lon': self.lon
            })
        else:
            return xr.Dataset({
                'tas': (('time', 'lat', 'lon'), self.tas_data),
                'pr': (('time', 'lat', 'lon'), self.pr_data)
            }, coords={
                'time': self.time,
                'lat': self.lat,
                'lon': self.lon
            })

    def test_yearly_average_with_time_bounds(self):
        ds = self.create_dataset(with_time_bounds=True)
        ds_yearly = yearly_average(ds, time_dim='time')

        # Check if the output dataset has a 'year' dimension instead of 'time'
        self.assertIn('year', ds_yearly.dims)
        self.assertNotIn('time', ds_yearly.dims)
        
        # Check that the shape of the yearly averaged data is correct
        expected_shape = (3, len(self.lat), len(self.lon))  # 3 years, 2 lat, 2 lon
        self.assertEqual(ds_yearly['tas'].shape, expected_shape)
        self.assertEqual(ds_yearly['pr'].shape, expected_shape)

    def test_yearly_average_without_time_bounds(self):
        ds = self.create_dataset(with_time_bounds=False)
        ds_yearly = yearly_average(ds, time_dim='time')

        # Check if the output dataset has a 'year' dimension instead of 'time'
        self.assertIn('year', ds_yearly.dims)
        self.assertNotIn('time', ds_yearly.dims)
        
        # Check that the shape of the yearly averaged data is correct
        expected_shape = (3, len(self.lat), len(self.lon))  # 3 years, 2 lat, 2 lon
        self.assertEqual(ds_yearly['tas'].shape, expected_shape)
        self.assertEqual(ds_yearly['pr'].shape, expected_shape)

    def test_skip_non_numeric_variable_with_time_bounds(self):
        ds = self.create_dataset(with_time_bounds=True)
        ds['string_var'] = (('time',), np.array(['a'] * len(ds['time'])))

        ds_yearly = yearly_average(ds, time_dim='time')

        # Ensure the non-numeric variable was skipped
        self.assertNotIn('string_var', ds_yearly)

    def test_skip_non_numeric_variable_without_time_bounds(self):
        ds = self.create_dataset(with_time_bounds=False)
        ds['string_var'] = (('time',), np.array(['a'] * len(ds['time'])))

        ds_yearly = yearly_average(ds, time_dim='time')

        # Ensure the non-numeric variable was skipped
        self.assertNotIn('string_var', ds_yearly)

    def test_with_and_without_time_bounds(self):
        """Combined test to check consistency between datasets with and without time bounds."""
        ds_with_bounds = self.create_dataset(with_time_bounds=True)
        ds_without_bounds = self.create_dataset(with_time_bounds=False)

        ds_yearly_with_bounds = yearly_average(ds_with_bounds, time_dim='time')
        ds_yearly_without_bounds = yearly_average(ds_without_bounds, time_dim='time')

        # Check that the yearly averages are approximately equal
        xr.testing.assert_allclose(ds_yearly_with_bounds['tas'], ds_yearly_without_bounds['tas'])
        xr.testing.assert_allclose(ds_yearly_with_bounds['pr'], ds_yearly_without_bounds['pr'])

