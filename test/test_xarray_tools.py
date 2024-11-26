import contextlib
import numpy as np
import pandas as pd
import optim_esm_tools as oet
import unittest
import xarray as xr
import cftime

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.stats import percentileofscore
from optim_esm_tools.analyze.xarray_tools import yearly_average


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

    def setUp(self):
        self.lat = [10.0, 20.0]
        self.lon = [30.0, 40.0]

    def create_dataset(self, with_time_bounds=True, use_cftime=False):
        if use_cftime:
            time = xr.cftime_range(
                '2000-01-01',
                '2002-12-31',
                freq='M',
                calendar='noleap',
            )
        else:
            time = pd.date_range('2000-01-01', '2002-12-31', freq='M')

        if with_time_bounds:
            if use_cftime:
                time_bnds = xr.DataArray(
                    np.array([[time[i], time[i + 1]] for i in range(len(time) - 1)]),
                    dims=['time', 'bnds'],
                )
                # Now, we have make the time stamps in the middle of each time bound
                time = np.array([t[0] + (t[1] - t[0]) / 2 for t in time_bnds.values])
                assert len(time_bnds) == len(time)
            else:
                time_bnds = xr.DataArray(
                    np.array(
                        [
                            [pd.Timestamp(t), pd.Timestamp(t + pd.DateOffset(months=1))]
                            for t in time
                        ],
                    ),
                    dims=['time', 'bnds'],
                )

        tas_data = np.random.rand(len(time), len(self.lat), len(self.lon)) * 300
        pr_data = np.random.rand(len(time), len(self.lat), len(self.lon)) * 10
        ds = xr.Dataset(
            {
                'tas': (('time', 'lat', 'lon'), tas_data),
                'pr': (('time', 'lat', 'lon'), pr_data),
            },
            coords={
                'time': time,
                'lat': self.lat,
                'lon': self.lon,
            },
        )
        if with_time_bounds:
            ds['time_bnds'] = time_bnds
        return ds

    def test_yearly_average_with_time_bounds_and_cftime(self):
        ds = self.create_dataset(with_time_bounds=True, use_cftime=True)
        ds_yearly = yearly_average(ds, time_dim='time')

        self.assertIn('year', ds_yearly.dims)
        self.assertNotIn('time', ds_yearly.dims)

        expected_shape = (3, len(self.lat), len(self.lon))  # 3 years, 2 lat, 2 lon
        self.assertEqual(ds_yearly['tas'].shape, expected_shape)
        self.assertEqual(ds_yearly['pr'].shape, expected_shape)

    def test_yearly_average_without_time_bounds_and_cftime(self):
        ds = self.create_dataset(with_time_bounds=False, use_cftime=True)
        ds_yearly = yearly_average(ds, time_dim='time')

        self.assertIn('year', ds_yearly.dims)
        self.assertNotIn('time', ds_yearly.dims)

        expected_shape = (3, len(self.lat), len(self.lon))  # 3 years, 2 lat, 2 lon
        self.assertEqual(ds_yearly['tas'].shape, expected_shape)
        self.assertEqual(ds_yearly['pr'].shape, expected_shape)

    def test_skip_non_numeric_variable_with_time_bounds_and_cftime(self):
        ds = self.create_dataset(with_time_bounds=True, use_cftime=True)
        ds['string_var'] = (('time',), np.array(['a'] * len(ds['time'])))

        ds_yearly = yearly_average(ds, time_dim='time')

        self.assertNotIn('string_var', ds_yearly)

    def test_with_and_without_time_bounds_and_cftime(self):
        ds_with_bounds = self.create_dataset(with_time_bounds=True, use_cftime=True)
        ds_without_bounds = ds_with_bounds.copy()
        del ds_without_bounds['time_bnds']

        ds_yearly_with_bounds = yearly_average(ds_with_bounds, time_dim='time')
        ds_yearly_without_bounds = yearly_average(ds_without_bounds, time_dim='time')

        xr.testing.assert_allclose(
            ds_yearly_with_bounds['tas'],
            ds_yearly_without_bounds['tas'],
            rtol=1 / 29,  # Max one day/month off
        )
        xr.testing.assert_allclose(
            ds_yearly_with_bounds['pr'],
            ds_yearly_without_bounds['pr'],
            rtol=1 / 29,  # Max one day/month off
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
    ds = oet._test_utils.complete_ds(len_x=2, len_y=2, len_time=2)
    assert not isinstance(ds['time'].values[0], int)
    oet.analyze.xarray_tools.set_time_int(ds)
    assert isinstance(ds['time'].values[0], np.integer)

    ds2 = oet._test_utils.complete_ds(len_x=2, len_y=2, len_time=2)
    ds3 = oet.analyze.xarray_tools.set_time_int(ds2)
    assert isinstance(ds3['time'].values[0], np.integer)
