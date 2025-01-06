import os
import tempfile
from unittest import main
from unittest import TestCase

import cftime
import numpy as np
import xarray as xr
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

import optim_esm_tools as oet
from optim_esm_tools.analyze.combine_variables import VariableMerger


class TestCombineVariables(TestCase):
    def test_merge_two(self, nx=5, ny=10, is_match=(True, True), **plot_kw):
        with tempfile.TemporaryDirectory() as temp_dir:
            setup_kw = dict(len_x=nx, len_y=ny, len_time=20, add_nans=False)
            names = list('abcefg')[: len(is_match)]
            paths = [os.path.join(temp_dir, f'{x}.nc') for x in names]
            post_path = []
            for name, path in zip(names, paths):
                ds = oet._test_utils.complete_ds(**setup_kw)
                ds = ds.rename(dict(var=name))
                assert name in ds

                ds.attrs.update(dict(file=path, variable_id=name))
                ds.to_netcdf(path)
                head, tail = os.path.split(path)
                post_ds = oet.read_ds(head, _file_name=tail, _skip_folder_info=True)
                post_path.append(post_ds.attrs['file'])

            merger = oet.analyze.combine_variables.VariableMerger(
                paths=[p for p, m in zip(post_path, is_match) if m],
                other_paths=[p for p, m in zip(post_path, is_match) if not m],
                merge_method='logical_or',
            )
            merged = merger.squash_sources()
            for n, m in zip(names, is_match):
                if m:
                    assert n in merged.data_vars
            oet.analyze.combine_variables.change_plt_table_height()
            merger.make_fig(merged, **plot_kw, add_history=False)

            assert np.array_equal(
                [t.year for t in merger.data_set['time'].values],
                [t.year for t in post_ds['time'].values],
            )
            return merger

    def test_merge_three(self):
        merger = self.test_merge_two(is_match=(True, True, False))
        assert merger.other_paths

    def test_merge_w_hist(self):
        self.test_merge_two(add_histograms=True)


class TestVariableMerger(TestCase):
    """This unittest was written using the help of CHATGPT, although it
    required a fair amount of optimization."""

    first_year = 2000

    def create_dummy_dataset(
        self,
        length,
        nx=5,
        ny=20,
        add_out_of_order_variable=False,
    ):
        time_values = [
            cftime.DatetimeNoLeap(self.first_year, 1, i + 1) for i in range(length)
        ]
        lat_values = np.linspace(-90, 90, ny)
        lon_values = np.linspace(0, 360, nx)
        variable1 = (
            ('time', 'lat', 'lon'),
            np.arange(length * ny * nx).reshape(length, ny, nx),
        )
        variable2 = (
            ('time', 'lat', 'lon'),
            np.random.rand(length, ny, nx),
        )
        variable3 = (
            ('time', 'lat', 'lon'),
            np.random.randint(0, 2, size=(length, ny, nx)),
        )

        # Create global mask as a boolean array
        common_mask = (('lat', 'lon'), np.random.choice([True, False], size=(ny, nx)))
        cell_area = (('lat', 'lon'), np.arange(ny * nx).reshape(ny, nx))

        dummy_data = {
            'variable1': variable1,
            'variable2': variable2,
            'variable3': variable3,
            'common_mask': common_mask,
            'cell_area': cell_area,
        }

        coords = {
            'time': time_values,
            'lat': lat_values,
            'lon': lon_values,
        }

        dataset = xr.Dataset(data_vars=dummy_data, coords=coords)

        variables = list(set(dummy_data) - {'common_mask', 'cell_area'})
        if add_out_of_order_variable:
            variables += ['offset_variable1']
        dataset.attrs['variables'] = variables
        dataset.attrs['source_files'] = ['' for _ in dataset.attrs['variables']]

        # Add a running mean with 10 samples to each variable while considering the new dimensions
        _ma_window = oet.config.config['analyze']['moving_average_years']
        for var_name in dummy_data:
            if var_name in ['cell_area', 'common_mask']:
                continue
            rm = np.zeros_like(dataset[var_name].values, dtype=np.float16)
            rm[:] = np.nan
            for lat_idx in range(ny):
                for lon_idx in range(nx):
                    running_mean = np.convolve(
                        dataset[var_name][:, lat_idx, lon_idx],
                        np.ones(10) / 10,
                        mode='valid',
                    )
                    rm[5:-4, lat_idx, lon_idx] = running_mean
            dataset[f'{var_name}_run_mean_{_ma_window}'] = (('time', 'lat', 'lon'), rm)
        # Create variables with offset time values
        if add_out_of_order_variable:
            offset_time1 = [
                cftime.DatetimeNoLeap(2000, 1, i + 5) for i in range(length)
            ]
            off_dims, offset_variable1 = (
                ('time', 'lat', 'lon'),
                np.random.rand(length, ny, nx),
            )
            da_off = xr.DataArray(
                data=offset_variable1,
                dims=off_dims,
                coords={
                    'time': offset_time1,
                    'lat': lat_values,
                    'lon': lon_values,
                },
            )

            dataset['offset_variable1'] = da_off
            dataset[f'offset_variable1_run_mean_{_ma_window}'] = da_off
        return dataset

    @settings(max_examples=10, deadline=None)
    @given(
        dummy_dataset_length=st.integers(min_value=11, max_value=20),
        random_seed=st.integers(min_value=1, max_value=1000),
        add_out_of_order_variable=st.booleans(),
    )
    def test_combine_masks(
        self,
        dummy_dataset_length,
        random_seed,
        add_out_of_order_variable,
    ):
        np.random.seed(random_seed)
        dummy_dataset = self.create_dummy_dataset(
            dummy_dataset_length,
            add_out_of_order_variable=add_out_of_order_variable,
        )
        merger = VariableMerger(data_set=dummy_dataset)
        merger.make_fig(add_summary=False, add_history=False)
        oet.plotting.plot._show(False)
        assert merger.data_set.equals(dummy_dataset)
        assert merger.mask_paths is None
        assert merger.merge_method == 'logical_or'
        assert np.array_equal(
            [t.year for t in merger.data_set['time'].values],
            [t.year for t in dummy_dataset['time'].values],
        )

    @settings(max_examples=3, deadline=None)
    @given(
        ds_length_0=st.integers(min_value=11, max_value=20),
        ds_length_1=st.integers(min_value=11, max_value=20),
        random_seed=st.integers(min_value=1, max_value=1000),
    )
    def test_history_merging(self, ds_length_0, ds_length_1, random_seed):
        np.random.seed(random_seed)
        ds_0 = self.create_dummy_dataset(
            ds_length_0,
            add_out_of_order_variable=False,
        )
        ds_1 = self.create_dummy_dataset(
            ds_length_1,
            add_out_of_order_variable=False,
        )
        ds_1['time'] = [
            cftime.DatetimeNoLeap(self.first_year - i - 1, 1, i + 1)
            for i, _ in enumerate(ds_1['time'])
        ][::-1]
        merger = VariableMerger(data_set=ds_0)
        merger.make_fig(add_history=True, _historical_ds=ds_1, add_summary=False)
        oet.plotting.plot._show(False)

    @settings(max_examples=3, deadline=None)
    @given(
        dummy_dataset_length=st.integers(min_value=11, max_value=20),
        random_seed=st.integers(min_value=1, max_value=1000),
        add_out_of_order_variable=st.booleans(),
    )
    def test_merge_independent(
        self,
        dummy_dataset_length,
        random_seed,
        add_out_of_order_variable,
    ):
        np.random.seed(random_seed)
        dummy_dataset = self.create_dummy_dataset(
            dummy_dataset_length,
            add_out_of_order_variable=add_out_of_order_variable,
        )
        merger = VariableMerger(data_set=dummy_dataset, merge_method='independent')
        merger.make_fig(add_summary=False, add_history=False)
        oet.plotting.plot._show(False)
        assert merger.data_set.equals(dummy_dataset)
        assert merger.mask_paths is None
        assert merger.merge_method == 'independent'


if __name__ == '__main__':
    main()
