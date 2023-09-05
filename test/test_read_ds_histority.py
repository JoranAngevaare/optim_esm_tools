import os
import tempfile
from unittest import TestCase

import optim_esm_tools as oet


class TestReadDs(TestCase):
    def test_read_ds_historical(self, nx=5, ny=5, len_time=20, start_year=2015):
        with tempfile.TemporaryDirectory() as temp_dir:
            kw = dict(len_x=nx, len_y=ny, len_time=len_time, add_nans=False)
            names = ['ssp', 'historical']
            starts = [start_year, start_year - len_time]
            paths = [os.path.join(temp_dir, f'{x}.nc') for x in names]
            for start, path in zip(starts, paths):
                ds = oet._test_utils.complete_ds(start_year=start, **kw)

                ds.attrs.update(dict(file=path))
                ds.to_netcdf(path)

            path_ssp, path_his = paths
            head, tail = os.path.split(path_ssp)
            ds = oet.read_ds(head, _file_name=tail, _skip_folder_info=True)
            assert len(ds['time']) == len_time

            with self.assertRaises(FileNotFoundError):
                oet.read_ds(
                    head,
                    add_history=True,
                    _file_name=tail,
                    _skip_folder_info=True,
                )
            with self.assertRaises(ValueError):
                oet.read_ds(
                    head,
                    add_history=False,
                    _file_name=tail,
                    _skip_folder_info=True,
                    _historical_path=path_his,
                )
            ds_with_hist = oet.read_ds(
                head,
                add_history=True,
                _file_name=tail,
                _skip_folder_info=True,
                _historical_path=path_his,
            )
            assert len(ds_with_hist['time']) == len_time * 2
            assert ds_with_hist['time'].values[0].year == start_year - len_time
