import os
import tempfile
from unittest import TestCase

import numpy as np

import optim_esm_tools as oet


class TestDirect(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._remap_orig = oet.config.config['analyze']['regrid_to']
        oet.config.config.read_dict({'analyze': {'regrid_to': 'n3'}})

    @classmethod
    def tearDownClass(cls):
        oet.config.config.read_dict({'analyze': {'regrid_to': cls._remap_orig}})

    def test_direct_cal(
        self,
        nx=5,
        ny=20,
        len_time=20,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            kw = dict(len_x=nx, len_y=ny, len_time=len_time, add_nans=False)
            names = ['ssp', 'picontrol']
            paths = [os.path.join(temp_dir, f'{x}.nc') for x in names]
            post_path = {}
            for name, path in zip(names, paths):
                ds = oet._test_utils.complete_ds(**kw)
                ds.attrs.update(dict(file=path))
                ds.to_netcdf(path)
                head, tail = os.path.split(path)
                post_ds = oet.read_ds(head, _file_name=tail, _skip_folder_info=True)
                post_path[name] = post_ds.attrs['file']

            ds = oet.load_glob(post_path['ssp'])
            ds_masked = ds.copy()
            l_time, l_lat, l_lon = ds_masked['var'].shape
            ds_masked['var'].data = np.repeat(
                [np.arange(l_lat * l_lon).reshape(l_lat, l_lon)],
                l_time,
                axis=0,
            )
            mask = ds_masked['var'].isel(time=0) > 0.25 * l_lat * l_lon

            ds_masked = oet.analyze.xarray_tools.mask_to_reduced_dataset(ds, mask)
            oet.analyze.direct_statistics.direct_test(
                ds_masked,
                _ds_global=ds,
                _ds_hist=oet.load_glob(post_path['picontrol']),
                over_ride_thresholds=dict(
                    max_jump=1,
                    p_dip=1,
                    p_symmetry=1,
                    n_breaks=0,
                    n_std_global=0,
                ),
            )
