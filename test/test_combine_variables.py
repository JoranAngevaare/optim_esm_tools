import tempfile
import optim_esm_tools as oet
import pandas as pd
import os
from unittest import TestCase
import numpy as np


class TestCombineVariables(TestCase):
    def test_merge_two(self, nx=4, ny=3, is_match=(True, True)):
        with tempfile.TemporaryDirectory() as temp_dir:
            kw = dict(len_x=nx, len_y=ny, len_time=2, add_nans=False)
            names = list('abcdefg'[: len(is_match)])
            paths = [os.path.join(temp_dir, f'{x}.nc') for x in names]
            for name, path in zip(names, paths):
                ds = oet._test_utils.minimal_xr_ds(**kw)
                ds = ds.rename(dict(var=name))
                assert name in ds
                ds['global_mask'] = (
                    oet.analyze.xarray_tools.default_rename_mask_dims_dict().values(),
                    np.ones((nx, ny), bool),
                )
                ds['cell_area'] = (
                    oet.config.config['analyze']['lon_lat_dim'].split(','),
                    np.ones((nx, ny), bool),
                )
                ds.attrs.update(dict(file=path, variable_id=name))
                ds.to_netcdf(path)

            merger = oet.analyze.combine_variables.VariableMerger(
                paths=[p for p, m in zip(paths, is_match) if m],
                other_paths=[p for p, m in zip(paths, is_match) if not m],
                merge_method='logical_or',
            )
            merged = merger.squash_sources()
            for n, m in zip(names, is_match):
                if m:
                    assert n in merged.data_vars

    def test_merge_three(self):
        return self.test_merge_two(is_match=(True, True, False))
