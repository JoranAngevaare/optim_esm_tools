import os
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

import optim_esm_tools as oet


class TestConsiseDataFrame(TestCase):
    def test_merge_two(self, nx=4, ny=3, is_match=(True, True)):
        with tempfile.TemporaryDirectory() as temp_dir:
            kw = dict(len_x=nx, len_y=ny, len_time=2, add_nans=False)
            names = list('abcdefg'[: len(is_match)])
            paths = [os.path.join(temp_dir, f'{x}.nc') for x in names]
            for path in paths:
                ds = oet._test_utils.minimal_xr_ds(**kw)
                print(ds['var'].shape, ds['var'].dims)
                ds['global_mask'] = (
                    oet.config.config['analyze']['lon_lat_dim'].split(','),
                    np.ones((nx, ny), bool),
                )
                ds.to_netcdf(path)
            _same = ['same'] * len(names)
            data_frame = pd.DataFrame(
                dict(
                    path=paths,
                    names=names,
                    tips=[True] * len(is_match),
                    institution_id=_same,
                    source_id=_same,
                    experiment_id=_same,
                    is_match=is_match,
                ),
            )
            concise_df = oet.analyze.concise_dataframe.ConciseDataFrame(
                data_frame,
                group=('path', 'names'),
            ).concise()
            assert len(concise_df) == len(np.unique(is_match))

    def test_merge_three(self):
        return self.test_merge_two(is_match=(True, True, False))
