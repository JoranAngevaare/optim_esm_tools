import tempfile
import optim_esm_tools as oet
import pandas as pd
import os
from unittest import TestCase
import numpy as np


class TestConsiseDataFrame(TestCase):
    def test_merge_two(self, nx=4, ny=3):
        with tempfile.TemporaryDirectory() as temp_dir:
            kw = dict(len_x=nx, len_y=ny, len_time=2, add_nans=False)
            names = list('ab')
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
                    tips=[True, True],
                    institution_id=_same,
                    source_id=_same,
                    experiment_id=_same,
                )
            )
            concise_df = oet.analyze.concise_dataframe.ConciseDataFrame(
                data_frame, group=('path', 'names')
            ).concise()
            assert len(concise_df) == 1
