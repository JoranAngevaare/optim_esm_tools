# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
import matplotlib.pyplot as plt
import subprocess
from optim_esm_tools._test_utils import synda_test_available, get_example_data_loc


@unittest.skipIf(not synda_test_available(), 'synda data not available')
class TestMapMaker(unittest.TestCase):
    # example_data_set = oet._test_utils.EXAMPLE_DATA_SET

    def from_amon_to_ayear(self):
        if os.path.exists(self.ayear_file):
            return

        os.makedirs(os.path.split(self.ayear_file)[0], exist_ok=1)
        # Doesn't work?
        # cdo.Cdo().yearmonmean(self.amon_file, self.ayear_file)
        cmd = f'cdo yearmonmean {self.amon_file} {self.ayear_file}'
        print(cmd)
        subprocess.call(cmd, shell=True)
        assert os.path.exists(self.ayear_file), self.ayear_file

    @classmethod
    def setUpClass(cls):
        cls.base = os.path.join(os.environ['ST_HOME'], 'data')
        cls.amon_file = get_example_data_loc()
        cls.ayear_file = os.path.join(
            os.path.split(cls.amon_file.replace('Amon', 'AYear'))[0], 'merged.nc'
        )

    def setUp(self):
        self.from_amon_to_ayear()
        super().setUp()

    def test_read_data(self):
        data_set = oet.synda_files.format_synda.load_glob(self.ayear_file)

    def test_make_map(self):
        data_set = oet.analyze.cmip_handler.read_ds(os.path.split(self.ayear_file)[0])
        oet.analyze.cmip_handler.MapMaker(data_set=data_set).plot_all(2)
        plt.clf()

    def test_map_maker_time_series(self):
        data_set = oet.analyze.cmip_handler.read_ds(os.path.split(self.ayear_file)[0])
        oet.analyze.cmip_handler.MapMaker(data_set=data_set).time_series()
        plt.clf()

    def test_apply_relative_units(self, unit='relative'):
        data_set = oet.analyze.cmip_handler.read_ds(os.path.split(self.ayear_file)[0])
        mm = oet.analyze.cmip_handler.MapMaker(data_set=data_set)
        from immutabledict import immutabledict
        from functools import partial

        mm.conditions = immutabledict(
            {
                'i ii iii iv v vi vii viii ix x'.split()[i]: props
                for i, props in enumerate(
                    zip(
                        (str(i) for i in range(4)),
                        [
                            partial(
                                oet.analyze.tipping_criteria.running_mean_diff,
                                unit=unit,
                            ),
                            partial(
                                oet.analyze.tipping_criteria.running_mean_std, unit=unit
                            ),
                            partial(
                                oet.analyze.tipping_criteria.max_change_xyr, unit=unit
                            ),
                            partial(
                                oet.analyze.tipping_criteria.max_derivative, unit=unit
                            ),
                        ],
                    )
                )
            }
        )
        for i in mm.conditions.keys():
            getattr(mm, i)

    def test_apply_std_unit(self):
        self.test_apply_relative_units(unit='std')

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.ayear_file)
        return super().tearDownClass()
