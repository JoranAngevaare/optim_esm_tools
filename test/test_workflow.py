# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
import matplotlib.pyplot as plt
import subprocess
from optim_esm_tools._test_utils import synda_test_available, get_example_data_loc


@unittest.skipIf(not synda_test_available(), 'synda data not available')
class TestMapMaker(unittest.TestCase):
    name_merged = 'test_merged.nc'

    def from_amon_to_ayear(self):
        if os.path.exists(self.ayear_file):
            return

        os.makedirs(os.path.split(self.ayear_file)[0], exist_ok=1)
        import cdo

        cmd = f'cdo yearmonmean {self.amon_file} {self.ayear_file}'
        print(cmd)
        cdo.Cdo().yearmonmean(input=self.amon_file, output=self.ayear_file)
        assert os.path.exists(self.ayear_file), self.ayear_file

    @classmethod
    def setUpClass(cls):
        cls.base = os.path.join(os.environ['ST_HOME'], 'data')
        cls.amon_file = get_example_data_loc()
        cls.ayear_file = os.path.join(
            os.path.split(cls.amon_file.replace('Amon', 'AYear'))[0], cls.name_merged
        )

    def tearDown(self) -> None:
        assert os.path.exists(self.ayear_file)
        return super().tearDown()

    def setUp(self):
        self.from_amon_to_ayear()
        assert os.path.exists(self.ayear_file)
        super().setUp()

    def test_read_data(self):
        dataset = oet.analyze.io.load_glob(self.ayear_file)

    def test_make_map(self):
        data_set = oet.analyze.cmip_handler.read_ds(
            os.path.split(self.ayear_file)[0], _file_name=self.name_merged
        )
        oet.plotting.map_maker.MapMaker(data_set=data_set).plot_all(2)
        plt.clf()

    def test_map_maker_time_series(self, **kw):
        data_set = oet.analyze.cmip_handler.read_ds(
            os.path.split(self.ayear_file)[0], _file_name=self.name_merged
        )
        oet.plotting.map_maker.MapMaker(data_set=data_set).time_series(**kw)
        plt.clf()

    def test_map_maker_time_series_only_running_mean(self):
        self.test_map_maker_time_series(only_rm=True)

    def test_date_out_of_range(self):
        with self.assertRaises(oet.analyze.pre_process.NoDataInTimeRangeError):
            oet.read_ds(
                os.path.split(self.ayear_file)[0],
                _file_name=self.name_merged,
                min_time=(3000, 1, 1),
            )

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.ayear_file)
        return super().tearDownClass()


class Units(unittest.TestCase):
    def test_apply_relative_units(self, unit='relative', refresh=False):
        from optim_esm_tools._test_utils import year_means, get_file_from_pangeo

        data_set = oet.analyze.cmip_handler.read_ds(
            os.path.split(
                year_means(
                    get_file_from_pangeo('ssp585', refresh=refresh), refresh=refresh
                )
            )[0],
            condition_kwargs=dict(unit=unit),
            _file_name='test_merged.nc',
        )
        mm = oet.plotting.map_maker.MapMaker(data_set=data_set)

    def test_apply_std_unit(self):
        self.test_apply_relative_units(unit='std')
