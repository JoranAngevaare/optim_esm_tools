# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools._test_utils
from optim_esm_tools.analyze import region_finding
import tempfile
import pytest
import os

class Work(unittest.TestCase):
    # example_data_set = oet._test_utils.EXAMPLE_DATA_SET
    def test(self):
        for data_name in ['ssp585', 'piControl']:
            self.get_path(data_name)

    
    @staticmethod
    def get_path(data_name):
        path = optim_esm_tools._test_utils.get_file_from_pangeo(data_name)
        year_path = optim_esm_tools._test_utils.year_means(path)
        assert year_path
        assert os.path.exists(year_path)
        return year_path
    
    @pytest.mark.paramerize('make', ['region_finding', 'Percentiles', 'PercentilesHistory'])
    def test_build_plots(self, make='MaxRegion',):
        cls = getattr(region_finding, make)
        with tempfile.TemporaryDirectory() as temp_dir:
            print(make)
            save_kw = dict(
    save_in = temp_dir,
    sub_dir = None,
    file_types=('png', 'pdf'),
    skip= False,
)   
            head, tail = os.path.split(self.get_path('ssp585'))        
            r=cls(path=head, read_ds_kw=dict(_file_name=tail), transform=True, save_kw=save_kw, extra_opt=dict())
            r.show=False
            r.workflow()
    # def from_amon_to_ayear(self):
    #     if os.path.exists(self.ayear_file):
    #         return

    #     os.makedirs(os.path.split(self.ayear_file)[0], exist_ok=1)
    #     # Doesn't work?
    #     # cdo.Cdo().yearmonmean(self.amon_file, self.ayear_file)
    #     cmd = f'cdo yearmonmean {self.amon_file} {self.ayear_file}'
    #     print(cmd)
    #     subprocess.call(cmd, shell=True)
    #     assert os.path.exists(self.ayear_file), self.ayear_file

    # @classmethod
    # def setUpClass(cls):
    #     cls.base = os.path.join(os.environ['ST_HOME'], 'data')
    #     cls.amon_file = get_example_data_loc()
    #     cls.ayear_file = os.path.join(
    #         os.path.split(cls.amon_file.replace('Amon', 'AYear'))[0], 'merged.nc'
    #     )

    # def setUp(self):
    #     self.from_amon_to_ayear()
    #     super().setUp()

    # def test_read_data(self):
    #     data_set = oet.synda_files.format_synda.load_glob(self.ayear_file)

    # def test_make_map(self):
    #     data_set = oet.analyze.cmip_handler.read_ds(os.path.split(self.ayear_file)[0])
    #     oet.analyze.cmip_handler.MapMaker(data_set=data_set).plot_all(2)
    #     plt.clf()

    # def test_map_maker_time_series(self):
    #     data_set = oet.analyze.cmip_handler.read_ds(os.path.split(self.ayear_file)[0])
    #     oet.analyze.cmip_handler.MapMaker(data_set=data_set).time_series()
    #     plt.clf()

    # def test_apply_relative_units(self, unit='relative'):
    #     data_set = oet.analyze.cmip_handler.read_ds(
    #         os.path.split(self.ayear_file)[0], condition_kwargs=dict(unit=unit)
    #     )
    #     mm = oet.analyze.cmip_handler.MapMaker(data_set=data_set)

    # def test_apply_std_unit(self):
    #     self.test_apply_relative_units(unit='std')

    # @classmethod
    # def tearDownClass(cls) -> None:
    #     os.remove(cls.ayear_file)
    #     return super().tearDownClass()
