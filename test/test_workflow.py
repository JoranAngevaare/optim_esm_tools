import unittest
import optim_esm_tools as oet
import os
import glob
import matplotlib.pyplot as plt
import subprocess

class TestMapMaker(unittest.TestCase):
    example_data_set = 'CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc'

    def from_amon_to_ayear(self):

        if os.path.exists(self.ayear_file):
            return
        import cdo

        os.makedirs(os.path.split(self.ayear_file)[0], exist_ok=1)
        # cdo.Cdo().yearmonmean(self.amon_file, self.ayear_file)
        cmd = f'cdo yearmonmean {self.amon_file} {self.ayear_file}'
        print(cmd)
        # os.system(cmd)
        subprocess.call(cmd, shell=True)  
        assert os.path.exists(self.ayear_file), self.ayear_file

    def setUp(self):
        self.base = os.path.join(os.environ['ST_HOME'], 'data')
        self.amon_file = os.path.join(self.base, self.example_data_set)
        self.ayear_file = os.path.join(os.path.split(self.amon_file.replace('Amon', 'AYear'))[0], 'merged.nc')
        self.from_amon_to_ayear()
        super().setUp()

    def test_read_data(self):
        data_set = oet.synda_files.format_synda.load_glob(self.ayear_file)
    
    def test_make_map(self):
        data_set = oet.analyze.cmip_handler.read_ds(os.path.split(self.ayear_file)[0])
        oet.analyze.cmip_handler.MapMaker(data_set=data_set).plot_all(2)
        plt.clf()

    def tearDown(self) -> None:
        os.remove(self.ayear_file)
        return super().tearDown()