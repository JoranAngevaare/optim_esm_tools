import unittest
import optim_esm_tools as oet
import os
import glob

class TestMapMaker(unittest.TestCase):
    example_data_set = '/data/CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc'
    
    def from_amon_to_ayear(self):
        amon_file = os.path.join(self.base, self.example_data_set)
        ayear_file = amon_file.replace('Amon', 'AYear')
        if os.path.exists(ayear_file):
            return
        import cdo

        cdo.Cdo().yearmonmean(amon_file,ayear_file)
        assert os.path.exists(ayear_file)
    
    def setUp(self):
        self.base = os.path.join(os.environ['ST_HOME'], 'data')
        self.from_amon_to_ayear()
        
        # self.dataset = oet.synda_files.format_synda.load_glob()