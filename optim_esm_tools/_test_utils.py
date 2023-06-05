# -*- coding: utf-8 -*-
import os

EXMPLE_DATA_SET = 'CMIP6/ScenarioMIP/CCCma/CanESM5/ssp585/r3i1p2f1/Amon/tas/gn/v20190429/tas_Amon_CanESM5_ssp585_r3i1p2f1_gn_201501-210012.nc'


def get_example_data_loc():
    return os.path.join(os.environ.get('ST_HOME'), 'data', EXMPLE_DATA_SET)


def synda_test_available():
    """Check if we can run a synda-dependent test"""
    synda_home = os.environ.get('ST_HOME')
    if synda_home is None:
        print(f'No ST_HOME')
        return False
    if not os.path.exists(get_example_data_loc()):
        print(f'No {get_example_data_loc()}')
        return False
    return True
