from optim_esm_tools._test_utils import get_file_from_pangeo
import pytest


@pytest.mark.parametrize('scenario', ['ssp585', 'piControl', 'historical'])
def test_download(scenario):
    get_file_from_pangeo(scenario)
