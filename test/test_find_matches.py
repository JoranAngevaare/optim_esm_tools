# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
from optim_esm_tools._test_utils import (
    synda_test_available,
    get_example_data_loc,
    get_synda_loc,
)


@unittest.skipIf(not synda_test_available(), 'synda data not available')
class TestMatches(unittest.TestCase):
    def test_find_matches(self):
        head = os.path.join(get_synda_loc(), 'CMIP6')
        path = get_example_data_loc()
        kw = oet.analyze.find_matches.folder_to_dict(path)
        assert len(
            oet.analyze.find_matches.find_matches(
                base=head,
                required_file=os.path.split(path)[1],
                **kw,
            )
        )
