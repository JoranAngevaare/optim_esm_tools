# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
from optim_esm_tools._test_utils import get_example_data_loc


class TestMatches(unittest.TestCase):
    def test_find_matches(self):
        path = get_example_data_loc()
        head = path.split('ScenarioMIP')[0]
        kw = oet.analyze.find_matches.folder_to_dict(path)
        assert len(
            oet.analyze.find_matches.find_matches(
                base=head,
                required_file=os.path.split(path)[1],
                **kw,
            )
        )
