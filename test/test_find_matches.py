# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
from optim_esm_tools._test_utils import get_example_data_loc


class TestMatches(unittest.TestCase):
    def test_find_matches(self):
        path = get_example_data_loc()
        assert os.path.exists(path), f'no {path}'
        base = path.split('ScenarioMIP')[0]
        head, tail = os.path.split(path)
        kw = oet.analyze.find_matches.folder_to_dict(head)
        matches = oet.analyze.find_matches.find_matches(
            base=base,
            required_file=tail,
            **kw,
        )
        assert len(matches), dict(
            base=base,
            required_file=tail,
            **kw,
        )
