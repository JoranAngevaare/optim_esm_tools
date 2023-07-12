# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
from optim_esm_tools._test_utils import get_example_data_loc


class TestMatches(unittest.TestCase):
    def test_find_matches(self):
        path = get_example_data_loc()
        if not os.path.exists(path):
            to_dir = os.path.split(path)[0]
            os.makedirs(to_dir, exist_ok=True)
            dummy_file = 'hello.nc'
            path = os.path.join(to_dir, dummy_file)
            with open(path, 'a') as f:
                f.write('hello world')
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
