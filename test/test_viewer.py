# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os


class TestViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = oet._test_utils.get_file_from_pangeo('ssp585', refresh=False)
        while os.path.split(path)[1] != 'data':
            path = os.path.split(path)[0]
        cls.base = path

    def test_basic(self, **kw):
        path = self.base
        viewer = oet.synda_files.synda_files.SyndaViewer(path, **kw)
        viewer.tree().show()

    def test_max_depth(self):
        self.test_basic(max_depth=3)

    def test_show_files(self):
        self.test_basic(show_files=1)

    def test_count_files(self):
        self.test_basic(count_files=1)
