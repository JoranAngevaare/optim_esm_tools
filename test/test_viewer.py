# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os


class TestViewer(unittest.TestCase):
    def test_basic(self, **kw):
        path = os.path.join(os.environ.get('ST_HOME', '.'), 'data')
        viewer = oet.synda_files.synda_files.SyndaViewer(path, **kw)
        viewer.tree().show()

    def test_max_depth(self):
        self.test_basic(max_depth=3)

    def test_show_files(self):
        self.test_basic(show_files=1)

    def test_count_files(self):
        self.test_basic(count_files=1)
