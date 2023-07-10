# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools as oet
import os
import xarray as xr
import numpy as np


class Area(unittest.TestCase):
    ext_var = 'areacella'

    @classmethod
    def setUpClass(cls):
        path = cls.get_path('ssp585')
        head, tail = os.path.split(path)
        ds = oet.analyze.cmip_handler.read_ds(
            head, _file_name=tail, add_area=False, apply_transform=False
        )
        cls.ds = ds
        cls.dummy_path = path.replace('tas', cls.ext_var)
        cls.write_dummy_area(cls, cls.dummy_path)

    @staticmethod
    def get_path(data_name, refresh=False):
        path = oet._test_utils.get_file_from_pangeo(data_name, refresh=refresh)
        year_path = oet._test_utils.year_means(path, refresh=refresh)
        assert year_path
        assert os.path.exists(year_path)
        return year_path

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.dummy_path):
            print(f'rm {cls.dummy_path}')
            os.remove(cls.dummy_path)

    def write_dummy_area(self, dummy_path):
        ds = self.ds
        dummy = xr.DataArray(
            np.ones(ds['lon'].values.shape), dims=ds['lon'].dims, name=self.ext_var
        )
        os.makedirs(os.path.split(dummy_path)[0], exist_ok=True)
        dummy.to_netcdf(dummy_path)
        assert os.path.exists(dummy_path)

    def _test(self, method, **kw):
        ds = self.ds.copy()
        oet.analyze.query_metric.add_area_to_ds(ds, method=method, **kw)

    def test_local(self):
        kw = dict(required_file=os.path.split(self.dummy_path)[1])
        self._test('local', **kw)

    def test_pangeo(self):
        kw = dict()
        self._test('pangeo', **kw)

    def test_local_fuzzy(self):
        kw = dict(required_file=os.path.split(self.dummy_path)[1])
        self._test('local_fuzzy', **kw)

    def test_pangeo_fuzzy(self):
        kw = dict()
        self._test('pangeo_fuzzy', **kw)

    def test_brute_force(self):
        kw = dict()
        self._test('brute_force', **kw)
