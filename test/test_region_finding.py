# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools._test_utils
from optim_esm_tools.analyze import region_finding
import tempfile
import os


class Work(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for data_name in ['ssp585', 'piControl']:
            cls.get_path(data_name)

    @staticmethod
    def get_path(data_name, refresh=True):
        path = optim_esm_tools._test_utils.get_file_from_pangeo(
            data_name, refresh=refresh
        )
        year_path = optim_esm_tools._test_utils.year_means(path, refresh=refresh)
        assert year_path
        assert os.path.exists(year_path)
        return year_path

    def test_max_region(self, make='MaxRegion', new_opt=None):
        cls = getattr(region_finding, make)
        extra_opt = dict(
            time_series_joined=True,
            scatter_medians=True,
            percentiles=50,
            search_kw=dict(required_file=os.path.split(self.get_path('ssp585'))[1]),
        )
        if new_opt:
            extra_opt.update(new_opt)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_kw = dict(
                save_in=temp_dir,
                sub_dir=None,
                file_types=('png',),
                dpi=25,
                skip=False,
            )
            head, tail = os.path.split(self.get_path('ssp585', refresh=False))
            region_finder = cls(
                path=head,
                read_ds_kw=dict(_file_name=tail),
                transform=True,
                save_kw=save_kw,
                extra_opt=extra_opt,
            )
            region_finder.show = False
            region_finder.workflow()
            return region_finder

    def test_max_region_wo_time_series(self):
        self.test_max_region('MaxRegion', new_opt=dict(time_series_joined=False))

    def test_percentiles(self):
        self.test_max_region('Percentiles', new_opt=dict(time_series_joined=False))

    def test_percentiles_history(self):
        region_finder = self.test_max_region('PercentilesHistory')
        with self.assertRaises(RuntimeError):
            # We only have piControl (so this should fail)!
            region_finder.find_historical('historical')

    def test_percentiles_product(self):
        self.test_max_region('ProductPercentiles')

    def test_local_history(self):
        self.test_max_region('LocalHistory')
