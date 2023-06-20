# -*- coding: utf-8 -*-
import unittest
import optim_esm_tools._test_utils
from optim_esm_tools.analyze import region_finding
import tempfile
import pytest
import os


class Work(unittest.TestCase):
    def test(self):
        for data_name in ['ssp585', 'piControl']:
            self.get_path(data_name)

    @staticmethod
    def get_path(data_name, refresh=True):
        path = optim_esm_tools._test_utils.get_file_from_pangeo(
            data_name, refresh=refresh
        )
        year_path = optim_esm_tools._test_utils.year_means(path, refresh=refresh)
        assert year_path
        assert os.path.exists(year_path)
        return year_path

    @pytest.mark.paramerize(
        'make', ['region_finding', 'Percentiles', 'PercentilesHistory']
    )
    def test_build_plots(
        self,
        make='MaxRegion',
    ):
        cls = getattr(region_finding, make)
        extra_opt = dict(time_series_joined=True, scatter_medians=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            print(make)
            save_kw = dict(
                save_in=temp_dir,
                sub_dir=None,
                file_types=('png',),
                dpi=25,
                skip=False,
            )
            head, tail = os.path.split(self.get_path('ssp585', refresh=False))
            r = cls(
                path=head,
                read_ds_kw=dict(_file_name=tail),
                transform=True,
                save_kw=save_kw,
                extra_opt=extra_opt,
            )
            r.show = False
            r.workflow()
