import inspect
import os
import tempfile
import unittest

import numpy as np

import optim_esm_tools._test_utils
from optim_esm_tools.analyze import region_finding
from optim_esm_tools.analyze.cmip_handler import read_ds
from optim_esm_tools.analyze.io import load_glob


class Work(unittest.TestCase):
    """Note of caution!

    _CACHE_TRUE=True can lead to funky behavior!
    """

    @classmethod
    def setUpClass(cls):
        for data_name in ['ssp585', 'piControl']:
            cls.get_path(data_name)

    @staticmethod
    def get_path(data_name, refresh=True):
        return optim_esm_tools._test_utils.get_path_for_ds(data_name, refresh=refresh)

    @staticmethod
    def _post_test_regions_no_overlap(masks):
        """Make sure that no gridpi."""
        if not len(masks):
            return
        n_masks = masks[0].copy().astype(int)
        n_masks[:] = 0
        for mask_i in masks:
            n_masks[mask_i] += 1
        assert np.max(n_masks) <= 1

    def test_max_region(self, make='MaxRegion', new_opt=None, skip_save=True):
        # sourcery skip: dict-assign-update-to-union
        cls = getattr(region_finding, make)
        file_path = self.get_path('ssp585', refresh=False)

        head, tail = os.path.split(file_path)
        extra_opt = dict(
            time_series_joined=True,
            scatter_medians=True,
            percentiles=50,
            search_kw=dict(required_file=tail),
            iterable_range=dict(
                percentiles=np.linspace(99.99, 85, 2),
                product_percentiles=np.linspace(99.9, 85, 2),
                n_times_historical=np.linspace(8, 2.5, 2),
            ),
        )
        #
        # skip: no-conditionals-in-tests
        if new_opt:
            extra_opt.update(new_opt)
        with tempfile.TemporaryDirectory() as temp_dir:
            save_kw = dict(
                save_in=temp_dir,
                sub_dir=None,
                file_types=('png',),
                dpi=25,
                skip=skip_save,
            )
            data_set = read_ds(
                head,
                _file_name=tail,
                _cache=os.environ.get('_CACHE_TRUE', 0),
                add_history=False,
            )
            cls_kw = dict(
                data_set=data_set,
                save_kw=save_kw,
                extra_opt=extra_opt,
            )
            signature = inspect.getfullargspec(cls.__init__)

            if (
                'data_set_pic' in signature.args
                or 'data_set_pic' in signature.kwonlyargs
            ):
                pi_base, pi_file = os.path.split(
                    self.get_path('piControl', refresh=False),
                )
                cls_kw['data_set_pic'] = read_ds(
                    pi_base,
                    _file_name=pi_file,
                    max_time=None,
                    min_time=None,
                    add_history=False,
                )
            region_finder = cls(**cls_kw)
            region_finder.show = False

            masks, _ = region_finder.get_masks()
            self._post_test_regions_no_overlap(masks)
            return region_finder

    def test_max_region_wo_time_series(self):
        self.test_max_region('MaxRegion', new_opt=dict(time_series_joined=False))

    def test_percentiles(self):
        self.test_max_region('Percentiles', new_opt=dict(time_series_joined=False))

    def test_percentiles_weighted(self):
        self.test_max_region('Percentiles', new_opt=dict(cluster_method='weighted'))

    def test_start_end_continous(self):
        self.test_max_region('IterStartEnd', new_opt=dict(force_continuity=True))

    def test_percentiles_product(self):
        self.test_max_region('ProductPercentiles', skip_save=False)

    def test_local_history(self):
        self.test_max_region('LocalHistory')

    def test_percentiles_product_weighted(self):
        self.test_max_region(
            'ProductPercentiles',
            new_opt=dict(cluster_method='weighted'),
        )

    def test_error_message(self, make='MaxRegion'):
        cls = getattr(region_finding, make)
        file_path = self.get_path('ssp585', refresh=False)
        head, tail = os.path.split(file_path)
        ds = read_ds(
            head,
            _file_name=tail,
        )
        region = cls(data_set=ds)
        with self.assertRaises(ValueError):
            region.check_shape(ds['cell_area'].T)  # type: ignore

    def test_iter_product_percentiles(self):
        self.test_max_region('IterProductPercentiles')

    def test_iter_local_history(self):
        self.test_max_region('IterLocalHistory')

    def test_iter_start_end_history(self):
        self.test_max_region(
            'IterStartEnd',
            new_opt=dict(
                iter_range=dict(product_percentiles=np.linspace(99.9, 85, 41)),
            ),
        )

    def test_iter_percentiles(self):
        self.test_max_region('IterPercentiles')

    def test_iter_raises(self):
        with self.assertRaises(NotImplementedError):
            self.test_max_region(
                'IterProductPercentiles',
                new_opt=dict(cluster_method='weighted'),
            )
        with self.assertRaises(NotImplementedError):
            self.test_max_region(
                'IterPercentiles',
                new_opt=dict(cluster_method='weighted'),
            )
        with self.assertRaises(NotImplementedError):
            self.test_max_region(
                'IterStartEnd',
                new_opt=dict(cluster_method='weighted'),
            )

    def test_mask_all(self):
        self.test_max_region('MaskAll')

    def test_mask_asia(self):
        self.test_max_region('Asia')

    def test_mask_medeteranian(self):
        self.test_max_region('Medeteranian')
