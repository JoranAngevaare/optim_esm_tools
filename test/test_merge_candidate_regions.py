import unittest
import numpy as np
import xarray as xr
from optim_esm_tools.analyze.merge_candidate_regions import *
from optim_esm_tools.analyze.merge_candidate_regions import (
    _should_merge_adjacent,
    _frac_overlap_nb,
)
import optim_esm_tools
from unittest.mock import MagicMock, patch


class TestShouldMerge(unittest.TestCase):
    def setUp(self):
        # Prepare test data
        self.grid1 = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        self.grid2 = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        self.overlapping_grid = np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        self.xr_grid1 = xr.DataArray(self.grid1)
        self.xr_grid2 = xr.DataArray(self.grid2)
        self.xr_overlapping_grid = xr.DataArray(self.overlapping_grid)

    def test_should_merge_overlap(self):
        # Test merging based on overlapping
        with patch(
            'optim_esm_tools.analyze.merge_candidate_regions._frac_overlap_nb',
            return_value=0.6,
        ):
            self.assertTrue(
                should_merge(
                    self.xr_grid1,
                    self.xr_overlapping_grid,
                    min_frac_overlap=0.5,
                ),
            )

    def test_should_merge_adjacent(self):
        # Test merging based on adjacency
        with patch(
            'optim_esm_tools.analyze.merge_candidate_regions._frac_overlap_nb',
            return_value=0.3,
        ):
            with patch(
                'optim_esm_tools.analyze.merge_candidate_regions._should_merge_adjacent',
                return_value=True,
            ):
                self.assertTrue(
                    should_merge(
                        self.grid1,
                        self.grid2,
                        min_border_frac=0.05,
                        min_n_adjacent=100,
                    ),
                )

    def test_should_not_merge(self):
        # Test when grids do not satisfy merge criteria
        with patch(
            'optim_esm_tools.analyze.merge_candidate_regions._frac_overlap_nb',
            return_value=0.3,
        ):
            with patch(
                'optim_esm_tools.analyze.merge_candidate_regions._should_merge_adjacent',
                return_value=False,
            ):
                self.assertFalse(
                    should_merge(
                        self.grid1,
                        self.grid2,
                        min_frac_overlap=0.5,
                        min_n_adjacent=100,
                    ),
                )

    def test_invalid_input(self):
        # Test with invalid input types
        with self.assertRaises(TypeError):
            should_merge(self.grid1, 'invalid_type')

    def test_fraction_overlap(self):
        # Test the fraction overlap calculation
        overlap = _frac_overlap_nb(self.grid1, self.overlapping_grid)
        self.assertAlmostEqual(overlap, 1)

    def test_should_merge_adjacent_function(self):
        # Test the adjacency calculation
        self.assertTrue(
            _should_merge_adjacent(
                self.grid1,
                self.grid2,
                min_border_frac=0.05,
                min_n_adjacent=2,
            ),
        )

        self.assertFalse(
            _should_merge_adjacent(
                self.grid1,
                self.grid2,
                min_border_frac=1,
                min_n_adjacent=100,
            ),
        )


def create_mock_dataset(global_mask_values):
    """
    Create a mock xarray Dataset with a `global_mask` variable.
    """
    data = xr.Dataset(
        {
            'global_mask': (
                ('lat_mask', 'lon_mask'),
                np.array(global_mask_values, dtype=bool),
            ),
        },
    )
    return data


class TestMerger(unittest.TestCase):
    @staticmethod
    def get_dummy_common(mock_ds):
        common_dummy = oet.analyze.xarray_tools.reverse_name_mask_coords(mock_ds.copy())
        common_dummy.attrs.update(variable_id='stat')
        cell_area = common_dummy.global_mask.copy().astype(np.int64)
        cell_area[:] = 1
        common_dummy['cell_area'] = cell_area
        return common_dummy

    @patch('optim_esm_tools.get_logger')
    @patch('optim_esm_tools.load_glob')
    def test_merger_initialization(self, mock_load_glob, mock_get_logger):
        """
        Test if the Merger initializes correctly.
        """
        mock_ds = create_mock_dataset([[1, 0], [0, 1]])
        mock_load_glob.return_value = mock_ds
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        def mock_pass_criteria(**kwargs):
            return True

        def mock_summary_calculation(**kwargs):
            return {'stat': 1}

        common_dummy = self.get_dummy_common(mock_ds)

        merger = Merger(
            pass_criteria=mock_pass_criteria,
            summary_calculation=mock_summary_calculation,
            data_sets=[mock_ds],
            common_mother=common_dummy,
            common_pi=common_dummy,
        )
        self.assertIsInstance(merger, Merger)
        self.assertEqual(len(merger.data_sets), 1)

    @patch('optim_esm_tools.get_logger')
    @patch('optim_esm_tools.load_glob')
    def test_merger_set_passing_largest_data_sets_first(
        self,
        mock_load_glob,
        mock_get_logger,
    ):
        """
        Test if datasets are sorted correctly based on the pass criteria and size.
        """
        mock_ds1 = create_mock_dataset([[1, 0], [0, 1]])
        mock_ds2 = create_mock_dataset([[1, 1], [1, 1]])
        mock_load_glob.return_value = mock_ds1

        def mock_pass_criteria(**kwargs):
            return True

        def mock_summary_calculation(**kwargs):
            return {'stat': 1}

        common_dummy = self.get_dummy_common(mock_ds1)
        merger = Merger(
            pass_criteria=mock_pass_criteria,
            summary_calculation=mock_summary_calculation,
            data_sets=[mock_ds1, mock_ds2],
            common_mother=common_dummy,
            common_pi=common_dummy,
        )

        # Set passing and ensure the larger dataset comes first
        merger.set_passing_largest_data_sets_first()
        self.assertEqual(merger.data_sets[0], mock_ds2)

    @patch('optim_esm_tools.get_logger')
    @patch('optim_esm_tools.load_glob')
    def test_merger_merge_datasets(self, mock_load_glob, mock_get_logger):
        """
        Test the merge_datasets method to ensure it merges datasets correctly.
        """
        mock_ds1 = create_mock_dataset([[1, 0], [0, 1]])
        mock_ds2 = create_mock_dataset([[0, 1], [1, 0]])
        mock_ds3 = create_mock_dataset([[1, 1], [1, 1]])
        mock_load_glob.return_value = mock_ds1

        def mock_pass_criteria(**kwargs):
            return True

        def mock_summary_calculation(**kwargs):
            return {'stat': 1}

        common_dummy = self.get_dummy_common(mock_ds1)
        merger = Merger(
            pass_criteria=mock_pass_criteria,
            summary_calculation=mock_summary_calculation,
            data_sets=[mock_ds1, mock_ds2, mock_ds3],
            common_mother=common_dummy,
            common_pi=common_dummy,
        )

        result = merger.merge_datasets()
        self.assertTrue(len(result) > 0)
        self.assertTrue('stats' in result[0])
        self.assertIsInstance(result[0]['ds'], xr.Dataset)

    @patch('optim_esm_tools.get_logger')
    @patch('optim_esm_tools.load_glob')
    def test_merger_group_to_first(self, mock_load_glob, mock_get_logger):
        """
        Test _group_to_first logic.
        """
        mock_ds1 = create_mock_dataset([[1, 0], [0, 1]])
        mock_ds2 = create_mock_dataset([[0, 1], [1, 0]])
        mock_load_glob.return_value = mock_ds1

        def mock_pass_criteria(**kwargs):
            return True

        def mock_summary_calculation(**kwargs):
            return {'stat': 1}

        common_dummy = self.get_dummy_common(mock_ds1)

        merger = Merger(
            pass_criteria=mock_pass_criteria,
            summary_calculation=mock_summary_calculation,
            data_sets=[mock_ds1, mock_ds2],
            common_mother=common_dummy,
            common_pi=common_dummy,
        )

        group_result = merger._group_to_first([mock_ds1, mock_ds2])
        self.assertTrue('stats' in group_result)
        self.assertIsInstance(group_result['ds'], xr.Dataset)

    @patch('optim_esm_tools.get_logger')
    @patch('optim_esm_tools.load_glob')
    def test_merger_merge_one_skip_one(self, mock_load_glob, mock_get_logger):
        """
        merging logic of merging one and failing one.
        """
        mock_ds1 = create_mock_dataset(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
        mock_ds2 = create_mock_dataset(
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
        mock_ds3 = create_mock_dataset(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
        )
        mock_load_glob.return_value = mock_ds1

        def mock_pass_criteria(**kwargs):
            return True

        def mock_summary_calculation(**kwargs):
            return {'stat': 1}

        common_dummy = self.get_dummy_common(mock_ds1)

        merger = Merger(
            pass_criteria=mock_pass_criteria,
            summary_calculation=mock_summary_calculation,
            data_sets=[mock_ds1, mock_ds2, mock_ds3],
            common_mother=common_dummy,
            common_pi=common_dummy,
        )
        res = merger.merge_datasets()

        assert len(res) == 2

    @patch('optim_esm_tools.get_logger')
    @patch('optim_esm_tools.load_glob')
    def test_merger_pass_one_fail_one(self, mock_load_glob, mock_get_logger):
        """
        test if passfunc does what it is supposed to do
        """
        mock_ds1 = create_mock_dataset(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
        )
        mock_ds2 = create_mock_dataset(
            [
                [0, 1, 1],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
        mock_ds3 = create_mock_dataset(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
            ],
        )
        mock_load_glob.return_value = mock_ds1

        def mock_pass_criteria(passes, **kw):
            return passes

        def mock_summary_calculation(mask, **kwargs):
            return dict(
                passes=(mask.values[:, 0].sum() > 1) & (mask.values[:, 1].sum() == 0),
            )

        common_dummy = self.get_dummy_common(mock_ds1)

        merger = MergerCached(
            pass_criteria=mock_pass_criteria,
            summary_calculation=mock_summary_calculation,
            data_sets=[mock_ds1, mock_ds2, mock_ds3],
            common_mother=common_dummy,
            common_pi=common_dummy,
        )
        res = merger.merge_datasets()

        assert len(res) == 1
        xr.testing.assert_allclose(
            res[0]['ds']['global_mask'],
            create_mock_dataset(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                ],
            )['global_mask'],
        )


if __name__ == '__main__':
    unittest.main()
