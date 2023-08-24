import tempfile
import optim_esm_tools as oet
import os
from unittest import TestCase


class TestCombineVariables(TestCase):
    def test_merge_two(self, nx=5, ny=20, is_match=(True, True)):
        with tempfile.TemporaryDirectory() as temp_dir:
            # temp_dir = '/home/aangevaare/software/paper_oet/notebooks/'
            kw = dict(len_x=nx, len_y=ny, len_time=20, add_nans=False)
            names = list('abcefg')[: len(is_match)]
            paths = [os.path.join(temp_dir, f'{x}.nc') for x in names]
            post_path = []
            for name, path in zip(names, paths):
                ds = oet._test_utils.complete_ds(**kw)
                ds = ds.rename(dict(var=name))
                assert name in ds

                ds.attrs.update(dict(file=path, variable_id=name))
                ds.to_netcdf(path)
                head, tail = os.path.split(path)
                post_ds = oet.read_ds(head, _file_name=tail, _skip_folder_info=True)
                post_path.append(post_ds.attrs['file'])

            merger = oet.analyze.combine_variables.VariableMerger(
                paths=[p for p, m in zip(post_path, is_match) if m],
                other_paths=[p for p, m in zip(post_path, is_match) if not m],
                merge_method='logical_or',
            )
            merged = merger.squash_sources()
            for n, m in zip(names, is_match):
                if m:
                    assert n in merged.data_vars
            oet.analyze.combine_variables.change_plt_table_height()
            merger.make_fig(merged)
            return merger

    def test_merge_three(self):
        merger = self.test_merge_two(is_match=(True, True, False))
        assert merger.other_paths
