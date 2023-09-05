import tempfile
import unittest

import matplotlib.pyplot as plt
import pytest

import optim_esm_tools as oet


class TestUtils(unittest.TestCase):
    def test_make_dummy_fig(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            print('created temporary directory', temp_dir)
            oet.utils.setup_plt()
            plt.scatter([1, 2], [3, 4])
            plt.legend(**oet.utils.legend_kw())
            plt.xlabel(oet.utils.string_to_mathrm('Some example x'))
            try:
                oet.utils.save_fig('bla', save_in=temp_dir)
            except RuntimeError as e:
                print(f'Most likely cause is that latex did not install, ran into {e}')

    def test_print_version(self):
        oet.utils.print_versions(['numpy', 'optim_esm_tools', 'somethingsomething'])

    def test_str_ops(self):
        assert oet.utils.mathrm('bla') == oet.utils.string_to_mathrm('bla')
        assert oet.utils.to_str_tuple(('bla',)) == ('bla',)


class TestTimed:
    @staticmethod
    def _timeing_decorator(**kw):
        @oet.utils.timed(**kw)
        def foo(a):
            print(a)

        foo(f'Test {kw}')

    @pytest.mark.parametrize(
        'seconds,report,args_max,fmt',
        [
            [0, 'print', 1, '.1f'],
            [0, 'info', -1, '.1e'],
            [0, 'debug', 1, '.06'],
            [0, 'warning', 1, '.1f'],
        ],
    )
    def test_timing(self, seconds, report, args_max, fmt):
        kw = dict(seconds=seconds, _report=report, _args_max=args_max, _fmt=fmt)
        self._timeing_decorator(**kw)


class TestDepricated(unittest.TestCase):
    def test(self):
        @oet.utils.deprecated
        def bla(a):
            return a

        with self.assertWarns(DeprecationWarning):
            bla(1)
