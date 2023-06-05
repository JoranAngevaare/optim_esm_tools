import optim_esm_tools as oet
import unittest
import tempfile
import matplotlib.pyplot as plt


@unittest.skip('LATEX IS BROKEN :(')
class TestUtils(unittest.TestCase):
    def test_setup_plt(self):
        oet.utils.setup_plt()

    def test_make_dummy_fig(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            print('created temporary directory', temp_dir)
            oet.utils.setup_plt()
            plt.scatter([1,2], [3,4])
            plt.legend(**oet.utils.legend_kw())
            plt.xlabel(oet.utils.string_to_mathrm('Some example x'))
            oet.utils.save_fig('bla', save_in=temp_dir)

    def test_print_version(self):
        oet.utils.print_versions(['numpy', 'optim_esm_tools'])
