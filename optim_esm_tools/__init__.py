__version__ = '3.1.0'
__author__ = 'Joran R. Angevaare'

from .plotting import plot_utils
from . import utils
from . import config
from . import analyze
from . import region_finding
from . import _test_utils
from . import plotting

# Forward some of the essential tools to this main
from .analyze.cmip_handler import read_ds
from .analyze.io import load_glob
from .utils import print_versions
from .config import get_logger
