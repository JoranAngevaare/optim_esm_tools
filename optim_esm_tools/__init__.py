# -*- coding: utf-8 -*-
__version__ = '1.1.0'
__author__ = 'Joran R. Angevaare'

from . import utils
from . import config
from . import analyze
from . import synda_files
from . import _test_utils
from . import plotting

# Forward some of the essential tools to this main
from .analyze.cmip_handler import read_ds
from .analyze.io import load_glob
from .plotting.map_maker import MapMaker
