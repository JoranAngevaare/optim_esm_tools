# -*- coding: utf-8 -*-
import numpy as np
from optim_esm_tools.analyze.xarray_tools import mask2d_to_xy_slice, _mask2d_to_xy_slice

# TODO write propper hypothesis test
def test_xarray_2d_slicer():
    random_mask = np.random.rand(100,100)>0.5
    np.array_equal(mask2d_to_xy_slice(random_mask), _mask2d_to_xy_slice(random_mask))
