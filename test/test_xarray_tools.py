# -*- coding: utf-8 -*-
import pytest


# TODO write propper hypothesis test
def test_xarray_2d_slicer():
    import numpy as np
    from optim_esm_tools.analyze.xarray_tools import (
        mask2d_to_xy_slice,
        _mask2d_to_xy_slice,
    )

    random_mask = np.random.rand(100, 100) > 0.5
    np.array_equal(mask2d_to_xy_slice(random_mask), _mask2d_to_xy_slice(random_mask))


@pytest.mark.parametrize('detrend_type', ['linear', 'constant'])
def test_detrend(detrend_type):
    import optim_esm_tools as oet
    import numpy as np

    ds = oet._test_utils.minimal_xr_ds(add_nans=False)
    arange = np.arange(len(ds['var'].values.flatten()))
    ds['var'] = (
        ds['var'].dims,
        (arange + len(arange) + np.sin(arange)).reshape(ds['var'].shape),
    )
    kw = dict(data_array=ds['var'], dimension='time', detrend_type=detrend_type)
    detrend_xrft = oet.analyze.xarray_tools.detrend(use_xrft=False, **kw)
    detrend = oet.analyze.xarray_tools.detrend(use_xrft=True, **kw)
    assert np.all(np.isclose(detrend_xrft.values, detrend.values))
